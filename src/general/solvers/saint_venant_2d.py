"""Verified finite-volume solver for the two-dimensional shallow-water equations.

The conserved state is ``U = (h, hu, hv)`` on a Cartesian cell-centred grid.
Rusanov fluxes are combined with hydrostatic reconstruction so a constant free
surface over non-flat bed elevation is preserved. A conservative draining
limiter scales outward numerical fluxes before the update, preventing negative
depth without relying on a mass-adding depth floor.

Boundary modes:

* x ``inflow_outflow``: prescribed unit discharge at the left, zero-gradient
  open boundary at the right.
* x ``inflow_stage``: prescribed unit discharge at the left and a measured
  water-surface elevation at the right.
* y ``wall``: reflecting, free-slip walls.
* x or y ``periodic``: periodic verification and benchmark domains.

``momentum_source`` is an optional deterministic forcing hook used by
manufactured-solution verification. Production scenarios leave it unset.

Units are metres and minutes. Manning's n must use the corresponding
minutes/metres convention (conventional SI n divided by 60).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from general.solvers.contract import Domain2D, Scenario, SimulationResult
from general.solvers.infiltration import (
    green_ampt_step,
    initial_cumulative_infiltration,
    prepare_green_ampt,
)


L = 10.0
W = 5.0
T_final = 30.0
S0x = 0.05
S0y = 0.0
MANNING_N_SECONDS = 0.05
n0 = MANNING_N_SECONDS / 60.0
g = 35316.0
DRY_TOL = 1e-10
POSITIVITY_TOL = 1e-12
CFL = 0.45


def r(t):
    """Default uniform rainfall source rate in m/min."""
    return 0.1


def _velocity(h, momentum):
    velocity = np.zeros_like(momentum, dtype=float)
    np.divide(momentum, h, out=velocity, where=h > DRY_TOL)
    return velocity


def _physical_flux_x(h, hu, hv):
    u = _velocity(h, hu)
    return hu, hu * u + 0.5 * g * h**2, hv * u


def _physical_flux_y(h, hu, hv):
    v = _velocity(h, hv)
    return hv, hu * v, hv * v + 0.5 * g * h**2


def _wave_speed(h, momentum):
    return np.abs(_velocity(h, momentum)) + np.sqrt(g * np.maximum(h, 0.0))


def _record_times(final_time, record_interval):
    count = int(np.floor(final_time / record_interval + 1e-9))
    values = [index * record_interval for index in range(count + 1)]
    if values[-1] < final_time - 1e-9:
        values.append(float(final_time))
    return values


def _grid(length, count):
    step = length / count
    return np.linspace(step / 2, length - step / 2, count), np.full(count, step)


def _field(value, default, shape, name):
    array = np.asarray(default if value is None else value, dtype=float)
    if array.ndim == 0:
        array = np.full(shape, float(array))
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be scalar or contain one finite value per cell")
    return array


def _bed_from_slopes(x, y, slope_x, slope_y):
    """Reconstruct a bed datum from supplied positive-downhill slope fields.

    Explicit ``bed_elevation_m`` is preferred. This compatibility reconstruction
    integrates x slopes from the first x cell and y slopes from the first y cell.
    For a non-integrable slope field the result is path-dependent, which is why
    verification cases always provide bed elevation directly.
    """
    bed_x = np.zeros_like(slope_x)
    if len(x) > 1:
        spacing_x = np.diff(x)[:, None]
        bed_x[1:, :] = -np.cumsum(
            0.5 * (slope_x[:-1, :] + slope_x[1:, :]) * spacing_x,
            axis=0,
        )
    bed_y = np.zeros_like(slope_y)
    if len(y) > 1:
        spacing_y = np.diff(y)[None, :]
        bed_y[:, 1:] = -np.cumsum(
            0.5 * (slope_y[:, :-1] + slope_y[:, 1:]) * spacing_y,
            axis=1,
        )
    return bed_x + bed_y


def _check_finite(time, **states):
    for name, values in states.items():
        if not np.all(np.isfinite(values)):
            bad = np.argwhere(~np.isfinite(values))[0]
            raise FloatingPointError(
                f"{name} became non-finite at t={time:.16g} min, cell={tuple(bad)}"
            )


def _inflow_values(left_inflow, time, ny):
    values = left_inflow(time) if callable(left_inflow) else left_inflow
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(ny, float(values))
    if values.shape != (ny,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(
            f"left_inflow must return a finite non-negative scalar or shape ({ny},)"
        )
    return values


def _cap_dt_for_wetting_source(dt, h, hu, hv, source, dx, dy, cfl):
    """Limit source-created wetting waves using the two-dimensional CFL rate."""
    positive_source = np.maximum(np.asarray(source, dtype=float), 0.0)
    if not np.any(positive_source > 0.0):
        return dt

    def maximum_courant(trial_dt):
        predicted_depth = h + positive_source * trial_dt
        velocity_x = _velocity(predicted_depth, hu)
        velocity_y = _velocity(predicted_depth, hv)
        gravity_speed = np.sqrt(g * predicted_depth)
        spectral_rate = (
            (np.abs(velocity_x) + gravity_speed) / dx[:, None]
            + (np.abs(velocity_y) + gravity_speed) / dy[None, :]
        )
        return float(np.max(trial_dt * spectral_rate))

    candidate = float(dt)
    courant = maximum_courant(candidate)
    if courant <= cfl * (1.0 + 1e-12):
        return candidate
    candidate *= (cfl / courant) ** (2.0 / 3.0)
    courant = maximum_courant(candidate)
    if courant > cfl:
        candidate *= cfl / courant
    return candidate


def _stage_values(downstream_stage_m, time, ny):
    values = (
        downstream_stage_m(time)
        if callable(downstream_stage_m)
        else downstream_stage_m
    )
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(ny, float(values))
    if values.shape != (ny,) or not np.all(np.isfinite(values)):
        raise ValueError(
            "downstream_stage_m must return a finite scalar or "
            f"shape ({ny},)"
        )
    return values


def _cap_dt_at_forcing_breakpoints(dt, time, *forcings):
    for forcing in forcings:
        for breakpoint in getattr(forcing, "breakpoints_min", ()):
            if time + 1e-12 < breakpoint < time + dt - 1e-12:
                dt = float(breakpoint - time)
                break
    return dt


def _extend_x(h, hu, hv, bed, boundary, inflow, downstream_stage=None):
    if boundary == "periodic":
        return tuple(
            np.concatenate([array[-1:, :], array, array[:1, :]], axis=0)
            for array in (h, hu, hv, bed)
        )
    if boundary not in {"inflow_outflow", "inflow_stage"}:
        raise ValueError(
            "boundary_x must be 'inflow_outflow', 'inflow_stage', or 'periodic'"
        )
    if boundary == "inflow_stage":
        ghost_h = np.maximum(downstream_stage - bed[-1, :], 0.0)
        u = _velocity(h[-1, :], hu[-1, :])
        v = _velocity(h[-1, :], hv[-1, :])
        right_h = ghost_h[None, :]
        right_hu = (u * ghost_h)[None, :]
        right_hv = (v * ghost_h)[None, :]
    else:
        right_h = h[-1:, :]
        right_hu = hu[-1:, :]
        right_hv = hv[-1:, :]
    return (
        np.concatenate([h[:1, :], h, right_h], axis=0),
        np.concatenate(
            [2.0 * inflow[None, :] - hu[:1, :], hu, right_hu],
            axis=0,
        ),
        np.concatenate([hv[:1, :], hv, right_hv], axis=0),
        np.concatenate([bed[:1, :], bed, bed[-1:, :]], axis=0),
    )


def _extend_y(h, hu, hv, bed, boundary):
    if boundary == "periodic":
        return tuple(
            np.concatenate([array[:, -1:], array, array[:, :1]], axis=1)
            for array in (h, hu, hv, bed)
        )
    if boundary != "wall":
        raise ValueError("boundary_y must be 'wall' or 'periodic'")
    return (
        np.concatenate([h[:, :1], h, h[:, -1:]], axis=1),
        np.concatenate([hu[:, :1], hu, hu[:, -1:]], axis=1),
        np.concatenate([-hv[:, :1], hv, -hv[:, -1:]], axis=1),
        np.concatenate([bed[:, :1], bed, bed[:, -1:]], axis=1),
    )


def _hydrostatic_states(h_left, mom_x_left, mom_y_left, bed_left,
                        h_right, mom_x_right, mom_y_right, bed_right):
    surface_left = h_left + bed_left
    surface_right = h_right + bed_right
    face_bed = np.maximum(bed_left, bed_right)
    depth_left = np.maximum(0.0, surface_left - face_bed)
    depth_right = np.maximum(0.0, surface_right - face_bed)

    scale_left = np.zeros_like(h_left)
    scale_right = np.zeros_like(h_right)
    np.divide(depth_left, h_left, out=scale_left, where=h_left > DRY_TOL)
    np.divide(depth_right, h_right, out=scale_right, where=h_right > DRY_TOL)
    return (
        depth_left,
        mom_x_left * scale_left,
        mom_y_left * scale_left,
        depth_right,
        mom_x_right * scale_right,
        mom_y_right * scale_right,
    )


def _limited_increments(values, axis, periodic):
    backward = values - np.roll(values, 1, axis=axis)
    forward = np.roll(values, -1, axis=axis) - values
    same_sign = backward * forward > 0.0
    increments = np.where(
        same_sign,
        np.sign(backward) * np.minimum(np.abs(backward), np.abs(forward)),
        0.0,
    )
    if not periodic:
        boundary_slice = [slice(None)] * values.ndim
        boundary_slice[axis] = 0
        increments[tuple(boundary_slice)] = 0.0
        boundary_slice[axis] = -1
        increments[tuple(boundary_slice)] = 0.0
    return increments


def _reconstructed_fields(h, hu, hv, bed, axis, periodic):
    surface = h + bed
    u = _velocity(h, hu)
    v = _velocity(h, hv)
    return tuple(
        (values, _limited_increments(values, axis, periodic))
        for values in (surface, u, v, bed)
    )


def _rusanov_x(
    h,
    hu,
    hv,
    bed,
    boundary,
    inflow,
    spatial_order=1,
    downstream_stage=None,
):
    h_ext, hu_ext, hv_ext, bed_ext = _extend_x(
        h, hu, hv, bed, boundary, inflow, downstream_stage
    )
    center_h_l, center_h_r = h_ext[:-1, :], h_ext[1:, :]
    h_l, h_r = center_h_l.copy(), center_h_r.copy()
    hu_l, hu_r = hu_ext[:-1, :].copy(), hu_ext[1:, :].copy()
    hv_l, hv_r = hv_ext[:-1, :].copy(), hv_ext[1:, :].copy()
    bed_l, bed_r = bed_ext[:-1, :].copy(), bed_ext[1:, :].copy()
    if spatial_order == 2:
        fields = _reconstructed_fields(
            h, hu, hv, bed, axis=0, periodic=boundary == "periodic"
        )
        (surface, ds), (u, du), (v, dv), (bed_cell, db) = fields
        surface_l = surface[:-1, :] + 0.5 * ds[:-1, :]
        surface_r = surface[1:, :] - 0.5 * ds[1:, :]
        bed_l[1:-1, :] = bed_cell[:-1, :] + 0.5 * db[:-1, :]
        bed_r[1:-1, :] = bed_cell[1:, :] - 0.5 * db[1:, :]
        h_l[1:-1, :] = np.maximum(surface_l - bed_l[1:-1, :], 0.0)
        h_r[1:-1, :] = np.maximum(surface_r - bed_r[1:-1, :], 0.0)
        hu_l[1:-1, :] = (u[:-1, :] + 0.5 * du[:-1, :]) * h_l[1:-1, :]
        hu_r[1:-1, :] = (u[1:, :] - 0.5 * du[1:, :]) * h_r[1:-1, :]
        hv_l[1:-1, :] = (v[:-1, :] + 0.5 * dv[:-1, :]) * h_l[1:-1, :]
        hv_r[1:-1, :] = (v[1:, :] - 0.5 * dv[1:, :]) * h_r[1:-1, :]
        if boundary == "periodic":
            periodic_bed_l = bed_cell[-1, :] + 0.5 * db[-1, :]
            periodic_bed_r = bed_cell[0, :] - 0.5 * db[0, :]
            periodic_h_l = np.maximum(
                surface[-1, :] + 0.5 * ds[-1, :] - periodic_bed_l, 0.0
            )
            periodic_h_r = np.maximum(
                surface[0, :] - 0.5 * ds[0, :] - periodic_bed_r, 0.0
            )
            for face in (0, -1):
                bed_l[face, :], bed_r[face, :] = periodic_bed_l, periodic_bed_r
                h_l[face, :], h_r[face, :] = periodic_h_l, periodic_h_r
                hu_l[face, :] = (u[-1, :] + 0.5 * du[-1, :]) * periodic_h_l
                hu_r[face, :] = (u[0, :] - 0.5 * du[0, :]) * periodic_h_r
                hv_l[face, :] = (v[-1, :] + 0.5 * dv[-1, :]) * periodic_h_l
                hv_r[face, :] = (v[0, :] - 0.5 * dv[0, :]) * periodic_h_r
    hs_l, hus_l, hvs_l, hs_r, hus_r, hvs_r = _hydrostatic_states(
        h_l, hu_l, hv_l, bed_l, h_r, hu_r, hv_r, bed_r
    )
    left_flux = _physical_flux_x(hs_l, hus_l, hvs_l)
    right_flux = _physical_flux_x(hs_r, hus_r, hvs_r)
    alpha = np.maximum(_wave_speed(hs_l, hus_l), _wave_speed(hs_r, hus_r))
    flux = tuple(
        0.5 * (left + right) - 0.5 * alpha * (state_r - state_l)
        for left, right, state_l, state_r in zip(
            left_flux, right_flux, (hs_l, hus_l, hvs_l), (hs_r, hus_r, hvs_r)
        )
    )
    correction_left = 0.5 * g * (center_h_l**2 - hs_l**2)
    correction_right = 0.5 * g * (center_h_r**2 - hs_r**2)
    return (*flux, correction_left, correction_right)


def _rusanov_y(h, hu, hv, bed, boundary, spatial_order=1):
    h_ext, hu_ext, hv_ext, bed_ext = _extend_y(h, hu, hv, bed, boundary)
    center_h_b, center_h_t = h_ext[:, :-1], h_ext[:, 1:]
    h_b, h_t = center_h_b.copy(), center_h_t.copy()
    hu_b, hu_t = hu_ext[:, :-1].copy(), hu_ext[:, 1:].copy()
    hv_b, hv_t = hv_ext[:, :-1].copy(), hv_ext[:, 1:].copy()
    bed_b, bed_t = bed_ext[:, :-1].copy(), bed_ext[:, 1:].copy()
    if spatial_order == 2:
        fields = _reconstructed_fields(
            h, hu, hv, bed, axis=1, periodic=boundary == "periodic"
        )
        (surface, ds), (u, du), (v, dv), (bed_cell, db) = fields
        surface_b = surface[:, :-1] + 0.5 * ds[:, :-1]
        surface_t = surface[:, 1:] - 0.5 * ds[:, 1:]
        bed_b[:, 1:-1] = bed_cell[:, :-1] + 0.5 * db[:, :-1]
        bed_t[:, 1:-1] = bed_cell[:, 1:] - 0.5 * db[:, 1:]
        h_b[:, 1:-1] = np.maximum(surface_b - bed_b[:, 1:-1], 0.0)
        h_t[:, 1:-1] = np.maximum(surface_t - bed_t[:, 1:-1], 0.0)
        hu_b[:, 1:-1] = (u[:, :-1] + 0.5 * du[:, :-1]) * h_b[:, 1:-1]
        hu_t[:, 1:-1] = (u[:, 1:] - 0.5 * du[:, 1:]) * h_t[:, 1:-1]
        hv_b[:, 1:-1] = (v[:, :-1] + 0.5 * dv[:, :-1]) * h_b[:, 1:-1]
        hv_t[:, 1:-1] = (v[:, 1:] - 0.5 * dv[:, 1:]) * h_t[:, 1:-1]
        if boundary == "periodic":
            periodic_bed_b = bed_cell[:, -1] + 0.5 * db[:, -1]
            periodic_bed_t = bed_cell[:, 0] - 0.5 * db[:, 0]
            periodic_h_b = np.maximum(
                surface[:, -1] + 0.5 * ds[:, -1] - periodic_bed_b, 0.0
            )
            periodic_h_t = np.maximum(
                surface[:, 0] - 0.5 * ds[:, 0] - periodic_bed_t, 0.0
            )
            for face in (0, -1):
                bed_b[:, face], bed_t[:, face] = periodic_bed_b, periodic_bed_t
                h_b[:, face], h_t[:, face] = periodic_h_b, periodic_h_t
                hu_b[:, face] = (u[:, -1] + 0.5 * du[:, -1]) * periodic_h_b
                hu_t[:, face] = (u[:, 0] - 0.5 * du[:, 0]) * periodic_h_t
                hv_b[:, face] = (v[:, -1] + 0.5 * dv[:, -1]) * periodic_h_b
                hv_t[:, face] = (v[:, 0] - 0.5 * dv[:, 0]) * periodic_h_t
    hs_b, hus_b, hvs_b, hs_t, hus_t, hvs_t = _hydrostatic_states(
        h_b, hu_b, hv_b, bed_b, h_t, hu_t, hv_t, bed_t
    )
    bottom_flux = _physical_flux_y(hs_b, hus_b, hvs_b)
    top_flux = _physical_flux_y(hs_t, hus_t, hvs_t)
    alpha = np.maximum(_wave_speed(hs_b, hvs_b), _wave_speed(hs_t, hvs_t))
    flux = tuple(
        0.5 * (bottom + top) - 0.5 * alpha * (state_t - state_b)
        for bottom, top, state_b, state_t in zip(
            bottom_flux, top_flux, (hs_b, hus_b, hvs_b), (hs_t, hus_t, hvs_t)
        )
    )
    correction_bottom = 0.5 * g * (center_h_b**2 - hs_b**2)
    correction_top = 0.5 * g * (center_h_t**2 - hs_t**2)
    return (*flux, correction_bottom, correction_top)


def _draining_factors(h, source, dt, dx, dy, flux_h_x, flux_h_y):
    area = dx[:, None] * dy[None, :]
    outgoing = (
        np.maximum(flux_h_x[1:, :], 0.0) * dy[None, :]
        + np.maximum(-flux_h_x[:-1, :], 0.0) * dy[None, :]
        + np.maximum(flux_h_y[:, 1:], 0.0) * dx[:, None]
        + np.maximum(-flux_h_y[:, :-1], 0.0) * dx[:, None]
    )
    available = (h + dt * source) * area
    theta = np.ones_like(h)
    draining = dt * outgoing > available
    np.divide(
        available,
        dt * outgoing,
        out=theta,
        where=draining,
    )
    return np.clip(theta, 0.0, 1.0)


def _limit_face_fluxes(theta, flux_x, flux_y, boundary_x, boundary_y):
    nx, ny = theta.shape
    if boundary_x == "periodic":
        donor_left = np.concatenate([theta[-1:, :], theta], axis=0)
        donor_right = np.concatenate([theta, theta[:1, :]], axis=0)
    else:
        ones_x = np.ones((1, ny))
        donor_left = np.concatenate([ones_x, theta], axis=0)
        donor_right = np.concatenate([theta, ones_x], axis=0)
    factor_x = np.where(flux_x[0] >= 0.0, donor_left, donor_right)

    if boundary_y == "periodic":
        donor_bottom = np.concatenate([theta[:, -1:], theta], axis=1)
        donor_top = np.concatenate([theta, theta[:, :1]], axis=1)
    else:
        ones_y = np.ones((nx, 1))
        donor_bottom = np.concatenate([ones_y, theta], axis=1)
        donor_top = np.concatenate([theta, ones_y], axis=1)
    factor_y = np.where(flux_y[0] >= 0.0, donor_bottom, donor_top)
    return (
        tuple(component * factor_x for component in flux_x),
        tuple(component * factor_y for component in flux_y),
    )


def run_model(
    L=L,
    W=W,
    T_final=T_final,
    record_interval=1.0,
    h_init=None,
    hu_init=None,
    hv_init=None,
    left_inflow=0.0,
    rainfall=None,
    x_m=None,
    y_m=None,
    dx_m=None,
    dy_m=None,
    slope_x=None,
    slope_y=None,
    manning_n=None,
    bed_elevation_m=None,
    soil_ksat_m_per_min=None,
    soil_suction_head_m=None,
    soil_moisture_deficit=None,
    initial_cumulative_infiltration_m=0.0,
    momentum_source=None,
    cfl=CFL,
    spatial_order=1,
    boundary_x="inflow_outflow",
    boundary_y="wall",
    downstream_stage_m=None,
):
    if not np.isfinite(T_final) or T_final < 0:
        raise ValueError("T_final must be finite and non-negative")
    if not np.isfinite(record_interval) or record_interval <= 0:
        raise ValueError("record_interval must be finite and positive")
    if not np.isfinite(cfl) or not (0 < cfl <= 0.5):
        raise ValueError("2-D cfl must be finite and in (0, 0.5]")
    if spatial_order not in (1, 2):
        raise ValueError("spatial_order must be 1 or 2")
    if boundary_x not in {"inflow_outflow", "inflow_stage", "periodic"}:
        raise ValueError(
            "boundary_x must be 'inflow_outflow', 'inflow_stage', or 'periodic'"
        )
    if boundary_y not in {"wall", "periodic"}:
        raise ValueError("boundary_y must be 'wall' or 'periodic'")
    if boundary_x == "periodic" and (
        callable(left_inflow) or float(left_inflow) != 0.0
    ):
        raise ValueError("left_inflow must be zero when boundary_x is periodic")
    if (boundary_x == "inflow_stage") != (downstream_stage_m is not None):
        raise ValueError(
            "downstream_stage_m is required only when boundary_x='inflow_stage'"
        )

    if x_m is None and y_m is None:
        if not (np.isfinite(L) and np.isfinite(W) and L > 0 and W > 0):
            raise ValueError("L and W must be finite and positive")
        nx, ny = max(1, int(L * 10)), max(1, int(W * 10))
        x, dx = _grid(L, nx)
        y, dy = _grid(W, ny)
    elif x_m is not None and y_m is not None and dx_m is not None and dy_m is not None:
        x = np.asarray(x_m, dtype=float)
        y = np.asarray(y_m, dtype=float)
        dx = np.asarray(dx_m, dtype=float)
        dy = np.asarray(dy_m, dtype=float)
        if (
            x.ndim != 1
            or y.ndim != 1
            or dx.shape != x.shape
            or dy.shape != y.shape
            or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))
            or not np.all(np.isfinite(dx))
            or not np.all(np.isfinite(dy))
            or np.any(dx <= 0)
            or np.any(dy <= 0)
            or (len(x) > 1 and np.any(np.diff(x) <= 0))
            or (len(y) > 1 and np.any(np.diff(y) <= 0))
        ):
            raise ValueError("Supplied x/y grids need increasing centres and positive widths")
        nx, ny = len(x), len(y)
    else:
        raise ValueError("x_m, y_m, dx_m, and dy_m must be supplied together")

    shape = (nx, ny)
    soil = prepare_green_ampt(
        soil_ksat_m_per_min,
        soil_suction_head_m,
        soil_moisture_deficit,
        shape,
    )
    cumulative_infiltration = initial_cumulative_infiltration(
        initial_cumulative_infiltration_m, shape
    )
    if soil is None and np.any(cumulative_infiltration != 0.0):
        raise ValueError(
            "initial_cumulative_infiltration_m requires Green-Ampt soil properties"
        )
    bed_slope_x = _field(slope_x, S0x, shape, "slope_x")
    bed_slope_y = _field(slope_y, S0y, shape, "slope_y")
    roughness = _field(manning_n, n0, shape, "manning_n")
    if np.any(roughness < 0):
        raise ValueError("manning_n cannot be negative")
    bed = (
        _bed_from_slopes(x, y, bed_slope_x, bed_slope_y)
        if bed_elevation_m is None
        else _field(bed_elevation_m, 0.0, shape, "bed_elevation_m")
    )

    h = _field(h_init, 0.01, shape, "h_init").copy()
    hu = _field(hu_init, 0.0, shape, "hu_init").copy()
    hv = _field(hv_init, 0.0, shape, "hv_init").copy()
    if np.any(h < 0):
        raise ValueError("h_init cannot contain negative depths")
    dry = h <= DRY_TOL
    hu[dry] = 0.0
    hv[dry] = 0.0
    _check_finite(0.0, h=h, hu=hu, hv=hv)

    h_initial = h.copy()
    hu_initial = hu.copy()
    hv_initial = hv.copy()
    record_marks = _record_times(T_final, record_interval)
    times = [0.0]
    h_history = [h.copy()]
    hu_history = [hu.copy()]
    hv_history = [hv.copy()]
    cumulative_infiltration_history = [cumulative_infiltration.copy()]
    next_record = 1

    mass_inflow = 0.0
    mass_outflow = 0.0
    mass_source = 0.0
    mass_infiltration = 0.0
    mass_floor_correction = 0.0
    t_current = 0.0
    area = dx[:, None] * dy[None, :]
    rainfall_function = rainfall

    while t_current < T_final - 1e-14:
        source_value = (
            r(t_current)
            if rainfall_function is None
            else rainfall_function(x, y, t_current)
        )
        source = _field(source_value, 0.0, shape, "rainfall")
        if np.any(source < 0):
            raise ValueError("rainfall cannot be negative")
        speed_x = _wave_speed(h, hu)
        speed_y = _wave_speed(h, hv)
        spectral_rate = speed_x / dx[:, None] + speed_y / dy[None, :]
        max_rate = float(np.max(spectral_rate))
        dt = T_final - t_current if max_rate <= 1e-14 else cfl / max_rate
        dt = min(dt, T_final - t_current)
        dt = _cap_dt_for_wetting_source(
            dt, h, hu, hv, source, dx, dy, cfl
        )
        if next_record < len(record_marks):
            dt = min(dt, record_marks[next_record] - t_current)
        dt = _cap_dt_at_forcing_breakpoints(
            dt,
            t_current,
            left_inflow,
            rainfall_function,
            downstream_stage_m,
        )
        if dt <= 0 or not np.isfinite(dt):
            raise FloatingPointError(f"Invalid time step {dt!r} at t={t_current}")

        inflow = _inflow_values(left_inflow, t_current, ny)
        downstream_stage = (
            None
            if downstream_stage_m is None
            else _stage_values(downstream_stage_m, t_current, ny)
        )
        x_raw = _rusanov_x(
            h,
            hu,
            hv,
            bed,
            boundary_x,
            inflow,
            spatial_order,
            downstream_stage,
        )
        y_raw = _rusanov_y(h, hu, hv, bed, boundary_y, spatial_order)
        theta = _draining_factors(h, source, dt, dx, dy, x_raw[0], y_raw[0])
        x_flux, y_flux = _limit_face_fluxes(
            theta, x_raw[:3], y_raw[:3], boundary_x, boundary_y
        )
        fh, fhu, fhv = x_flux
        gh, ghu, ghv = y_flux
        corr_x_left, corr_x_right = x_raw[3], x_raw[4]
        corr_y_bottom, corr_y_top = y_raw[3], y_raw[4]

        h_new = h - dt * (
            (fh[1:, :] - fh[:-1, :]) / dx[:, None]
            + (gh[:, 1:] - gh[:, :-1]) / dy[None, :]
        ) + dt * source
        hu_star = hu - dt * (
            ((fhu + corr_x_left)[1:, :] - (fhu + corr_x_right)[:-1, :])
            / dx[:, None]
            + (ghu[:, 1:] - ghu[:, :-1]) / dy[None, :]
        )
        hv_star = hv - dt * (
            (fhv[1:, :] - fhv[:-1, :]) / dx[:, None]
            + ((ghv + corr_y_bottom)[:, 1:] - (ghv + corr_y_top)[:, :-1])
            / dy[None, :]
        )
        if momentum_source is not None:
            momentum_values = momentum_source(x, y, t_current)
            if not isinstance(momentum_values, (tuple, list)) or len(momentum_values) != 2:
                raise ValueError("momentum_source must return (source_hu, source_hv)")
            source_hu = _field(momentum_values[0], 0.0, shape, "source_hu")
            source_hv = _field(momentum_values[1], 0.0, shape, "source_hv")
            hu_star += dt * source_hu
            hv_star += dt * source_hv

        negative = np.minimum(h_new, 0.0)
        negative_volume = -float(np.sum(negative * area))
        scale = max(1.0, float(np.sum(h * area)))
        if negative_volume > POSITIVITY_TOL * scale:
            raise FloatingPointError(
                f"Positivity limiter failed at t={t_current + dt:.16g}: "
                f"negative volume={negative_volume:.6e}"
            )
        mass_floor_correction += negative_volume
        h_new = np.maximum(h_new, 0.0)

        infiltration_volume = 0.0
        if soil is not None:
            infiltrated_depth, cumulative_infiltration = green_ampt_step(
                h_new,
                cumulative_infiltration,
                *soil,
                dt,
            )
            depth_before_infiltration = h_new.copy()
            h_new = np.maximum(h_new - infiltrated_depth, 0.0)
            retained_fraction = np.zeros_like(h_new)
            np.divide(
                h_new,
                depth_before_infiltration,
                out=retained_fraction,
                where=depth_before_infiltration > DRY_TOL,
            )
            hu_star *= retained_fraction
            hv_star *= retained_fraction
            infiltration_volume = float(np.sum(infiltrated_depth * area))

        u_star = _velocity(h_new, hu_star)
        v_star = _velocity(h_new, hv_star)
        speed = np.sqrt(u_star**2 + v_star**2)
        friction = np.zeros_like(h_new)
        wet = h_new > DRY_TOL
        friction[wet] = roughness[wet] ** 2 * speed[wet] / h_new[wet] ** (4.0 / 3.0)
        denominator = 1.0 + dt * g * friction
        hu_new = hu_star / denominator
        hv_new = hv_star / denominator
        hu_new[~wet] = 0.0
        hv_new[~wet] = 0.0
        _check_finite(t_current + dt, h=h_new, hu=hu_new, hv=hv_new)

        if boundary_x != "periodic":
            left_volume = float(np.sum(fh[0, :] * dy) * dt)
            right_volume = float(np.sum(fh[-1, :] * dy) * dt)
            if left_volume >= 0.0:
                mass_inflow += left_volume
            else:
                mass_outflow -= left_volume
            if right_volume >= 0.0:
                mass_outflow += right_volume
            else:
                mass_inflow -= right_volume
        mass_source += float(np.sum(source * area) * dt) - infiltration_volume
        mass_infiltration += infiltration_volume

        h, hu, hv = h_new, hu_new, hv_new
        t_current += dt
        if next_record < len(record_marks) and t_current >= record_marks[next_record] - 1e-12:
            times.append(record_marks[next_record])
            h_history.append(h.copy())
            hu_history.append(hu.copy())
            hv_history.append(hv.copy())
            cumulative_infiltration_history.append(
                cumulative_infiltration.copy()
            )
            next_record += 1

    return {
        "x": x,
        "y": y,
        "dx_m": dx,
        "dy_m": dy,
        "bed_elevation_m": bed,
        "times": np.asarray(times),
        "h_history": np.asarray(h_history),
        "hu_history": np.asarray(hu_history),
        "hv_history": np.asarray(hv_history),
        "h_initial": h_initial,
        "h_final": h,
        "hu_initial": hu_initial,
        "hu_final": hu,
        "hv_initial": hv_initial,
        "hv_final": hv,
        "mass_inflow": mass_inflow,
        "mass_outflow": mass_outflow,
        "mass_source": mass_source,
        "mass_infiltration": mass_infiltration,
        "soil_ksat_m_per_min": None if soil is None else soil[0],
        "soil_suction_head_m": None if soil is None else soil[1],
        "soil_moisture_deficit": None if soil is None else soil[2],
        "cumulative_infiltration_history": np.asarray(
            cumulative_infiltration_history
        ),
        "cumulative_infiltration_final": cumulative_infiltration,
        "mass_floor_correction": mass_floor_correction,
        "spatial_order": spatial_order,
    }


class _SaintVenant2DSolver:
    name = "saint_venant_2d"
    supports = frozenset({
        "initial_depth",
        "initial_discharge",
        "initial_discharge_y",
        "left_inflow",
        "rainfall",
        "rainfall_2d",
        "cfl",
        "boundary_x",
        "boundary_y",
        "downstream_stage",
        "spatial_order",
        "initial_cumulative_infiltration",
        "soil_infiltration",
    })

    def run(self, domain: Domain2D, scenario: Scenario) -> SimulationResult:
        if not isinstance(domain, Domain2D):
            raise TypeError("saint_venant_2d requires a Domain2D")
        shape = (len(domain.x_m), len(domain.y_m))

        def state(value, name):
            array = np.asarray(value, dtype=float)
            if array.ndim == 0:
                array = np.full(shape, float(array))
            elif array.shape == (shape[0],):
                array = np.broadcast_to(array[:, None], shape).copy()
            if array.shape != shape:
                raise ValueError(f"{name} must be scalar, longitudinal, or have shape {shape}")
            return array

        if scenario.rainfall_2d is not None:
            rainfall = scenario.rainfall_2d
        elif scenario.rainfall is not None:
            def rainfall(x, y, time):
                del y
                values = np.asarray(scenario.rainfall(x, time), dtype=float)
                return np.broadcast_to(values[:, None], shape)
            if hasattr(scenario.rainfall, "breakpoints_min"):
                rainfall.breakpoints_min = scenario.rainfall.breakpoints_min
        else:
            rainfall = lambda x, y, time: np.zeros(shape)

        raw = run_model(
            T_final=scenario.t_final_min,
            record_interval=scenario.record_interval_min,
            h_init=state(scenario.initial_depth_m, "initial_depth_m"),
            hu_init=state(scenario.initial_discharge, "initial_discharge"),
            hv_init=state(scenario.initial_discharge_y, "initial_discharge_y"),
            left_inflow=scenario.left_inflow,
            rainfall=rainfall,
            x_m=domain.x_m,
            y_m=domain.y_m,
            dx_m=domain.dx_m,
            dy_m=domain.dy_m,
            slope_x=domain.slope_x,
            slope_y=domain.slope_y,
            manning_n=domain.manning_n,
            bed_elevation_m=domain.bed_elevation_m,
            soil_ksat_m_per_min=domain.soil_ksat_m_per_min,
            soil_suction_head_m=domain.soil_suction_head_m,
            soil_moisture_deficit=domain.soil_moisture_deficit,
            initial_cumulative_infiltration_m=(
                scenario.initial_cumulative_infiltration_m
            ),
            cfl=scenario.cfl,
            spatial_order=scenario.spatial_order,
            boundary_x=scenario.boundary_x,
            boundary_y=scenario.boundary_y,
            downstream_stage_m=scenario.downstream_stage_m,
        )
        return SimulationResult(
            domain=domain,
            times=raw["times"],
            depth_history=raw["h_history"],
            depth_initial=raw["h_initial"],
            depth_final=raw["h_final"],
            mass_inflow=raw["mass_inflow"],
            mass_source=raw["mass_source"],
            mass_outflow=raw["mass_outflow"],
            mass_correction=raw["mass_floor_correction"],
            extra={
                "bed_elevation_m": raw["bed_elevation_m"],
                "discharge_x_history": raw["hu_history"],
                "discharge_y_history": raw["hv_history"],
                "discharge_x_final": raw["hu_final"],
                "discharge_y_final": raw["hv_final"],
                "mass_floor_correction": raw["mass_floor_correction"],
                "mass_infiltration": raw["mass_infiltration"],
                "soil_ksat_m_per_min": raw["soil_ksat_m_per_min"],
                "soil_suction_head_m": raw["soil_suction_head_m"],
                "soil_moisture_deficit": raw["soil_moisture_deficit"],
                "cumulative_infiltration_history": raw[
                    "cumulative_infiltration_history"
                ],
                "cumulative_infiltration_final": raw[
                    "cumulative_infiltration_final"
                ],
            },
        )


SOLVER = _SaintVenant2DSolver()


def save_time_series_csv(result, path):
    """Write a y-averaged depth table for compatibility with 1-D viewers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_profiles = np.mean(result["h_history"], axis=2)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t"] + [f"{xi:.6f}" for xi in result["x"]])
        for time, depth_row in zip(result["times"], depth_profiles):
            writer.writerow([f"{time:.6f}"] + [f"{depth:.10g}" for depth in depth_row])


if __name__ == "__main__":
    result = run_model()
    plt.plot(result["x"], np.mean(result["h_initial"], axis=1), label="Initial")
    plt.plot(result["x"], np.mean(result["h_final"], axis=1), label="Final", ls="--")
    plt.legend()
    plt.xlabel("x (m)")
    plt.ylabel("y-averaged h (m)")
    plt.savefig("data/saint_venant_2d.png")
    save_time_series_csv(result, "data/saint_venant_2d_timeseries.csv")
