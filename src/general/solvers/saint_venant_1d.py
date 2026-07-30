import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from general.solvers.contract import Domain, Scenario, SimulationResult
from general.solvers import cross_section as tabulated_section
from general.solvers.infiltration import (
    green_ampt_step,
    initial_cumulative_infiltration,
    prepare_green_ampt,
)


# Units: meters and minutes throughout.
L = 10.0
T_final = 30.0
S0 = 0.05
MANNING_N_SECONDS = 0.05
n0 = MANNING_N_SECONDS / 60.0
g = 35316.0  # 9.81 m/s^2 converted to m/min^2.
H_FLOOR = 1e-10
CFL = 0.5


def r(x, t):
    if 0 <= t < 50:
        return np.full_like(x, 0.00002, dtype=float)
    return np.zeros_like(x, dtype=float)


def _cross_section_area(
    h,
    bottom_width,
    side_slope,
    table_depth=None,
    table_width=None,
):
    if table_depth is not None:
        return tabulated_section.area(table_depth, table_width, h)
    depth = np.asarray(h, dtype=float)
    return bottom_width * depth + side_slope * depth**2


def _top_width(
    h,
    bottom_width,
    side_slope,
    table_depth=None,
    table_width=None,
):
    if table_depth is not None:
        return tabulated_section.top_width(table_depth, table_width, h)
    return bottom_width + 2.0 * side_slope * np.asarray(h, dtype=float)


def _hydrostatic_moment(
    h,
    bottom_width,
    side_slope,
    table_depth=None,
    table_width=None,
):
    if table_depth is not None:
        return tabulated_section.hydrostatic_moment(
            table_depth, table_width, h
        )
    depth = np.asarray(h, dtype=float)
    return 0.5 * bottom_width * depth**2 + side_slope * depth**3 / 3.0


def _depth_from_area(
    area,
    bottom_width,
    side_slope,
    table_depth=None,
    table_width=None,
):
    if table_depth is not None:
        return tabulated_section.depth_from_area(
            table_depth, table_width, area
        )
    area, bottom_width, side_slope = np.broadcast_arrays(
        np.asarray(area, dtype=float),
        np.asarray(bottom_width, dtype=float),
        np.asarray(side_slope, dtype=float),
    )
    discriminant = np.sqrt(bottom_width**2 + 4.0 * side_slope * area)
    trapezoidal_depth = 2.0 * area / (bottom_width + discriminant)
    rectangular_depth = area / bottom_width
    return np.where(side_slope > 1e-14, trapezoidal_depth, rectangular_depth)


def _hydraulic_radius(
    h,
    bottom_width,
    side_slope,
    table_depth=None,
    table_width=None,
    table_perimeter=None,
):
    if table_depth is not None:
        return tabulated_section.hydraulic_radius(
            table_depth, table_width, h, table_perimeter
        )
    area = _cross_section_area(h, bottom_width, side_slope)
    perimeter = bottom_width + 2.0 * h * np.sqrt(1.0 + side_slope**2)
    radius = np.zeros_like(area)
    np.divide(area, perimeter, out=radius, where=perimeter > 0.0)
    return radius


def _velocity(
    h,
    discharge,
    bottom_width,
    side_slope,
    table_depth=None,
    table_width=None,
):
    velocity = np.zeros_like(discharge, dtype=float)
    area = _cross_section_area(
        h, bottom_width, side_slope, table_depth, table_width
    )
    np.divide(discharge, area, out=velocity, where=area > H_FLOOR)
    return velocity


def _physical_flux(
    h,
    discharge,
    bottom_width,
    side_slope,
    table_depth=None,
    table_width=None,
):
    velocity = _velocity(
        h, discharge, bottom_width, side_slope, table_depth, table_width
    )
    return (
        discharge,
        discharge * velocity
        + g
        * _hydrostatic_moment(
            h, bottom_width, side_slope, table_depth, table_width
        ),
    )


def _left_discharge(left_inflow, t):
    value = left_inflow(t) if callable(left_inflow) else left_inflow
    value = 0.0 if value is None else float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("left_inflow must return a finite, non-negative discharge")
    return value


def _downstream_ghost(
    h,
    discharge,
    bed,
    downstream_boundary,
    downstream_stage_m,
    time,
    bottom_width,
    side_slope,
    table_depth=None,
    table_width=None,
):
    if downstream_boundary == "outflow":
        return h[-1], discharge[-1]
    if downstream_boundary == "wall":
        return h[-1], -discharge[-1]
    if downstream_boundary != "stage":
        raise ValueError(
            "downstream_boundary must be 'outflow', 'wall', or 'stage'"
        )
    if downstream_stage_m is None:
        raise ValueError(
            "downstream_stage_m is required for a stage boundary"
        )
    stage = (
        downstream_stage_m(time)
        if callable(downstream_stage_m)
        else downstream_stage_m
    )
    stage = float(stage)
    if not np.isfinite(stage):
        raise ValueError("downstream_stage_m must be finite")
    ghost_depth = max(stage - bed[-1], 0.0)
    ghost_discharge = 0.0
    if h[-1] > H_FLOOR:
        interior_area = _cross_section_area(
            h[-1],
            bottom_width[-1],
            side_slope[-1],
            table_depth,
            None if table_width is None else table_width[-1],
        )
        ghost_area = _cross_section_area(
            ghost_depth,
            bottom_width[-1],
            side_slope[-1],
            table_depth,
            None if table_width is None else table_width[-1],
        )
        ghost_discharge = discharge[-1] * ghost_area / interior_area
    return ghost_depth, ghost_discharge


def _bed_from_slope(x, slope):
    bed = np.zeros_like(x, dtype=float)
    if len(x) > 1:
        bed[1:] = -np.cumsum(
            0.5 * (slope[:-1] + slope[1:]) * np.diff(x)
        )
    return bed


def _hydrostatic_states(
    h_left,
    q_left,
    bed_left,
    bottom_left,
    side_left,
    h_right,
    q_right,
    bed_right,
    bottom_right,
    side_right,
    table_depth=None,
    table_left=None,
    table_right=None,
):
    face_bed = np.maximum(bed_left, bed_right)
    depth_left = np.maximum(0.0, h_left + bed_left - face_bed)
    depth_right = np.maximum(0.0, h_right + bed_right - face_bed)
    scale_left = np.zeros_like(h_left)
    scale_right = np.zeros_like(h_right)
    area_left = _cross_section_area(
        h_left, bottom_left, side_left, table_depth, table_left
    )
    area_right = _cross_section_area(
        h_right, bottom_right, side_right, table_depth, table_right
    )
    np.divide(
        _cross_section_area(
            depth_left, bottom_left, side_left, table_depth, table_left
        ),
        area_left,
        out=scale_left,
        where=area_left > H_FLOOR,
    )
    np.divide(
        _cross_section_area(
            depth_right, bottom_right, side_right, table_depth, table_right
        ),
        area_right,
        out=scale_right,
        where=area_right > H_FLOOR,
    )
    return depth_left, q_left * scale_left, depth_right, q_right * scale_right


def _limited_increments(values):
    """Minmod-limited cell increments; zero at physical boundaries."""
    values = np.asarray(values, dtype=float)
    increments = np.zeros_like(values)
    backward = values[1:-1] - values[:-2]
    forward = values[2:] - values[1:-1]
    same_sign = backward * forward > 0.0
    increments[1:-1] = np.where(
        same_sign,
        np.sign(backward) * np.minimum(np.abs(backward), np.abs(forward)),
        0.0,
    )
    return increments


def _interior_face_states(
    h,
    discharge,
    bed,
    bottom_width,
    side_slope,
    spatial_order,
):
    """Return left/right raw states at the n-1 interior faces."""
    if spatial_order == 1:
        return (
            h[:-1].copy(),
            discharge[:-1].copy(),
            bed[:-1].copy(),
            bottom_width[:-1].copy(),
            side_slope[:-1].copy(),
            h[1:].copy(),
            discharge[1:].copy(),
            bed[1:].copy(),
            bottom_width[1:].copy(),
            side_slope[1:].copy(),
        )

    surface = h + bed
    unit_discharge = discharge / bottom_width
    surface_increment = _limited_increments(surface)
    bed_increment = _limited_increments(bed)
    bottom_increment = _limited_increments(bottom_width)
    side_increment = _limited_increments(side_slope)
    discharge_increment = _limited_increments(unit_discharge)

    surface_left = surface[:-1] + 0.5 * surface_increment[:-1]
    surface_right = surface[1:] - 0.5 * surface_increment[1:]
    bed_left = bed[:-1] + 0.5 * bed_increment[:-1]
    bed_right = bed[1:] - 0.5 * bed_increment[1:]
    bottom_left = bottom_width[:-1] + 0.5 * bottom_increment[:-1]
    bottom_right = bottom_width[1:] - 0.5 * bottom_increment[1:]
    side_left = side_slope[:-1] + 0.5 * side_increment[:-1]
    side_right = side_slope[1:] - 0.5 * side_increment[1:]
    if np.any(bottom_left <= 0.0) or np.any(bottom_right <= 0.0):
        raise FloatingPointError("Limited reconstruction produced non-positive width")
    if np.any(side_left < 0.0) or np.any(side_right < 0.0):
        raise FloatingPointError("Limited reconstruction produced negative side slope")
    h_left = np.maximum(surface_left - bed_left, 0.0)
    h_right = np.maximum(surface_right - bed_right, 0.0)
    q_left = (
        unit_discharge[:-1] + 0.5 * discharge_increment[:-1]
    ) * bottom_left
    q_right = (
        unit_discharge[1:] - 0.5 * discharge_increment[1:]
    ) * bottom_right
    q_left[h_left <= H_FLOOR] = 0.0
    q_right[h_right <= H_FLOOR] = 0.0
    return (
        h_left,
        q_left,
        bed_left,
        bottom_left,
        side_left,
        h_right,
        q_right,
        bed_right,
        bottom_right,
        side_right,
    )


def _rusanov_fluxes(
    h,
    discharge,
    bed,
    bottom_width,
    side_slope,
    left_inflow,
    t,
    downstream_boundary,
    downstream_stage_m,
    spatial_order,
    table_depth=None,
    table_width=None,
):
    inflow = _left_discharge(left_inflow, t)
    right_h, right_q = _downstream_ghost(
        h,
        discharge,
        bed,
        downstream_boundary,
        downstream_stage_m,
        t,
        bottom_width,
        side_slope,
        table_depth,
        table_width,
    )

    # Equal ghost/interior depths remove numerical mass diffusion at each
    # boundary. Mirroring q around the requested inflow makes the left face
    # mass flux exactly equal to that prescribed discharge.
    h_ext = np.concatenate(([h[0]], h, [right_h]))
    q_ext = np.concatenate(
        ([2.0 * inflow - discharge[0]], discharge, [right_q])
    )
    bed_ext = np.concatenate(([bed[0]], bed, [bed[-1]]))
    bottom_ext = np.concatenate(
        ([bottom_width[0]], bottom_width, [bottom_width[-1]])
    )
    side_ext = np.concatenate(
        ([side_slope[0]], side_slope, [side_slope[-1]])
    )
    table_ext = (
        None
        if table_width is None
        else np.concatenate(
            (table_width[[0]], table_width, table_width[[-1]]), axis=0
        )
    )
    center_h_left, center_h_right = h_ext[:-1], h_ext[1:]
    center_bottom_left, center_bottom_right = bottom_ext[:-1], bottom_ext[1:]
    center_side_left, center_side_right = side_ext[:-1], side_ext[1:]
    center_table_left = None if table_ext is None else table_ext[:-1]
    center_table_right = None if table_ext is None else table_ext[1:]
    h_left, h_right = center_h_left.copy(), center_h_right.copy()
    q_left, q_right = q_ext[:-1].copy(), q_ext[1:].copy()
    bed_left, bed_right = bed_ext[:-1].copy(), bed_ext[1:].copy()
    bottom_left, bottom_right = (
        center_bottom_left.copy(),
        center_bottom_right.copy(),
    )
    side_left, side_right = center_side_left.copy(), center_side_right.copy()
    (
        h_left[1:-1],
        q_left[1:-1],
        bed_left[1:-1],
        bottom_left[1:-1],
        side_left[1:-1],
        h_right[1:-1],
        q_right[1:-1],
        bed_right[1:-1],
        bottom_right[1:-1],
        side_right[1:-1],
    ) = _interior_face_states(
        h,
        discharge,
        bed,
        bottom_width,
        side_slope,
        spatial_order,
    )
    table_left = (
        None if center_table_left is None else center_table_left.copy()
    )
    table_right = (
        None if center_table_right is None else center_table_right.copy()
    )
    face_bottom = 0.5 * (bottom_left + bottom_right)
    face_side = 0.5 * (side_left + side_right)
    face_table = (
        None
        if table_left is None
        else 0.5 * (table_left + table_right)
    )

    hs_left, qs_left, hs_right, qs_right = _hydrostatic_states(
        h_left,
        q_left,
        bed_left,
        bottom_left,
        side_left,
        h_right,
        q_right,
        bed_right,
        bottom_right,
        side_right,
        table_depth,
        table_left,
        table_right,
    )
    # Reconstruct a common face area while retaining each side's velocity.
    side_area_left = _cross_section_area(
        hs_left, bottom_left, side_left, table_depth, table_left
    )
    side_area_right = _cross_section_area(
        hs_right, bottom_right, side_right, table_depth, table_right
    )
    area_left = _cross_section_area(
        hs_left, face_bottom, face_side, table_depth, face_table
    )
    area_right = _cross_section_area(
        hs_right, face_bottom, face_side, table_depth, face_table
    )
    scale_left = np.zeros_like(qs_left)
    scale_right = np.zeros_like(qs_right)
    np.divide(area_left, side_area_left, out=scale_left, where=side_area_left > H_FLOOR)
    np.divide(
        area_right,
        side_area_right,
        out=scale_right,
        where=side_area_right > H_FLOOR,
    )
    qs_left *= scale_left
    qs_right *= scale_right
    flux_h_left, flux_q_left = _physical_flux(
        hs_left,
        qs_left,
        face_bottom,
        face_side,
        table_depth,
        face_table,
    )
    flux_h_right, flux_q_right = _physical_flux(
        hs_right,
        qs_right,
        face_bottom,
        face_side,
        table_depth,
        face_table,
    )
    hydraulic_depth_left = np.zeros_like(area_left)
    hydraulic_depth_right = np.zeros_like(area_right)
    np.divide(
        area_left,
        _top_width(
            hs_left, face_bottom, face_side, table_depth, face_table
        ),
        out=hydraulic_depth_left,
        where=area_left > H_FLOOR,
    )
    np.divide(
        area_right,
        _top_width(
            hs_right, face_bottom, face_side, table_depth, face_table
        ),
        out=hydraulic_depth_right,
        where=area_right > H_FLOOR,
    )
    speed_left = np.abs(
        _velocity(
            hs_left,
            qs_left,
            face_bottom,
            face_side,
            table_depth,
            face_table,
        )
    ) + np.sqrt(g * hydraulic_depth_left)
    speed_right = np.abs(
        _velocity(
            hs_right,
            qs_right,
            face_bottom,
            face_side,
            table_depth,
            face_table,
        )
    ) + np.sqrt(g * hydraulic_depth_right)
    alpha = np.maximum(speed_left, speed_right)

    flux_h = 0.5 * (flux_h_left + flux_h_right) - 0.5 * alpha * (
        area_right - area_left
    )
    flux_q = 0.5 * (flux_q_left + flux_q_right) - 0.5 * alpha * (qs_right - qs_left)
    correction_left = g * (
        _hydrostatic_moment(
            center_h_left,
            center_bottom_left,
            center_side_left,
            table_depth,
            center_table_left,
        )
        - _hydrostatic_moment(
            hs_left,
            center_bottom_left,
            center_side_left,
            table_depth,
            center_table_left,
        )
    )
    correction_right = g * (
        _hydrostatic_moment(
            center_h_right,
            center_bottom_right,
            center_side_right,
            table_depth,
            center_table_right,
        )
        - _hydrostatic_moment(
            hs_right,
            center_bottom_right,
            center_side_right,
            table_depth,
            center_table_right,
        )
    )
    geometry_balance = g * (
        _hydrostatic_moment(
            hs_left[1:],
            face_bottom[1:],
            face_side[1:],
            table_depth,
            None if face_table is None else face_table[1:],
        )
        - _hydrostatic_moment(
            hs_left[1:],
            bottom_width,
            side_slope,
            table_depth,
            table_width,
        )
        - _hydrostatic_moment(
            hs_right[:-1],
            face_bottom[:-1],
            face_side[:-1],
            table_depth,
            None if face_table is None else face_table[:-1],
        )
        + _hydrostatic_moment(
            hs_right[:-1],
            bottom_width,
            side_slope,
            table_depth,
            table_width,
        )
    )
    return (
        flux_h,
        flux_q,
        alpha,
        correction_left,
        correction_right,
        geometry_balance,
    )


def _limit_draining_fluxes(area, area_source, dt, dx, flux_h, flux_q):
    outgoing = np.maximum(flux_h[1:], 0.0) + np.maximum(-flux_h[:-1], 0.0)
    available = (area + dt * area_source) * dx
    theta = np.ones_like(area)
    draining = dt * outgoing > available
    np.divide(available, dt * outgoing, out=theta, where=draining)
    theta = np.clip(theta, 0.0, 1.0)
    donor_left = np.concatenate(([1.0], theta))
    donor_right = np.concatenate((theta, [1.0]))
    factor = np.where(flux_h >= 0.0, donor_left, donor_right)
    return flux_h * factor, flux_q * factor


def _cell_values(values, default, n_cells, name):
    if values is None:
        array = np.full(n_cells, default, dtype=float)
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            array = np.full(n_cells, float(array), dtype=float)
    if array.shape != (n_cells,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain one finite value per cell")
    return array


def _prepare_manning_table(depth_m, values, n_cells):
    if (depth_m is None) != (values is None):
        raise ValueError(
            "Manning depth and n tables must be supplied together"
        )
    if depth_m is None:
        return None, None
    depths = np.asarray(depth_m, dtype=float)
    roughness = np.asarray(values, dtype=float)
    if depths.ndim == 1:
        depths = np.broadcast_to(depths, (n_cells, len(depths))).copy()
    if roughness.ndim == 1:
        roughness = np.broadcast_to(
            roughness, (n_cells, len(roughness))
        ).copy()
    if (
        depths.ndim != 2
        or roughness.shape != depths.shape
        or depths.shape[0] != n_cells
        or depths.shape[1] < 2
        or np.any(~np.isfinite(depths))
        or np.any(~np.isfinite(roughness))
        or np.any(depths < 0.0)
        or np.any(np.diff(depths, axis=1) <= 0.0)
        or np.any(roughness <= 0.0)
    ):
        raise ValueError(
            "Manning tables need at least two strictly increasing finite "
            "non-negative depths and one finite positive n value per depth "
            "and cell"
        )
    return depths, roughness


def _manning_at_depth(depth_m, base_values, table_depth, table_values):
    if table_depth is None:
        return base_values
    return np.asarray(
        [
            np.interp(depth, levels, values)
            for depth, levels, values in zip(
                depth_m, table_depth, table_values
            )
        ],
        dtype=float,
    )


def _prepare_grid(domain_length, x_m, dx_m, slope, manning_n):
    if x_m is None:
        if not np.isfinite(domain_length) or domain_length < 0.2:
            raise ValueError("L must be finite and at least 0.2 m")
        n_cells = int(domain_length * 10)
        cell_widths = np.full(n_cells, domain_length / n_cells, dtype=float)
        stations = np.linspace(
            cell_widths[0] / 2,
            domain_length - cell_widths[0] / 2,
            n_cells,
        )
    else:
        stations = np.asarray(x_m, dtype=float)
        cell_widths = np.asarray(dx_m, dtype=float)
        if stations.ndim != 1 or len(stations) < 2 or not np.all(np.isfinite(stations)):
            raise ValueError("x_m must contain at least two finite cell stations")
        if np.any(np.diff(stations) <= 0):
            raise ValueError("x_m values must be strictly increasing")
        if cell_widths.shape != stations.shape or not np.all(np.isfinite(cell_widths)):
            raise ValueError("dx_m must contain one finite width per cell")
        if np.any(cell_widths <= 0):
            raise ValueError("dx_m values must be positive")
        n_cells = len(stations)

    bed_slope = _cell_values(slope, S0, n_cells, "slope")
    roughness = _cell_values(manning_n, n0, n_cells, "manning_n")
    if np.any(bed_slope < 0):
        raise ValueError("slope values must be non-negative")
    if np.any(roughness <= 0):
        raise ValueError("manning_n values must be positive")
    return stations, cell_widths, bed_slope, roughness


def _validate_inputs(final_time, record_interval, n_cells, h_init, q_init):
    if not np.isfinite(final_time) or final_time < 0:
        raise ValueError("T_final must be finite and non-negative")
    if not np.isfinite(record_interval) or record_interval <= 0:
        raise ValueError("record_interval must be finite and positive")

    for name, values in (("h_init", h_init), ("q_init", q_init)):
        if values is None:
            continue
        array = np.asarray(values, dtype=float)
        if array.shape != (n_cells,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain {n_cells} finite values")
    if h_init is not None and np.any(np.asarray(h_init) < 0):
        raise ValueError("h_init cannot contain negative depths")


def _evaluate_rainfall(rainfall, x_m, t):
    values = np.asarray(rainfall(x_m, t), dtype=float)
    if values.ndim == 0:
        values = np.full_like(x_m, float(values), dtype=float)
    if values.shape != x_m.shape:
        raise ValueError("rainfall must return one value per cell")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("rainfall must return finite, non-negative rates")
    return values


def _evaluate_lateral_inflow(lateral_inflow, x_m, t):
    """Evaluate signed lateral discharge in m^3/min per metre of reach."""
    if lateral_inflow is None:
        return np.zeros_like(x_m, dtype=float)
    values = np.asarray(lateral_inflow(x_m, t), dtype=float)
    if values.ndim == 0:
        values = np.full_like(x_m, float(values), dtype=float)
    if values.shape != x_m.shape:
        raise ValueError("lateral_inflow must return one value per cell")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "lateral_inflow must return finite discharge per reach length"
        )
    return values


def _cap_dt_at_forcing_breakpoints(dt, time, *forcings):
    for forcing in forcings:
        for breakpoint in getattr(forcing, "breakpoints_min", ()):
            if time + 1e-12 < breakpoint < time + dt - 1e-12:
                dt = float(breakpoint - time)
                break
    return dt


def _record_times(final_time, record_interval):
    count = int(np.floor(final_time / record_interval + 1e-9))
    values = [index * record_interval for index in range(count + 1)]
    if values[-1] < final_time - 1e-9:
        values.append(float(final_time))
    return values


def _cap_dt_for_wetting_source(
    dt,
    h,
    discharge,
    dx,
    bottom_width,
    side_slope,
    table_depth,
    table_width,
    rainfall_rate,
    lateral_rate,
    cfl,
):
    """Apply a CFL bound to water that positive sources create during a step.

    A dry cell has zero current wave speed, so the ordinary CFL calculation can
    otherwise select the entire remaining simulation as one step.  This bound
    predicts source-added area over a trial step and limits the resulting
    gravity-wave Courant number.  It is independent of output recording times.
    """
    area = _cross_section_area(
        h, bottom_width, side_slope, table_depth, table_width
    )
    top_width = _top_width(
        h, bottom_width, side_slope, table_depth, table_width
    )
    positive_area_rate = np.maximum(
        top_width * rainfall_rate + lateral_rate, 0.0
    )
    if not np.any(positive_area_rate > 0.0):
        return dt

    def maximum_courant(trial_dt):
        predicted_area = area + positive_area_rate * trial_dt
        predicted_depth = _depth_from_area(
            predicted_area,
            bottom_width,
            side_slope,
            table_depth,
            table_width,
        )
        predicted_top_width = _top_width(
            predicted_depth,
            bottom_width,
            side_slope,
            table_depth,
            table_width,
        )
        hydraulic_depth = np.zeros_like(predicted_area)
        np.divide(
            predicted_area,
            predicted_top_width,
            out=hydraulic_depth,
            where=predicted_top_width > 0.0,
        )
        velocity = np.zeros_like(discharge)
        np.divide(
            discharge,
            predicted_area,
            out=velocity,
            where=predicted_area > H_FLOOR,
        )
        speed = np.abs(velocity) + np.sqrt(g * hydraulic_depth)
        return float(np.max(trial_dt * speed / dx))

    candidate = float(dt)
    courant = maximum_courant(candidate)
    if courant <= cfl * (1.0 + 1e-12):
        return candidate
    # Source-created gravity speed scales as sqrt(dt), so its Courant number
    # scales as dt**(3/2). This correction is exact in the dry-source limit.
    candidate *= (cfl / courant) ** (2.0 / 3.0)
    courant = maximum_courant(candidate)
    # When pre-existing velocity dominates, Courant scales linearly with dt;
    # this second conservative correction handles that limit.
    if courant > cfl:
        candidate *= cfl / courant
    return candidate


def run_model(
    L,
    T_final,
    record_interval=0.05,
    h_init=None,
    q_init=None,
    left_inflow=None,
    rainfall=None,
    lateral_inflow=None,
    x_m=None,
    dx_m=None,
    slope=None,
    manning_n=None,
    bed_elevation_m=None,
    channel_width_m=None,
    channel_bottom_width_m=None,
    side_slope_h_to_v=None,
    cross_section_depth_m=None,
    cross_section_top_width_m=None,
    cross_section_wetted_perimeter_m=None,
    manning_depth_m=None,
    manning_n_table=None,
    soil_ksat_m_per_min=None,
    soil_suction_head_m=None,
    soil_moisture_deficit=None,
    initial_cumulative_infiltration_m=0.0,
    downstream_boundary="outflow",
    downstream_stage_m=None,
    spatial_order=1,
    cfl=None,
):
    """Run the 1D Saint-Venant equations on a uniform or supplied cell grid.

    left_inflow is either a non-negative discharge or a callable of time
    returning that discharge. With ``channel_width_m`` or tabulated compound
    geometry it is total flow in m^3/min; the legacy unit-width mode uses
    m^2/min. None is a no-inflow upstream boundary. rainfall is a callable
    ``rainfall(x_m, t_min)`` returning one non-negative rate in m/min per cell.
    ``lateral_inflow(x_m, t_min)`` returns signed distributed lateral discharge
    in m^3/min per metre of reach. Positive flow adds water without
    longitudinal momentum. Negative flow removes water and its proportional
    local momentum; extraction is capped by the water available in each step.
    ``spatial_order=2`` uses limited reconstruction with a two-stage SSP update.
    The downstream boundary may be transmissive outflow, a reflecting wall, or
    a prescribed stage. Supplying all three soil properties enables event-scale
    Green-Ampt infiltration; cumulative infiltration is retained as model state.
    """
    x, dx, bed_slope, roughness = _prepare_grid(
        L,
        x_m,
        dx_m,
        slope,
        manning_n,
    )
    n_cells = len(x)
    soil = prepare_green_ampt(
        soil_ksat_m_per_min,
        soil_suction_head_m,
        soil_moisture_deficit,
        (n_cells,),
    )
    cumulative_infiltration = initial_cumulative_infiltration(
        initial_cumulative_infiltration_m, (n_cells,)
    )
    if soil is None and np.any(cumulative_infiltration != 0.0):
        raise ValueError(
            "initial_cumulative_infiltration_m requires Green-Ampt soil properties"
        )
    manning_table_depth, manning_table_values = _prepare_manning_table(
        manning_depth_m, manning_n_table, n_cells
    )
    bed = (
        _bed_from_slope(x, bed_slope)
        if bed_elevation_m is None
        else _cell_values(bed_elevation_m, 0.0, n_cells, "bed_elevation_m")
    )
    reference_width = _cell_values(
        channel_width_m, 1.0, n_cells, "channel_width_m"
    )
    if np.any(reference_width <= 0):
        raise ValueError("channel_width_m values must be positive")
    if (channel_bottom_width_m is None) != (side_slope_h_to_v is None):
        raise ValueError(
            "channel_bottom_width_m and side_slope_h_to_v must be supplied together"
        )
    if (cross_section_depth_m is None) != (
        cross_section_top_width_m is None
    ):
        raise ValueError(
            "cross_section_depth_m and cross_section_top_width_m must be "
            "supplied together"
        )
    if (
        cross_section_wetted_perimeter_m is not None
        and cross_section_depth_m is None
    ):
        raise ValueError(
            "cross_section_wetted_perimeter_m requires a tabulated section"
        )
    if cross_section_depth_m is not None and (
        channel_bottom_width_m is not None or side_slope_h_to_v is not None
    ):
        raise ValueError(
            "tabulated compound sections cannot be combined with trapezoidal "
            "geometry"
        )
    table_depth = None
    table_width = None
    table_perimeter = None
    if cross_section_depth_m is not None:
        (
            table_depth,
            table_width,
            table_perimeter,
        ) = tabulated_section.validate_table(
            cross_section_depth_m,
            cross_section_top_width_m,
            cell_count=n_cells,
            wetted_perimeter_m=cross_section_wetted_perimeter_m,
        )
    bottom_width = _cell_values(
        (
            table_width[:, 0]
            if table_width is not None
            else (
                channel_bottom_width_m
                if channel_bottom_width_m is not None
                else reference_width
            )
        ),
        1.0,
        n_cells,
        "channel_bottom_width_m",
    )
    side_slope = _cell_values(
        side_slope_h_to_v, 0.0, n_cells, "side_slope_h_to_v"
    )
    if np.any(bottom_width <= 0):
        raise ValueError("channel_bottom_width_m values must be positive")
    if np.any(side_slope < 0):
        raise ValueError("side_slope_h_to_v values must be non-negative")
    if table_width is None and np.any(bottom_width > reference_width):
        raise ValueError(
            "channel_bottom_width_m cannot exceed channel_width_m"
        )
    uses_cross_section = (
        channel_width_m is not None or table_width is not None
    )
    cross_section_shape = (
        (
            "surveyed_asymmetric"
            if table_perimeter is not None
            else "compound_tabulated"
        )
        if table_width is not None
        else (
            "trapezoidal"
            if np.any(side_slope > 0.0)
            else "rectangular"
        )
    )
    _validate_inputs(T_final, record_interval, n_cells, h_init, q_init)
    cfl_value = CFL if cfl is None else float(cfl)
    if not np.isfinite(cfl_value) or not (0 < cfl_value <= 1):
        raise ValueError("cfl must be finite and in the interval (0, 1]")
    if spatial_order not in (1, 2):
        raise ValueError("spatial_order must be 1 or 2")
    _downstream_ghost(
        np.ones(n_cells),
        np.zeros(n_cells),
        bed,
        downstream_boundary,
        downstream_stage_m,
        0.0,
        bottom_width,
        side_slope,
        table_depth,
        table_width,
    )
    if downstream_boundary != "stage" and downstream_stage_m is not None:
        raise ValueError(
            "downstream_stage_m is only valid with downstream_boundary='stage'"
        )
    rainfall_function = r if rainfall is None else rainfall
    center = float(x[0] + 0.5 * (x[-1] - x[0]))

    if h_init is None:
        h = 0.01 * np.exp(-((x - center) ** 2) / 0.2)
    else:
        h = np.asarray(h_init, dtype=float).copy()
    h = np.maximum(h, 0.0)

    if q_init is None:
        q = np.zeros(n_cells)
    else:
        q = np.asarray(q_init, dtype=float).copy()
    q[h <= H_FLOOR] = 0.0

    h_initial = h.copy()
    q_initial = q.copy()

    def boundary_fluxes(h_state, q_state, time):
        flux_h = _rusanov_fluxes(
            h_state,
            q_state,
            bed,
            bottom_width,
            side_slope,
            left_inflow,
            time,
            downstream_boundary,
            downstream_stage_m,
            spatial_order,
            table_depth,
            table_width,
        )[0]
        return float(flux_h[0]), float(flux_h[-1])

    record_times = _record_times(T_final, record_interval)
    times = [0.0]
    h_history = [h.copy()]
    q_history = [q.copy()]
    cumulative_infiltration_history = [cumulative_infiltration.copy()]
    initial_upstream_flux, initial_downstream_flux = boundary_fluxes(
        h, q, 0.0
    )
    upstream_flux_history = [initial_upstream_flux]
    downstream_flux_history = [initial_downstream_flux]
    next_record_idx = 1

    mass_inflow = 0.0
    mass_source = 0.0
    mass_rainfall = 0.0
    mass_lateral_inflow = 0.0
    mass_infiltration = 0.0
    mass_outflow = 0.0
    mass_floor_correction = 0.0
    t_current = 0.0

    def euler_stage(
        h_stage,
        q_stage,
        stage_time,
        dt,
        flux_data=None,
        rainfall_source=None,
        requested_lateral_source=None,
    ):
        if flux_data is None:
            flux_data = _rusanov_fluxes(
                h_stage,
                q_stage,
                bed,
                bottom_width,
                side_slope,
                left_inflow,
                stage_time,
                downstream_boundary,
                downstream_stage_m,
                spatial_order,
                table_depth,
                table_width,
            )
        (
            flux_h,
            flux_q,
            _,
            correction_left,
            correction_right,
            geometry_balance,
        ) = flux_data
        if rainfall_source is None:
            rainfall_source = _evaluate_rainfall(
                rainfall_function, x, stage_time
            )
        if requested_lateral_source is None:
            requested_lateral_source = _evaluate_lateral_inflow(
                lateral_inflow, x, stage_time
            )
        area = _cross_section_area(
            h_stage,
            bottom_width,
            side_slope,
            table_depth,
            table_width,
        )
        rainfall_area_source = (
            _top_width(
                h_stage,
                bottom_width,
                side_slope,
                table_depth,
                table_width,
            )
            * rainfall_source
        )
        # A withdrawal is a demand, not permission to create negative area.
        # Cap it at the water available before flux drainage; the draining
        # limiter below then allocates any remaining volume to face outflows.
        lateral_source = np.maximum(
            requested_lateral_source,
            -area / dt - rainfall_area_source,
        )
        area_source = rainfall_area_source + lateral_source
        flux_h, flux_q = _limit_draining_fluxes(
            area, area_source, dt, dx, flux_h, flux_q
        )
        area_new = (
            area
            - (dt / dx) * (flux_h[1:] - flux_h[:-1])
            + dt * area_source
        )
        q_new = q_stage - (dt / dx) * (
            (flux_q + correction_left)[1:]
            - (flux_q + correction_right)[:-1]
            - geometry_balance
        )
        # Incoming side flow is assumed to carry no longitudinal momentum.
        # Withdrawals carry the local velocity, so removing water does not
        # spuriously accelerate what remains.
        q_new += (
            dt
            * np.minimum(lateral_source, 0.0)
            * _velocity(
                h_stage,
                q_stage,
                bottom_width,
                side_slope,
                table_depth,
                table_width,
            )
        )

        floor_addition = np.maximum(-area_new, 0.0)
        area_new = np.maximum(area_new, 0.0)
        h_new = _depth_from_area(
            area_new,
            bottom_width,
            side_slope,
            table_depth,
            table_width,
        )
        velocity_new = _velocity(
            h_new,
            q_new,
            bottom_width,
            side_slope,
            table_depth,
            table_width,
        )
        friction_coeff = np.zeros_like(h_new)
        wet = h_new > H_FLOOR
        effective_roughness = _manning_at_depth(
            h_new,
            roughness,
            manning_table_depth,
            manning_table_values,
        )
        hydraulic_radius = (
            _hydraulic_radius(
                h_new,
                bottom_width,
                side_slope,
                table_depth,
                table_width,
                table_perimeter,
            )
            if uses_cross_section
            else h_new
        )
        friction_coeff[wet] = (
            effective_roughness[wet] ** 2
            * np.abs(velocity_new[wet])
            / hydraulic_radius[wet] ** (4.0 / 3.0)
        )
        q_new = q_new / (1.0 + dt * g * friction_coeff)
        q_new[h_new <= H_FLOOR] = 0.0
        diagnostics = {
            "inflow_rate": float(flux_h[0]),
            "downstream_rate": float(flux_h[-1]),
            "source_rate": float(np.sum(area_source * dx)),
            "rainfall_rate": float(np.sum(rainfall_area_source * dx)),
            "lateral_inflow_rate": float(np.sum(lateral_source * dx)),
            "floor_volume": float(np.sum(floor_addition * dx)),
        }
        return h_new, q_new, diagnostics

    while t_current < T_final - 1e-12:
        (
            flux_h,
            flux_q,
            interface_speed,
            correction_left,
            correction_right,
            geometry_balance,
        ) = _rusanov_fluxes(
            h,
            q,
            bed,
            bottom_width,
            side_slope,
            left_inflow,
            t_current,
            downstream_boundary,
            downstream_stage_m,
            spatial_order,
            table_depth,
            table_width,
        )
        cell_speed = np.maximum(interface_speed[:-1], interface_speed[1:])
        moving = cell_speed > 1e-12
        if np.any(moving):
            dt = cfl_value * float(np.min(dx[moving] / cell_speed[moving]))
        else:
            dt = T_final - t_current
        dt = min(dt, T_final - t_current)
        if rainfall is None and t_current < 50 < t_current + dt:
            dt = 50 - t_current
        dt = _cap_dt_at_forcing_breakpoints(
            dt,
            t_current,
            left_inflow,
            rainfall_function,
            lateral_inflow,
            downstream_stage_m,
        )
        wetting_rainfall = _evaluate_rainfall(
            rainfall_function, x, t_current
        )
        wetting_lateral = _evaluate_lateral_inflow(
            lateral_inflow, x, t_current
        )
        bound_rainfall = wetting_rainfall
        bound_lateral = wetting_lateral
        if spatial_order == 2 and dt > 0.0:
            bound_time = np.nextafter(t_current + dt, t_current)
            bound_rainfall = np.maximum(
                bound_rainfall,
                _evaluate_rainfall(rainfall_function, x, bound_time),
            )
            bound_lateral = np.maximum(
                bound_lateral,
                _evaluate_lateral_inflow(lateral_inflow, x, bound_time),
            )
        dt = _cap_dt_for_wetting_source(
            dt,
            h,
            q,
            dx,
            bottom_width,
            side_slope,
            table_depth,
            table_width,
            bound_rainfall,
            bound_lateral,
            cfl_value,
        )
        h_previous = h.copy()
        q_previous = q.copy()
        cumulative_infiltration_previous = cumulative_infiltration.copy()
        first_flux_data = (
            flux_h,
            flux_q,
            interface_speed,
            correction_left,
            correction_right,
            geometry_balance,
        )
        h_stage, q_stage, first_diagnostics = euler_stage(
            h,
            q,
            t_current,
            dt,
            first_flux_data,
            wetting_rainfall,
            wetting_lateral,
        )
        if spatial_order == 2:
            second_time = np.nextafter(t_current + dt, t_current)
            h_euler, q_euler, second_diagnostics = euler_stage(
                h_stage, q_stage, second_time, dt
            )
            area_new = 0.5 * (
                _cross_section_area(
                    h,
                    bottom_width,
                    side_slope,
                    table_depth,
                    table_width,
                )
                + _cross_section_area(
                    h_euler,
                    bottom_width,
                    side_slope,
                    table_depth,
                    table_width,
                )
            )
            h_new = _depth_from_area(
                area_new,
                bottom_width,
                side_slope,
                table_depth,
                table_width,
            )
            q_new = 0.5 * (q + q_euler)
            diagnostics = {
                key: 0.5 * (first_diagnostics[key] + second_diagnostics[key])
                for key in first_diagnostics
            }
        else:
            h_new, q_new = h_stage, q_stage
            diagnostics = first_diagnostics

        infiltration_volume = 0.0
        if soil is not None:
            infiltrated_depth, cumulative_infiltration = green_ampt_step(
                h_new,
                cumulative_infiltration,
                *soil,
                dt,
            )
            area_before_infiltration = _cross_section_area(
                h_new,
                bottom_width,
                side_slope,
                table_depth,
                table_width,
            )
            h_new = np.maximum(h_new - infiltrated_depth, 0.0)
            area_after_infiltration = _cross_section_area(
                h_new,
                bottom_width,
                side_slope,
                table_depth,
                table_width,
            )
            retained_fraction = np.zeros_like(area_after_infiltration)
            np.divide(
                area_after_infiltration,
                area_before_infiltration,
                out=retained_fraction,
                where=area_before_infiltration > H_FLOOR,
            )
            q_new *= retained_fraction
            q_new[h_new <= H_FLOOR] = 0.0
            infiltration_volume = float(
                np.sum(
                    (area_before_infiltration - area_after_infiltration) * dx
                )
            )

        mass_floor_correction += diagnostics["floor_volume"]
        mass_inflow += diagnostics["inflow_rate"] * dt
        downstream_mass = diagnostics["downstream_rate"] * dt
        if downstream_mass >= 0.0:
            mass_outflow += downstream_mass
        else:
            mass_inflow -= downstream_mass
        mass_source += diagnostics["source_rate"] * dt
        mass_source -= infiltration_volume
        mass_rainfall += diagnostics["rainfall_rate"] * dt
        mass_lateral_inflow += diagnostics["lateral_inflow_rate"] * dt
        mass_infiltration += infiltration_volume

        h, q = h_new, q_new
        t_next = t_current + dt
        while (
            next_record_idx < len(record_times)
            and record_times[next_record_idx] <= t_next + 1e-10
        ):
            record_time = record_times[next_record_idx]
            fraction = 1.0 if dt == 0 else (record_time - t_current) / dt
            fraction = min(max(fraction, 0.0), 1.0)
            record_h = h_previous + fraction * (h_new - h_previous)
            record_q = q_previous + fraction * (q_new - q_previous)
            record_cumulative_infiltration = (
                cumulative_infiltration_previous
                + fraction
                * (
                    cumulative_infiltration
                    - cumulative_infiltration_previous
                )
            )
            upstream_flux, downstream_flux = boundary_fluxes(
                record_h, record_q, record_time
            )
            times.append(record_time)
            h_history.append(record_h)
            q_history.append(record_q)
            cumulative_infiltration_history.append(
                record_cumulative_infiltration
            )
            upstream_flux_history.append(upstream_flux)
            downstream_flux_history.append(downstream_flux)
            next_record_idx += 1
        t_current = t_next

    return {
        "x": x,
        "dx_m": dx,
        "slope": bed_slope,
        "manning_n": roughness,
        "manning_depth_m": manning_table_depth,
        "manning_n_table": manning_table_values,
        "soil_ksat_m_per_min": None if soil is None else soil[0],
        "soil_suction_head_m": None if soil is None else soil[1],
        "soil_moisture_deficit": None if soil is None else soil[2],
        "cumulative_infiltration_history": np.asarray(
            cumulative_infiltration_history
        ),
        "cumulative_infiltration_final": cumulative_infiltration,
        "manning_n_history": np.asarray(
            [
                _manning_at_depth(
                    depth,
                    roughness,
                    manning_table_depth,
                    manning_table_values,
                )
                for depth in h_history
            ]
        ),
        "bed_elevation_m": bed,
        "channel_width_m": reference_width,
        "channel_bottom_width_m": bottom_width,
        "side_slope_h_to_v": side_slope,
        "cross_section_depth_m": table_depth,
        "cross_section_top_width_m": table_width,
        "cross_section_wetted_perimeter_m": table_perimeter,
        "cross_section_shape": cross_section_shape,
        "top_width_history": _top_width(
            np.array(h_history),
            bottom_width[None, :],
            side_slope[None, :],
            table_depth,
            None if table_width is None else table_width[None, :, :],
        ),
        "cross_section_area_history": _cross_section_area(
            np.array(h_history),
            bottom_width[None, :],
            side_slope[None, :],
            table_depth,
            None if table_width is None else table_width[None, :, :],
        ),
        "uses_cross_section": uses_cross_section,
        "downstream_boundary": downstream_boundary,
        "spatial_order": spatial_order,
        "times": np.array(times),
        "h_history": np.array(h_history),
        "q_history": np.array(q_history),
        "upstream_flux_history": np.array(upstream_flux_history),
        "downstream_flux_history": np.array(downstream_flux_history),
        "h_initial": h_initial,
        "h_final": h,
        "q_initial": q_initial,
        "q_final": q,
        "mass_inflow": mass_inflow,
        "mass_source": mass_source,
        "mass_rainfall": mass_rainfall,
        "mass_lateral_inflow": mass_lateral_inflow,
        "mass_infiltration": mass_infiltration,
        "mass_outflow": mass_outflow,
        "mass_floor_correction": mass_floor_correction,
    }


def save_time_series_csv(result, path):
    """Write a depth-versus-time table compatible with animate_depth.py."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t"] + [f"{xi:.6f}" for xi in result["x"]])
        for time, depth_row in zip(result["times"], result["h_history"]):
            writer.writerow([f"{time:.6f}"] + [f"{depth:.10g}" for depth in depth_row])


class _SaintVenantSolver:
    name = "saint_venant"
    supports = frozenset(
        {
            "initial_depth",
            "initial_discharge",
            "left_inflow",
            "rainfall",
            "lateral_inflow",
            "cfl",
            "downstream_boundary",
            "downstream_stage",
            "spatial_order",
            "initial_cumulative_infiltration",
            "soil_infiltration",
        }
    )

    def run(self, domain: Domain, scenario: Scenario) -> SimulationResult:
        n_cells = len(domain.x_m)

        def state_array(value, name):
            array = np.asarray(value, dtype=float)
            if array.ndim == 0:
                array = np.full(n_cells, float(array), dtype=float)
            if array.shape != (n_cells,):
                raise ValueError(f"{name} must contain one value per domain cell")
            return array

        raw = run_model(
            float(np.sum(domain.dx_m)),
            scenario.t_final_min,
            record_interval=scenario.record_interval_min,
            h_init=state_array(scenario.initial_depth_m, "initial_depth_m"),
            q_init=state_array(scenario.initial_discharge, "initial_discharge"),
            left_inflow=scenario.left_inflow,
            rainfall=(
                scenario.rainfall
                if scenario.rainfall is not None
                else lambda x, t: np.zeros_like(x, dtype=float)
            ),
            lateral_inflow=scenario.lateral_inflow,
            x_m=domain.x_m,
            dx_m=domain.dx_m,
            slope=domain.slope,
            manning_n=domain.manning_n,
            bed_elevation_m=getattr(domain, "bed_elevation_m", None),
            channel_width_m=getattr(domain, "channel_width_m", None),
            channel_bottom_width_m=getattr(
                domain, "channel_bottom_width_m", None
            ),
            side_slope_h_to_v=getattr(domain, "side_slope_h_to_v", None),
            cross_section_depth_m=getattr(
                domain, "cross_section_depth_m", None
            ),
            cross_section_top_width_m=getattr(
                domain, "cross_section_top_width_m", None
            ),
            cross_section_wetted_perimeter_m=getattr(
                domain, "cross_section_wetted_perimeter_m", None
            ),
            manning_depth_m=getattr(domain, "manning_depth_m", None),
            manning_n_table=getattr(domain, "manning_n_table", None),
            soil_ksat_m_per_min=getattr(
                domain, "soil_ksat_m_per_min", None
            ),
            soil_suction_head_m=getattr(
                domain, "soil_suction_head_m", None
            ),
            soil_moisture_deficit=getattr(
                domain, "soil_moisture_deficit", None
            ),
            initial_cumulative_infiltration_m=(
                scenario.initial_cumulative_infiltration_m
            ),
            downstream_boundary=scenario.downstream_boundary,
            downstream_stage_m=scenario.downstream_stage_m,
            spatial_order=scenario.spatial_order,
            cfl=scenario.cfl,
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
                "channel_width_m": raw["channel_width_m"],
                "channel_bottom_width_m": raw["channel_bottom_width_m"],
                "side_slope_h_to_v": raw["side_slope_h_to_v"],
                "cross_section_depth_m": raw["cross_section_depth_m"],
                "cross_section_top_width_m": raw[
                    "cross_section_top_width_m"
                ],
                "cross_section_wetted_perimeter_m": raw[
                    "cross_section_wetted_perimeter_m"
                ],
                "cross_section_shape": raw["cross_section_shape"],
                "manning_depth_m": raw["manning_depth_m"],
                "manning_n_table": raw["manning_n_table"],
                "manning_n_history": raw["manning_n_history"],
                "top_width_history": raw["top_width_history"],
                "cross_section_area_history": raw[
                    "cross_section_area_history"
                ],
                "discharge_history": raw["q_history"],
                "upstream_flux_history": raw["upstream_flux_history"],
                "downstream_flux_history": raw[
                    "downstream_flux_history"
                ],
                "discharge_initial": raw["q_initial"],
                "discharge_final": raw["q_final"],
                "mass_rainfall": raw["mass_rainfall"],
                "mass_lateral_inflow": raw["mass_lateral_inflow"],
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


SOLVER = _SaintVenantSolver()


if __name__ == "__main__":
    result = run_model(L, T_final)
    plt.plot(result["x"], result["h_initial"], label="Initial")
    plt.plot(result["x"], result["h_final"], label=f"After t = {T_final}", ls="--")
    plt.legend()
    plt.xlabel("x (m)")
    plt.ylabel("h (m)")
    plt.savefig("data/saint_venant_1d.png")
    save_time_series_csv(result, "data/saint_venant_1d_timeseries.csv")
