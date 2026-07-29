import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from general.solvers.contract import Domain, Scenario, SimulationResult


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


def _velocity(h, discharge, width):
    velocity = np.zeros_like(discharge, dtype=float)
    area = width * h
    np.divide(discharge, area, out=velocity, where=area > H_FLOOR)
    return velocity


def _physical_flux(h, discharge, width):
    velocity = _velocity(h, discharge, width)
    area = width * h
    return discharge, discharge * velocity + 0.5 * g * width * h**2


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
        ghost_discharge = discharge[-1] * ghost_depth / h[-1]
    return ghost_depth, ghost_discharge


def _bed_from_slope(x, slope):
    bed = np.zeros_like(x, dtype=float)
    if len(x) > 1:
        bed[1:] = -np.cumsum(
            0.5 * (slope[:-1] + slope[1:]) * np.diff(x)
        )
    return bed


def _hydrostatic_states(h_left, q_left, bed_left, h_right, q_right, bed_right):
    face_bed = np.maximum(bed_left, bed_right)
    depth_left = np.maximum(0.0, h_left + bed_left - face_bed)
    depth_right = np.maximum(0.0, h_right + bed_right - face_bed)
    scale_left = np.zeros_like(h_left)
    scale_right = np.zeros_like(h_right)
    np.divide(depth_left, h_left, out=scale_left, where=h_left > H_FLOOR)
    np.divide(depth_right, h_right, out=scale_right, where=h_right > H_FLOOR)
    return depth_left, q_left * scale_left, depth_right, q_right * scale_right


def _rusanov_fluxes(
    h,
    discharge,
    bed,
    width,
    left_inflow,
    t,
    downstream_boundary,
    downstream_stage_m,
):
    inflow = _left_discharge(left_inflow, t)
    right_h, right_q = _downstream_ghost(
        h,
        discharge,
        bed,
        downstream_boundary,
        downstream_stage_m,
        t,
    )

    # Equal ghost/interior depths remove numerical mass diffusion at each
    # boundary. Mirroring q around the requested inflow makes the left face
    # mass flux exactly equal to that prescribed discharge.
    h_ext = np.concatenate(([h[0]], h, [right_h]))
    q_ext = np.concatenate(
        ([2.0 * inflow - discharge[0]], discharge, [right_q])
    )
    bed_ext = np.concatenate(([bed[0]], bed, [bed[-1]]))
    width_ext = np.concatenate(([width[0]], width, [width[-1]]))
    h_left, h_right = h_ext[:-1], h_ext[1:]
    q_left, q_right = q_ext[:-1], q_ext[1:]
    bed_left, bed_right = bed_ext[:-1], bed_ext[1:]
    width_left, width_right = width_ext[:-1], width_ext[1:]
    face_width = 0.5 * (width_left + width_right)

    hs_left, qs_left, hs_right, qs_right = _hydrostatic_states(
        h_left, q_left, bed_left, h_right, q_right, bed_right
    )
    # Reconstruct a common face area while retaining each side's velocity.
    qs_left *= face_width / width_left
    qs_right *= face_width / width_right
    flux_h_left, flux_q_left = _physical_flux(hs_left, qs_left, face_width)
    flux_h_right, flux_q_right = _physical_flux(hs_right, qs_right, face_width)
    speed_left = np.abs(_velocity(hs_left, qs_left, face_width)) + np.sqrt(g * hs_left)
    speed_right = np.abs(_velocity(hs_right, qs_right, face_width)) + np.sqrt(g * hs_right)
    alpha = np.maximum(speed_left, speed_right)

    area_left = face_width * hs_left
    area_right = face_width * hs_right
    flux_h = 0.5 * (flux_h_left + flux_h_right) - 0.5 * alpha * (
        area_right - area_left
    )
    flux_q = 0.5 * (flux_q_left + flux_q_right) - 0.5 * alpha * (qs_right - qs_left)
    correction_left = 0.5 * g * width_left * (h_left**2 - hs_left**2)
    correction_right = 0.5 * g * width_right * (h_right**2 - hs_right**2)
    geometry_balance = 0.5 * g * (
        (face_width[1:] - width) * hs_left[1:] ** 2
        - (face_width[:-1] - width) * hs_right[:-1] ** 2
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


def run_model(
    L,
    T_final,
    record_interval=0.05,
    h_init=None,
    q_init=None,
    left_inflow=None,
    rainfall=None,
    x_m=None,
    dx_m=None,
    slope=None,
    manning_n=None,
    bed_elevation_m=None,
    channel_width_m=None,
    downstream_boundary="outflow",
    downstream_stage_m=None,
    cfl=None,
):
    """Run the 1D Saint-Venant equations on a uniform or supplied cell grid.

    left_inflow is either a non-negative discharge or a callable of time
    returning that discharge. With ``channel_width_m`` it is total flow in
    m^3/min; the legacy unit-width mode uses m^2/min. None is a no-inflow upstream
    boundary. rainfall is a callable ``rainfall(x_m, t_min)`` returning one
    non-negative rate in m/min per cell. The downstream boundary is
    zero-gradient free outflow.
    """
    x, dx, bed_slope, roughness = _prepare_grid(
        L,
        x_m,
        dx_m,
        slope,
        manning_n,
    )
    n_cells = len(x)
    bed = (
        _bed_from_slope(x, bed_slope)
        if bed_elevation_m is None
        else _cell_values(bed_elevation_m, 0.0, n_cells, "bed_elevation_m")
    )
    width = _cell_values(channel_width_m, 1.0, n_cells, "channel_width_m")
    if np.any(width <= 0):
        raise ValueError("channel_width_m values must be positive")
    uses_cross_section = channel_width_m is not None
    _validate_inputs(T_final, record_interval, n_cells, h_init, q_init)
    cfl_value = CFL if cfl is None else float(cfl)
    if not np.isfinite(cfl_value) or not (0 < cfl_value <= 1):
        raise ValueError("cfl must be finite and in the interval (0, 1]")
    _downstream_ghost(
        np.ones(n_cells),
        np.zeros(n_cells),
        bed,
        downstream_boundary,
        downstream_stage_m,
        0.0,
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
    record_times = _record_times(T_final, record_interval)
    times = [0.0]
    h_history = [h.copy()]
    q_history = [q.copy()]
    next_record_idx = 1

    mass_inflow = 0.0
    mass_source = 0.0
    mass_outflow = 0.0
    mass_floor_correction = 0.0
    t_current = 0.0

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
            width,
            left_inflow,
            t_current,
            downstream_boundary,
            downstream_stage_m,
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
            downstream_stage_m,
        )

        h_previous = h.copy()
        q_previous = q.copy()
        source = _evaluate_rainfall(rainfall_function, x, t_current)
        area = width * h
        area_source = width * source
        flux_h, flux_q = _limit_draining_fluxes(
            area, area_source, dt, dx, flux_h, flux_q
        )

        area_new = (
            area
            - (dt / dx) * (flux_h[1:] - flux_h[:-1])
            + dt * area_source
        )
        q_new = q - (dt / dx) * (
            (flux_q + correction_left)[1:]
            - (flux_q + correction_right)[:-1]
            - geometry_balance
        )

        floor_addition = np.maximum(-area_new, 0.0)
        mass_floor_correction += float(np.sum(floor_addition * dx))
        area_new = np.maximum(area_new, 0.0)
        h_new = area_new / width

        velocity_new = _velocity(h_new, q_new, width)
        friction_coeff = np.zeros_like(h_new)
        wet = h_new > H_FLOOR
        hydraulic_radius = h_new.copy()
        if uses_cross_section:
            hydraulic_radius[wet] = (
                width[wet] * h_new[wet] / (width[wet] + 2.0 * h_new[wet])
            )
        friction_coeff[wet] = (
            roughness[wet] ** 2
            * np.abs(velocity_new[wet])
            / hydraulic_radius[wet] ** (4.0 / 3.0)
        )
        q_new = q_new / (1.0 + dt * g * friction_coeff)
        q_new[h_new <= H_FLOOR] = 0.0

        mass_inflow += float(flux_h[0] * dt)
        downstream_mass = float(flux_h[-1] * dt)
        if downstream_mass >= 0.0:
            mass_outflow += downstream_mass
        else:
            mass_inflow -= downstream_mass
        mass_source += float(np.sum(area_source * dx) * dt)

        h, q = h_new, q_new
        t_next = t_current + dt
        while (
            next_record_idx < len(record_times)
            and record_times[next_record_idx] <= t_next + 1e-10
        ):
            record_time = record_times[next_record_idx]
            fraction = 1.0 if dt == 0 else (record_time - t_current) / dt
            fraction = min(max(fraction, 0.0), 1.0)
            times.append(record_time)
            h_history.append(h_previous + fraction * (h_new - h_previous))
            q_history.append(q_previous + fraction * (q_new - q_previous))
            next_record_idx += 1
        t_current = t_next

    return {
        "x": x,
        "dx_m": dx,
        "slope": bed_slope,
        "manning_n": roughness,
        "bed_elevation_m": bed,
        "channel_width_m": width,
        "uses_cross_section": uses_cross_section,
        "downstream_boundary": downstream_boundary,
        "times": np.array(times),
        "h_history": np.array(h_history),
        "q_history": np.array(q_history),
        "h_initial": h_initial,
        "h_final": h,
        "q_initial": q_initial,
        "q_final": q,
        "mass_inflow": mass_inflow,
        "mass_source": mass_source,
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
            "cfl",
            "downstream_boundary",
            "downstream_stage",
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
            x_m=domain.x_m,
            dx_m=domain.dx_m,
            slope=domain.slope,
            manning_n=domain.manning_n,
            bed_elevation_m=getattr(domain, "bed_elevation_m", None),
            channel_width_m=getattr(domain, "channel_width_m", None),
            downstream_boundary=scenario.downstream_boundary,
            downstream_stage_m=scenario.downstream_stage_m,
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
                "cross_section_area_history": (
                    raw["h_history"] * raw["channel_width_m"][None, :]
                ),
                "discharge_history": raw["q_history"],
                "discharge_initial": raw["q_initial"],
                "discharge_final": raw["q_final"],
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
