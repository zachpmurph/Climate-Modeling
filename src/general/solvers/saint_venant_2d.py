"""2D shallow-water (Saint-Venant) solver.

Conservative finite-volume scheme with Rusanov (local Lax-Friedrichs) numerical
fluxes in both x and y, an adaptive CFL time step, and a semi-implicit
gravity/friction source (the same stable treatment used by saint_venant_1d.py).

State per cell: water depth ``h`` and unit-width momenta ``hu`` (x) and ``hv`` (y),
on a cell-centred (nx, ny) grid where axis 0 is x and axis 1 is y.

Boundaries:
  * x = 0 : inflow — a prescribed unit-width discharge ``left_inflow`` (m^2/min),
    imposed by mirroring the hu ghost about it (equal-depth ghost).
  * x = L : open outflow — zero-gradient ghost.
  * y = 0, y = W : reflecting solid walls — equal-depth ghost with the normal
    momentum hv reflected (watertight) and tangential hu free-slip.

Units are meters and minutes throughout. Manning's n uses the meters-and-minutes
form (SI n divided by 60) and g is 9.81 m/s^2 converted to m/min^2, matching
saint_venant_1d.py; this file is internally consistent in that convention (see
docs/ingestion_integration_requests.md for the separate 1D-kinematic-wave note).
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from general.solvers.contract import Domain2D, Scenario, SimulationResult

# Units: meters and minutes throughout.
L = 10.0
W = 5.0
T_final = 30.0
S0x = 0.05
S0y = 0.0
MANNING_N_SECONDS = 0.05
n0 = MANNING_N_SECONDS / 60.0
g = 35316.0  # 9.81 m/s^2 converted to m/min^2.
H_FLOOR = 1e-10
CFL = 0.5


def r(t):
    """Uniform rainfall source rate (m/min) at time t."""
    return 0.1


def _velocity(h, mom):
    vel = np.zeros_like(mom, dtype=float)
    np.divide(mom, h, out=vel, where=h > H_FLOOR)
    return vel


def _wave_speed(h, mom):
    return np.abs(_velocity(h, mom)) + np.sqrt(g * np.maximum(h, 0.0))


def _flux_x(h, hu, hv):
    u = _velocity(h, hu)
    return hu, hu * u + 0.5 * g * h**2, hv * u


def _flux_y(h, hu, hv):
    v = _velocity(h, hv)
    return hv, hu * v, hv * v + 0.5 * g * h**2


def _rusanov_x(h, hu, hv, inflow):
    """Rusanov x-fluxes on all (nx+1, ny) x-faces, with inflow/outflow ghosts."""
    h_ext = np.concatenate([h[:1, :], h, h[-1:, :]], axis=0)
    hu_ext = np.concatenate([2.0 * inflow - hu[:1, :], hu, hu[-1:, :]], axis=0)
    hv_ext = np.concatenate([hv[:1, :], hv, hv[-1:, :]], axis=0)

    h_l, h_r = h_ext[:-1, :], h_ext[1:, :]
    hu_l, hu_r = hu_ext[:-1, :], hu_ext[1:, :]
    hv_l, hv_r = hv_ext[:-1, :], hv_ext[1:, :]

    fh_l, fhu_l, fhv_l = _flux_x(h_l, hu_l, hv_l)
    fh_r, fhu_r, fhv_r = _flux_x(h_r, hu_r, hv_r)
    alpha = np.maximum(_wave_speed(h_l, hu_l), _wave_speed(h_r, hu_r))

    fh = 0.5 * (fh_l + fh_r) - 0.5 * alpha * (h_r - h_l)
    fhu = 0.5 * (fhu_l + fhu_r) - 0.5 * alpha * (hu_r - hu_l)
    fhv = 0.5 * (fhv_l + fhv_r) - 0.5 * alpha * (hv_r - hv_l)
    return fh, fhu, fhv


def _rusanov_y(h, hu, hv):
    """Rusanov y-fluxes on all (nx, ny+1) y-faces, with reflecting-wall ghosts."""
    h_ext = np.concatenate([h[:, :1], h, h[:, -1:]], axis=1)
    hu_ext = np.concatenate([hu[:, :1], hu, hu[:, -1:]], axis=1)  # free-slip tangential
    hv_ext = np.concatenate([-hv[:, :1], hv, -hv[:, -1:]], axis=1)  # reflected normal

    h_b, h_t = h_ext[:, :-1], h_ext[:, 1:]
    hu_b, hu_t = hu_ext[:, :-1], hu_ext[:, 1:]
    hv_b, hv_t = hv_ext[:, :-1], hv_ext[:, 1:]

    gh_b, ghu_b, ghv_b = _flux_y(h_b, hu_b, hv_b)
    gh_t, ghu_t, ghv_t = _flux_y(h_t, hu_t, hv_t)
    alpha = np.maximum(_wave_speed(h_b, hv_b), _wave_speed(h_t, hv_t))

    gh = 0.5 * (gh_b + gh_t) - 0.5 * alpha * (h_t - h_b)
    ghu = 0.5 * (ghu_b + ghu_t) - 0.5 * alpha * (hu_t - hu_b)
    ghv = 0.5 * (ghv_b + ghv_t) - 0.5 * alpha * (hv_t - hv_b)
    return gh, ghu, ghv


def _record_times(final_time, record_interval):
    count = int(np.floor(final_time / record_interval + 1e-9))
    values = [index * record_interval for index in range(count + 1)]
    if not values or values[-1] < final_time - 1e-9:
        values.append(float(final_time))
    return values


def _grid(length, count):
    step = length / count
    centers = np.linspace(step / 2, length - step / 2, count)
    return centers, step


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
    cfl=CFL,
):
    if not (np.isfinite(L) and np.isfinite(W) and L > 0 and W > 0):
        raise ValueError("L and W must be finite and positive")
    if not np.isfinite(T_final) or T_final < 0:
        raise ValueError("T_final must be finite and non-negative")
    if record_interval <= 0:
        raise ValueError("record_interval must be positive")
    if not callable(left_inflow):
        inflow = float(left_inflow)
        if not np.isfinite(inflow) or inflow < 0:
            raise ValueError("left_inflow must be a finite, non-negative discharge")
    if not (np.isfinite(cfl) and 0 < cfl <= 1):
        raise ValueError("cfl must be finite and in (0, 1]")

    if x_m is None and y_m is None:
        nx, ny = int(L * 10), int(W * 10)
        x, dx_scalar = _grid(L, nx)
        y, dy_scalar = _grid(W, ny)
        dx = np.full(nx, dx_scalar)
        dy = np.full(ny, dy_scalar)
    elif x_m is not None and y_m is not None and dx_m is not None and dy_m is not None:
        x = np.asarray(x_m, dtype=float)
        y = np.asarray(y_m, dtype=float)
        dx = np.asarray(dx_m, dtype=float)
        dy = np.asarray(dy_m, dtype=float)
        nx, ny = len(x), len(y)
        if x.shape != dx.shape or y.shape != dy.shape:
            raise ValueError("x_m/dx_m and y_m/dy_m must have matching one-dimensional shapes")
        if np.any(dx <= 0) or np.any(dy <= 0):
            raise ValueError("All cell widths must be positive")
    else:
        raise ValueError("x_m, y_m, dx_m, and dy_m must be supplied together")

    shape = (nx, ny)

    def field(value, default, name):
        array = np.asarray(default if value is None else value, dtype=float)
        if array.ndim == 0:
            array = np.full(shape, float(array))
        if array.shape != shape:
            raise ValueError(f"{name} must be scalar or have shape {shape}")
        return array

    bed_slope_x = field(slope_x, S0x, "slope_x")
    bed_slope_y = field(slope_y, S0y, "slope_y")
    roughness = field(manning_n, n0, "manning_n")
    if np.any(roughness <= 0):
        raise ValueError("manning_n must be positive")
    cell_area = dx[:, None] * dy[None, :]

    h = field(h_init, 0.01, "h_init").copy()
    hu = field(hu_init, 0.0, "hu_init").copy()
    hv = field(hv_init, 0.0, "hv_init").copy()
    h = np.maximum(h, H_FLOOR)
    hu[h <= H_FLOOR] = 0.0
    hv[h <= H_FLOOR] = 0.0

    h_initial = h.copy()
    record_times = _record_times(T_final, record_interval)
    times = [0.0]
    h_history = [h.copy()]
    next_record_idx = 1

    mass_inflow = 0.0
    mass_outflow = 0.0
    mass_source = 0.0
    mass_floor_correction = 0.0
    t_current = 0.0

    while t_current < T_final - 1e-12:
        # Adaptive 2D CFL step; guard the denominator so an all-dry domain (tiny
        # wave speeds) can't divide by zero and hang.
        ax = float(np.max(_wave_speed(h, hu)))
        ay = float(np.max(_wave_speed(h, hv)))
        wave_denom = ax / dx[:, None] + ay / dy[None, :]
        dt = min(cfl / max(float(np.max(wave_denom)), 1e-12), T_final - t_current)
        if next_record_idx < len(record_times):
            dt = min(dt, record_times[next_record_idx] - t_current)

        inflow = left_inflow(t_current) if callable(left_inflow) else left_inflow
        inflow = np.asarray(inflow, dtype=float)
        if inflow.ndim == 0:
            inflow = np.full(ny, float(inflow))
        if inflow.shape != (ny,) or np.any(~np.isfinite(inflow)) or np.any(inflow < 0):
            raise ValueError(f"left_inflow must return a finite non-negative scalar or shape ({ny},)")

        fh, fhu, fhv = _rusanov_x(h, hu, hv, inflow)
        gh, ghu, ghv = _rusanov_y(h, hu, hv)

        div_h = ((fh[1:, :] - fh[:-1, :]) / dx[:, None]
                 + (gh[:, 1:] - gh[:, :-1]) / dy[None, :])
        div_hu = ((fhu[1:, :] - fhu[:-1, :]) / dx[:, None]
                  + (ghu[:, 1:] - ghu[:, :-1]) / dy[None, :])
        div_hv = ((fhv[1:, :] - fhv[:-1, :]) / dx[:, None]
                  + (ghv[:, 1:] - ghv[:, :-1]) / dy[None, :])

        source_value = r(t_current) if rainfall is None else rainfall(x, y, t_current)
        source = field(source_value, 0.0, "rainfall")
        if np.any(~np.isfinite(source)) or np.any(source < 0):
            raise ValueError("rainfall must return finite non-negative rates")
        h_new = h - dt * div_h + dt * source
        hu_star = hu - dt * div_hu
        hv_star = hv - dt * div_hv

        floor_addition = np.maximum(H_FLOOR - h_new, 0.0)
        mass_floor_correction += float(np.sum(floor_addition * cell_area))
        h_new = np.maximum(h_new, H_FLOOR)

        # Semi-implicit gravity (bed slope) + Manning friction, coupling u and v
        # through the flow speed |U| (evaluated on the post-flux velocities).
        u_s = _velocity(h_new, hu_star)
        v_s = _velocity(h_new, hv_star)
        speed = np.sqrt(u_s**2 + v_s**2)
        friction = roughness**2 * speed / h_new ** (4.0 / 3.0)
        denom_src = 1.0 + dt * g * friction
        hu_new = (hu_star + dt * g * h_new * bed_slope_x) / denom_src
        hv_new = (hv_star + dt * g * h_new * bed_slope_y) / denom_src
        hu_new[h_new <= H_FLOOR] = 0.0
        hv_new[h_new <= H_FLOOR] = 0.0

        mass_inflow += float(np.sum(fh[0, :] * dy) * dt)
        mass_outflow += float(np.sum(fh[-1, :] * dy) * dt)
        mass_source += float(np.sum(source * cell_area) * dt)

        h, hu, hv = h_new, hu_new, hv_new
        t_current += dt

        if next_record_idx < len(record_times) and t_current >= record_times[next_record_idx] - 1e-9:
            times.append(record_times[next_record_idx])
            h_history.append(h.copy())
            next_record_idx += 1

    return {
        "x": x,
        "y": y,
        "times": np.array(times),
        "h_history": np.array(h_history),  # (n_times, nx, ny)
        "h_initial": h_initial,
        "h_final": h,
        "hu_final": hu,
        "hv_final": hv,
        "mass_inflow": mass_inflow,
        "mass_outflow": mass_outflow,
        "mass_source": mass_source,
        "mass_floor_correction": mass_floor_correction,
    }


class _SaintVenant2DSolver:
    name = "saint_venant_2d"
    supports = frozenset({
        "initial_depth", "initial_discharge", "initial_discharge_y",
        "left_inflow", "rainfall", "rainfall_2d", "cfl",
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
                "discharge_x_final": raw["hu_final"],
                "discharge_y_final": raw["hv_final"],
                "mass_floor_correction": raw["mass_floor_correction"],
            },
        )


SOLVER = _SaintVenant2DSolver()


def save_time_series_csv(result, path):
    """Write a y-averaged depth-versus-time table compatible with animate_depth.py."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_profiles = np.mean(result["h_history"], axis=2)  # average over y -> (n_times, nx)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t"] + [f"{xi:.6f}" for xi in result["x"]])
        for time, depth_row in zip(result["times"], depth_profiles):
            writer.writerow([f"{time:.6f}"] + [f"{depth:.10g}" for depth in depth_row])


if __name__ == "__main__":
    result = run_model()
    h_init_profile = np.mean(result["h_initial"], axis=1)
    h_final_profile = np.mean(result["h_final"], axis=1)
    plt.plot(result["x"], h_init_profile, label="Initial")
    plt.plot(result["x"], h_final_profile, label=f"After t = {T_final}", ls="--")
    plt.legend()
    plt.xlabel("x (m)")
    plt.ylabel("y-averaged h (m)")
    plt.savefig("data/saint_venant_2d.png")
    save_time_series_csv(result, "data/saint_venant_2d_timeseries.csv")
