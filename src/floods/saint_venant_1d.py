import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Units: meters and minutes throughout.
L = 10.0
T_final = 300.0
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


def _velocity(h, q):
    velocity = np.zeros_like(q, dtype=float)
    np.divide(q, h, out=velocity, where=h > H_FLOOR)
    return velocity


def _physical_flux(h, q):
    velocity = _velocity(h, q)
    return q, q * velocity + 0.5 * g * h**2


def _left_discharge(left_inflow, t):
    value = left_inflow(t) if callable(left_inflow) else left_inflow
    value = 0.0 if value is None else float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("left_inflow must return a finite, non-negative discharge")
    return value


def _rusanov_fluxes(h, q, left_inflow, t):
    inflow = _left_discharge(left_inflow, t)

    # Equal ghost/interior depths remove numerical mass diffusion at each
    # boundary. Mirroring q around the requested inflow makes the left face
    # mass flux exactly equal to that prescribed discharge.
    h_ext = np.concatenate(([h[0]], h, [h[-1]]))
    q_ext = np.concatenate(([2.0 * inflow - q[0]], q, [q[-1]]))
    h_left, h_right = h_ext[:-1], h_ext[1:]
    q_left, q_right = q_ext[:-1], q_ext[1:]

    flux_h_left, flux_q_left = _physical_flux(h_left, q_left)
    flux_h_right, flux_q_right = _physical_flux(h_right, q_right)
    speed_left = np.abs(_velocity(h_left, q_left)) + np.sqrt(g * h_left)
    speed_right = np.abs(_velocity(h_right, q_right)) + np.sqrt(g * h_right)
    alpha = np.maximum(speed_left, speed_right)

    flux_h = 0.5 * (flux_h_left + flux_h_right) - 0.5 * alpha * (h_right - h_left)
    flux_q = 0.5 * (flux_q_left + flux_q_right) - 0.5 * alpha * (q_right - q_left)
    return flux_h, flux_q, alpha


def _validate_inputs(domain_length, final_time, record_interval, h_init, q_init):
    if not np.isfinite(domain_length) or domain_length < 0.2:
        raise ValueError("L must be finite and at least 0.2 m")
    if not np.isfinite(final_time) or final_time < 0:
        raise ValueError("T_final must be finite and non-negative")
    if not np.isfinite(record_interval) or record_interval <= 0:
        raise ValueError("record_interval must be finite and positive")

    nx = int(domain_length * 10)
    for name, values in (("h_init", h_init), ("q_init", q_init)):
        if values is None:
            continue
        array = np.asarray(values, dtype=float)
        if array.shape != (nx,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain {nx} finite values")
    if h_init is not None and np.any(np.asarray(h_init) < 0):
        raise ValueError("h_init cannot contain negative depths")
    return nx


def _record_times(final_time, record_interval):
    count = int(np.floor(final_time / record_interval + 1e-9))
    values = [index * record_interval for index in range(count + 1)]
    if values[-1] < final_time - 1e-9:
        values.append(float(final_time))
    return values


def run_model(
    L,
    T_final,
    record_interval=1.0,
    h_init=None,
    q_init=None,
    left_inflow=None,
):
    """Run the 1D Saint-Venant equations with unit-width discharge.

    left_inflow is either a non-negative discharge in m^2/min or a callable
    of time returning that discharge. None is a closed/no-inflow upstream
    boundary. The downstream boundary is zero-gradient free outflow.
    """
    nx = _validate_inputs(L, T_final, record_interval, h_init, q_init)
    dx = L / nx
    x = np.linspace(dx / 2, L - dx / 2, nx)

    if h_init is None:
        h = 0.01 * np.exp(-((x - 3.0) ** 2) / 0.2)
    else:
        h = np.asarray(h_init, dtype=float).copy()
    h = np.maximum(h, H_FLOOR)

    if q_init is None:
        q = np.zeros(nx)
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
        flux_h, flux_q, interface_speed = _rusanov_fluxes(h, q, left_inflow, t_current)
        max_speed = max(float(np.max(interface_speed)), 1e-12)
        dt = min(CFL * dx / max_speed, T_final - t_current)
        if t_current < 50 < t_current + dt:
            dt = 50 - t_current

        h_previous = h.copy()
        q_previous = q.copy()
        source = np.asarray(r(x, t_current), dtype=float)
        if source.shape != h.shape or not np.all(np.isfinite(source)):
            raise ValueError("r(x, t) must return one finite source value per cell")

        h_new = h - (dt / dx) * (flux_h[1:] - flux_h[:-1]) + dt * source
        q_new = q - (dt / dx) * (flux_q[1:] - flux_q[:-1])

        floor_addition = np.maximum(H_FLOOR - h_new, 0.0)
        mass_floor_correction += float(np.sum(floor_addition) * dx)
        h_new = np.maximum(h_new, H_FLOOR)

        velocity_new = _velocity(h_new, q_new)
        friction_coeff = n0**2 * np.abs(velocity_new) / h_new ** (4.0 / 3.0)
        q_new = (q_new + dt * g * h_new * S0) / (1.0 + dt * g * friction_coeff)
        q_new[h_new <= H_FLOOR] = 0.0

        mass_inflow += float(flux_h[0] * dt)
        mass_outflow += float(flux_h[-1] * dt)
        mass_source += float(np.sum(source) * dx * dt)

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

        h, q = h_new, q_new
        t_current = t_next

    return {
        "x": x,
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


if __name__ == "__main__":
    result = run_model(L, T_final)
    plt.plot(result["x"], result["h_initial"], label="Initial")
    plt.plot(result["x"], result["h_final"], label=f"After t = {T_final}", ls="--")
    plt.legend()
    plt.xlabel("x (m)")
    plt.ylabel("h (m)")
    plt.savefig("data/saint_venant_1d.png")
    save_time_series_csv(result, "data/saint_venant_1d_timeseries.csv")
