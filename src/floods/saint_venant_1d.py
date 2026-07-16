import csv

import numpy as np
import matplotlib.pyplot as plt

# Units: meters and minutes throughout
L = 10.0
T_final = 300.0
S0 = 0.05
n0 = 0.05
g = 35316.0   # 9.81 m/s^2 converted: 9.81 * 60^2 = 35316 m/min^2


def r(x, t):
    if 0 <= t < 50:
        return 0.00002 * np.ones(len(x))
    return np.zeros(len(x))


def run_model(L, T_final, record_interval=1.0, h_init=None, q_init=None):
    Nx = int(L * 10)
    dx = L / Nx
    x = np.linspace(dx / 2, L - dx / 2, Nx)

    h_floor = 1e-10
    CFL = 0.5

    if h_init is None:
        center = 3.0
        h = 0.01 * np.exp(-((x - center) ** 2) / 0.2)
    else:
        h = h_init.copy()

    if q_init is None:
        q = np.zeros(Nx)
    else:
        q = q_init.copy()

    h_initial = h.copy()
    q_initial = q.copy()

    mass_source = 0.0
    mass_outflow = 0.0

    n_marks = int(np.floor(T_final / record_interval + 1e-9))
    record_times = [i * record_interval for i in range(n_marks + 1)]
    if record_times[-1] < T_final - 1e-9:
        record_times.append(T_final)

    times = [0.0]
    h_history = [h_initial.copy()]
    q_history = [q_initial.copy()]
    next_record_idx = 1
    t_current = 0.0

    while t_current < T_final:
        h[0] = h_floor
        q[0] = 0.0
        vel = np.where(h > h_floor, q / h, 0.0)
        c_wave = np.sqrt(g * np.maximum(h, 0.0))
        max_speed = np.max(np.abs(vel) + c_wave)
        if max_speed < 1e-12:
            max_speed = 1e-12
        dt = CFL * dx / max_speed
        if t_current + dt > T_final:
            dt = T_final - t_current
        if next_record_idx < len(record_times):
            dt = min(dt, record_times[next_record_idx] - t_current)

        # Physical fluxes at time n
        F_h = q.copy()
        F_q = q ** 2 / np.maximum(h, h_floor) + 0.5 * g * h ** 2

        # Mass bookkeeping
        source = r(x, t_current)
        mass_source += np.sum(source[1:-1]) * dx * dt

        # LxF boundary fluxes for mass tracking.
        # F_left: LxF numerical flux at the left interior face (cell 0 → cell 1).
        # Negative means mass flows leftward out of the interior (left-BC leakage).
        F_left_face = 0.5 * q[1] - 0.5 * (dx / dt) * (h[1] - h_floor)
        mass_outflow += (q[-2] - F_left_face) * dt

        # Lax-Friedrichs flux update (interior cells 1..Nx-2)
        h_new = h.copy()
        q_new = q.copy()
        h_new[1:-1] = (0.5 * (h[2:] + h[:-2])
                       - 0.5 * (dt / dx) * (F_h[2:] - F_h[:-2]))
        q_new[1:-1] = (0.5 * (q[2:] + q[:-2])
                       - 0.5 * (dt / dx) * (F_q[2:] - F_q[:-2]))

        # Boundary conditions
        h_new[0] = h_floor   # no-inflow left BC
        q_new[0] = 0.0
        h_new[-1] = h_new[-2]   # zero-gradient right BC (free outflow)
        q_new[-1] = q_new[-2]

        # Operator-split source terms (interior cells only)
        h_new[1:-1] += dt * source[1:-1]
        h_new[-1] = h_new[-2]

        vel_new = np.where(h_new > h_floor, q_new / h_new, 0.0)
        # Semi-implicit friction: bed slope explicit, friction implicit.
        # Prevents stiff blow-up in near-dry cells where vel = q/h can be large.
        friction_coeff = n0 ** 2 * np.abs(vel_new) / np.maximum(h_new, h_floor) ** (4 / 3)
        denom = 1.0 + dt * g * friction_coeff
        q_new = (q_new + dt * g * h_new * S0) / denom

        # Safeguards
        h_new = np.maximum(h_new, h_floor)
        q_new = np.where(h_new <= h_floor, 0.0, q_new)

        h = h_new
        q = q_new
        t_current += dt

        if (next_record_idx < len(record_times)
                and t_current >= record_times[next_record_idx] - 1e-9):
            times.append(record_times[next_record_idx])
            h_history.append(h.copy())
            q_history.append(q.copy())
            next_record_idx += 1

    return {
        "x": x,
        "times": np.array(times),
        "h_history": np.array(h_history),
        "q_history": np.array(q_history),
        "h_initial": h_initial,
        "h_final": h,
        "q_initial": q_initial,
        "q_final": q,
        "mass_source": mass_source,
        "mass_outflow": mass_outflow,
    }


def save_time_series_csv(result, path):
    """Write (t, h(x)) table to CSV: one row per recorded time, one column
    per cell. Same format as linear_advection_timeseries.csv — compatible
    with src/tools/animate_depth.py."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t"] + [f"{xi:.6f}" for xi in result["x"]])
        for t, h_row in zip(result["times"], result["h_history"]):
            writer.writerow([f"{t:.6f}"] + [f"{hi:.10g}" for hi in h_row])


if __name__ == "__main__":
    result = run_model(L, T_final)
    plt.plot(result["x"], result["h_initial"], label="Initial")
    plt.plot(result["x"], result["h_final"], label=f"After t = {T_final}", ls="--")
    plt.legend()
    plt.xlabel("x (m)")
    plt.ylabel("h (m)")
    plt.savefig("data/saint_venant_1d.png")
    save_time_series_csv(result, "data/saint_venant_1d_timeseries.csv")
