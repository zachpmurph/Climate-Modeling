import csv

import numpy as np
import matplotlib.pyplot as plt

from general.solvers.contract import Domain, Scenario, SimulationResult

##NOTE: Units are in meters and minutes
#Parameters:
L = 10.0         # domain length
T_final = 300.0    # total simulation time
S0 = 0.05
n0 = 0.05


def r(x, t):
    if (t >=0 and t < 50):
        return 0.00002 * np.ones(len(x))
    return np.zeros(len(x))

def c(u):
    u = np.maximum(u, 0.0)  # prevents accidental negatives from error
    return (5/(3 * n0)) * (u ** (2/3)) * np.sqrt(S0)          # wave speed

def q(u):
    u = np.maximum(u, 0.0)  # prevents accidental negatives from error
    return (1/(n0)) * (u ** (5/3)) * np.sqrt(S0)          # wave speed 

#Runs the kinematic wave model
def run_model(L, T_final, record_interval=1.0):
    #Establish Domain
    Nx = int(L*10)       # number of cells
    dx = L / Nx
    x = np.linspace(dx/2, L-(dx/2), Nx)

    #Initial Condition
    center = 3.0
    u = 0.01 *np.exp(-((x - center)**2) / 0.2)
    u_initial = u.copy()
    t_current = 0

    # Mass bookkeeping over the interior control volume (cells 1..Nx-1).
    # Cell 0 is a boundary-condition cell (reset to ~0 every step) rather
    # than a physical control volume, so it's excluded from both sides of
    # the balance: mass_source only counts source added to cells 1..Nx-1,
    # and mass_outflow is the flux leaving through the right edge.
    mass_source = 0.0
    mass_outflow = 0.0

    # Record snapshots on a fixed wall-clock grid (every record_interval
    # minutes) rather than every adaptive dt, so the time series is easy
    # to animate/tabulate. record_times always includes t=0 and T_final.
    n_marks = int(np.floor(T_final / record_interval + 1e-9))
    record_times = [i * record_interval for i in range(n_marks + 1)]
    if record_times[-1] < T_final - 1e-9:
        record_times.append(T_final)

    times = [0.0]
    history = [u_initial.copy()]
    next_record_idx = 1

    while t_current < T_final:
        # Adaptive time step
        c_max = np.max(c(u))
        CFL = 0.5        # Courant number (must be <= 1)
        dt = CFL * dx / c_max
        if t_current + dt > T_final:
            dt = T_final - t_current
        # Also cap dt so we land exactly on the next recording mark instead
        # of stepping past it -- keeps the recorded snapshots exact rather
        # than approximated from whichever step happened to overshoot.
        if next_record_idx < len(record_times):
            dt = min(dt, record_times[next_record_idx] - t_current)

        # Conservative upwind update
        flux = q(u)
        u_new = u.copy()
        u_new[1:] = u[1:] - (dt/dx) * (flux[1:] - flux[:-1])
        u_new[0] = 1e-10   # zero-depth at watershed divide (adjust to taste)

        source = r(x, t_current)
        mass_outflow += flux[-1] * dt
        mass_source += np.sum(source[1:]) * dx * dt

        # Add source
        u = u_new + dt * source

        # Enforce non-negativity
        u = np.maximum(u, 1e-10)

        t_current += dt

        if next_record_idx < len(record_times) and t_current >= record_times[next_record_idx] - 1e-9:
            times.append(record_times[next_record_idx])
            history.append(u.copy())
            next_record_idx += 1

    return {
        "x": x,
        "times": np.array(times),
        "u_history": np.array(history),
        "u_initial": u_initial,
        "u_final": u,
        "mass_source": mass_source,
        "mass_outflow": mass_outflow,
    }

def save_time_series_csv(result, path):
    """Write the recorded (t, u(x)) table to a CSV: one row per recorded
    time, one column per spatial cell. Read back by src/general/viz/animate_depth.py."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t"] + [f"{xi:.6f}" for xi in result["x"]])
        for t, u_row in zip(result["times"], result["u_history"]):
            writer.writerow([f"{t:.6f}"] + [f"{ui:.10g}" for ui in u_row])

class _KinematicWaveSolver:
    name = "kinematic_wave"
    supports = frozenset({"rainfall"})

    def run(self, domain: Domain, scenario: Scenario) -> SimulationResult:
        import general.solvers.linear_advection as _la

        # Temporarily set globals from the Domain (uniform assumed: use first cell)
        _orig_S0, _orig_n0 = _la.S0, _la.n0
        _la.S0 = float(domain.slope[0])
        _la.n0 = float(domain.manning_n[0])

        # Inject per-scenario rainfall as module-level r if provided
        _orig_r = _la.r
        if scenario.rainfall is not None:
            _la.r = scenario.rainfall

        L = float(domain.x_m[-1] + domain.dx_m[-1] / 2)
        try:
            result = run_model(L, scenario.t_final_min, scenario.record_interval_min)
        finally:
            _la.S0 = _orig_S0
            _la.n0 = _orig_n0
            _la.r = _orig_r

        x_int = result["x"]
        dx_int = x_int[1] - x_int[0] if len(x_int) > 1 else domain.dx_m[0]
        internal_domain = Domain(
            x_m=x_int,
            dx_m=np.full_like(x_int, dx_int),
            slope=np.full_like(x_int, float(domain.slope[0])),
            manning_n=np.full_like(x_int, float(domain.manning_n[0])),
        )
        return SimulationResult(
            domain=internal_domain,
            times=result["times"],
            depth_history=result["u_history"],
            depth_initial=result["u_initial"],
            depth_final=result["u_final"],
            mass_inflow=0.0,
            mass_source=result["mass_source"],
            mass_outflow=result["mass_outflow"],
        )


SOLVER = _KinematicWaveSolver()


if __name__ == "__main__":
    result = run_model(L, T_final)
    # Plot
    plt.plot(result["x"], result["u_initial"], label='Initial')
    plt.plot(result["x"], result["u_final"], label=f'After t = {T_final}', ls = '--')
    plt.legend(); plt.xlabel('x'); plt.ylabel('u')
    plt.savefig("data/linear_advection.png")

    save_time_series_csv(result, "data/linear_advection_timeseries.csv")

