import numpy as np
import matplotlib.pyplot as plt

##NOTE: Units are in meters and minutes
#Parameters:
L = 10.0         # domain length
T_final = 10.0    # total simulation time
S0 = 0.05
n0 = 0.05


def r(x, t):
    if (t >=0 and t < 50):
        return 0.00002 * (L-x)
    return 0

def c(u):
    u = np.maximum(u, 0.0)  # prevents accidental negatives from error
    return (5/(3 * n0)) * (u ** (2/3)) * np.sqrt(S0)          # wave speed

def q(u):
    u = np.maximum(u, 0.0)  # prevents accidental negatives from error
    return (1/(n0)) * (u ** (5/3)) * np.sqrt(S0)          # wave speed 

#Runs the kinematic wave model
def run_model(L, T_final):
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

    while t_current < T_final:
        # Adaptive time step
        c_max = np.max(c(u))
        CFL = 0.5        # Courant number (must be <= 1)
        dt = CFL * dx / c_max
        if t_current + dt > T_final:
            dt = T_final - t_current

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

    return {
        "x": x,
        "u_initial": u_initial,
        "u_final": u,
        "mass_source": mass_source,
        "mass_outflow": mass_outflow,
    }

if __name__ == "__main__":
    result = run_model(L, T_final)
    # Plot
    plt.plot(result["x"], result["u_initial"], label='Initial')
    plt.plot(result["x"], result["u_final"], label=f'After t = {T_final}', ls = '--')
    plt.legend(); plt.xlabel('x'); plt.ylabel('u')
    plt.savefig("graphs/linear_advection.png")

