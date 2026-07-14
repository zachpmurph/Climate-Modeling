import numpy as np
import matplotlib.pyplot as plt

##NOTE: Units are in meters and minutes

# Parameters
S0 = 0.05
n0 = 0.05
def c(u):
    #return np.ones(len(u))
    return (5/(3 * n0)) * (u ** (2/3)) * np.sqrt(S0)          # wave speed
L = 10.0         # domain length
Nx = int(L*10)       # number of cells
dx = L / (Nx - 1)
CFL = 0.5        # Courant number (must be <= 1)
u_max_expected = 1.0  # or however big you expect u to get
dt = CFL * dx / c(u_max_expected) 
T_final = 10.0    # total simulation time
Nt = int(T_final / dt)
center = 3.0
#nu = c * dt / dx
def r(x, t):
    if (t >=0 and t < 50):
        return 0.00002 * (L-x)
    return 0

# Grid
x = np.linspace(0, L, Nx)

# Initial condition: a Gaussian pulse
u = 0.01 *np.exp(-((x - center)**2) / 0.2)
#u = np.zeros(len(x))
u_initial = u.copy()

# Time stepping with upwind scheme
for n in range(Nt):
    u_new = u.copy()
    nu = c(u) * dt/dx
    #if c > 0:
    u_new[1:] = u[1:] - nu[1:] * (u[1:] - u[:-1])
    u_new[0] = u[0]    # left boundary is inflow (nothing coming in), right boundary is outflow
    u = u_new + (dt * r(x, n *dt))

# Analytic solution for constant source g0 over the whole run
g0 = 0.1
#u_analytic = g0 * T_final * np.ones_like(x)
#u_analytic[x < c * T_final] = g0 * (x[x < c * T_final] / c)  # ramp near boundary

# Verification
#l2_error = np.sqrt(np.mean((u - u_analytic)**2))
#print(f"L2 error: {l2_error:.4e}")

# Plot
plt.plot(x, u_initial, label='Initial')
plt.plot(x, u, label=f'After t = {T_final}', ls = '--')
plt.legend(); plt.xlabel('x'); plt.ylabel('u')
plt.show()