import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = -1.0          # wave speed
L = 10.0         # domain length
Nx = 200         # number of cells
dx = L / Nx
CFL = 0.9        # Courant number (must be <= 1)
dt = CFL * dx / np.abs(c)
T_final = 5.0    # total simulation time
Nt = int(T_final / dt)
center = 8.0
nu = c * dt / dx

# Grid
x = np.linspace(0, L, Nx)

# Initial condition: a Gaussian pulse
u = np.exp(-((x - center)**2) / 0.2)
u_initial = u.copy()

# Time stepping with upwind scheme
for n in range(Nt):
    u_new = u.copy()
    if c > 0:
        u_new[1:] = u[1:] - nu * (u[1:] - u[:-1])
        u_new[0] = 0.0    # left boundary is inflow (nothing coming in)
    else:
        u_new[:-1] = u[:-1] - nu * (u[1:] - u[:-1])
        u_new[-1] = 0.0   # right boundary is now inflow
    u = u_new

# Plot
plt.plot(x, u_initial, label='Initial')
plt.plot(x, u, label=f'After t = {T_final}', ls = '--')
plt.axvline(center + c * T_final, linestyle='--', label='Expected position')
plt.legend(); plt.xlabel('x'); plt.ylabel('u')
plt.show()