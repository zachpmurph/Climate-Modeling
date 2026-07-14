import numpy as np
import matplotlib.pyplot as plt

##NOTE: Units are in meters and minutes
#Parameters:
L = 10.0         # domain length
Nx = int(L*10)       # number of cells
dx = L / Nx
x = np.linspace(dx/2, L-(dx/2), Nx)
CFL = 0.5        # Courant number (must be <= 1)
T_final = 10.0    # total simulation time
t_current = 0
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

# Initial condition: a Gaussian pulse
center = 3.0
u = 0.01 *np.exp(-((x - center)**2) / 0.2)
u_initial = u.copy()

while t_current < T_final:
    # Adaptive time step
    c_max = np.max(c(u))
    dt = CFL * dx / c_max
    if t_current + dt > T_final:
        dt = T_final - t_current
    
    # Conservative upwind update
    flux = q(u)
    u_new = u.copy()
    u_new[1:] = u[1:] - (dt/dx) * (flux[1:] - flux[:-1])
    u_new[0] = 1e-10   # zero-depth at watershed divide (adjust to taste)
    
    # Add source
    u = u_new + dt * r(x, t_current)
    
    # Enforce non-negativity
    u = np.maximum(u, 1e-10)
    
    t_current += dt

# Plot
plt.plot(x, u_initial, label='Initial')
plt.plot(x, u, label=f'After t = {T_final}', ls = '--')
plt.legend(); plt.xlabel('x'); plt.ylabel('u')
plt.savefig("graphs/linear_advection.png")