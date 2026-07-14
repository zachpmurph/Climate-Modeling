# Stage 1: Linear Advection with Source Term

## Overview

This stage extends the linear advection equation from stage 0 by adding a source term. It models the transport of a conserved quantity at constant speed along a 1D domain, with continuous addition of that quantity at every point in space and time.

The purpose of this stage is to verify that source terms are correctly implemented in the numerical scheme before moving to nonlinear closures in stage 2.

## Governing Equation

The PDE is:

$$\frac{\partial u}{\partial t} + c \, \frac{\partial u}{\partial x} = g(x, t)$$

where $u(x, t)$ is the conserved quantity, $c$ is the constant wave speed, and $g(x, t)$ is the source term with units of $u$ per unit time.

Initial condition: $u(x, 0) = 0$ for the verification test (the code also supports a Gaussian pulse initial condition).

Boundary condition: $u(0, t) = 0$ (no inflow at the left boundary, appropriate for $c > 0$).

## Analytic Solution

The method of characteristics gives:

$$u(x, t) = f(x - ct) + \int_0^t g(x - c(t - \tau), \tau) \, d\tau$$

For the verification test with $f = 0$ and constant source $g(x, t) = g_0$:

- Where $x > ct$ (initial-condition region unaffected by the boundary): $u(x, t) = g_0 t$
- Where $x < ct$ (boundary-influenced region): $u(x, t) = g_0 \cdot (x/c)$

The second case arises because the left boundary enforces $u = 0$, and characteristics reaching a point $x < ct$ originated at the boundary at time $t - x/c$, so they have been accumulating source for only $x/c$ units of time.

## Numerical Method

**Spatial discretization.** First-order upwind finite differences on a cell-centered grid. For $c > 0$:

$$\left. \frac{\partial u}{\partial x} \right|_i \approx \frac{u_i - u_{i-1}}{\Delta x}$$

**Time discretization.** Explicit forward Euler with first-order operator splitting between the advection and source steps:

$$u_i^* = u_i^n - \frac{c \, \Delta t}{\Delta x} (u_i^n - u_{i-1}^n)$$

$$u_i^{n+1} = u_i^* + \Delta t \cdot g(x_i, t^n)$$

**Stability.** The Courant-Friedrichs-Lewy (CFL) condition:

$$C = \frac{|c| \, \Delta t}{\Delta x} \leq 1$$

The time step is set to $\Delta t = 0.9 \cdot \Delta x / |c|$ to remain safely inside the stability region.

## Verification

**Test case.**
- Domain: $x \in [0, 20]$
- Grid: 200 cells, $\Delta x = 0.1$
- Wave speed: $c = 1.0$
- Source: $g_0 = 0.1$ for $0 \leq t < 5$
- Initial condition: $u(x, 0) = np.exp(-((x - center)**2) / 0.2)$
- Simulation time: $T = 2.0$

**Result.** At $t = 2$, the numerical solution shows a constant value of $g_0 T = 0.2$ in the region $x > cT = 2$, and a linear ramp from 0 to 0.2 in the region $x < 2$, matching the analytic prediction. L² error against the analytic solution: 1.5936e-01.

The plot `figures/verification.png` shows the numerical solution overlaid with the analytic solution and the initial condition.

## Files

- `linear_advection.py` — Solver and verification test in a single script.

## How to Run

```
python linear_advection.py

```

Prints the L² error against the analytic solution and displays the comparison plot.

## Physical Interpretation

Although this stage does not model a specific physical system, the equation form applies directly to several real problems:

- Passive tracer transport in a river with distributed input (e.g., dye or pollutant added along the length)
- Simplified overland flow at constant velocity: $u$ is depth, $c$ is drainage speed, $g$ is rainfall
- Traffic flow (linear approximation) with continuous on-ramp input

The overland flow case is the direct predecessor of the kinematic wave equation implemented in stage 2, with the difference that $c$ will become a function of $u$.

## Limitations

- Wave speed $c$ is constant, so no wave steepening, no shocks, no nonlinear behavior.
- First-order upwind is dissipative: sharp features smear over time. This is expected and does not affect this stage's verification, but will matter when discontinuities appear in later stages.
- No spatial variation in the source in the current test. The `source()` function supports spatially varying input; only the test case uses a uniform source.
- Boundary conditions are simple. Real problems require inflow boundary specification (Dirichlet, Neumann, or transmissive), which will become more important in later stages.

## Next Stage

Stage 2 introduces the nonlinear closure $c = c(u)$, giving the kinematic wave equation. This is where genuinely new behavior emerges: wave steepening, characteristic convergence, and shock formation. The numerical scheme structure from stages 0 and 1 carries over, with the modification that the flux depends nonlinearly on the state.