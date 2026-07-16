# Climate-Modeling

A repository for climate-related numerical models. It will eventually house several
independent models; the only one under active development right now is a 1D flood
model built up in stages from a simple linear advection solver toward a full
kinematic wave overland-flow model.

## Current Model: Kinematic Wave Overland Flow

`src/floods/linear_advection.py` currently solves the kinematic wave equation for
1D overland flow depth $h(x, t)$:

$$\frac{\partial h}{\partial t} + \frac{\partial q(h)}{\partial x} = r(x, t)$$

with a Manning's-equation closure relating flux and wave speed to depth:

$$q(h) = \frac{1}{n_0} h^{5/3} \sqrt{S_0}, \qquad c(h) = \frac{dq}{dh} = \frac{5}{3 n_0} h^{2/3} \sqrt{S_0}$$

where $S_0$ is the bed slope, $n_0$ is Manning's roughness coefficient, and $r(x, t)$
is a rainfall source term (currently a fixed spatial ramp, active for the whole
simulation window).

**Numerical method:** finite-volume conservative upwind in space, explicit forward
Euler with operator splitting between the flux update and the source term. The time
step is recomputed every iteration from the CFL condition using the current maximum
wave speed ($\Delta t = \text{CFL} \cdot \Delta x / c_{max}$), since $c$ now depends
nonlinearly on $h$ and a fixed $\Delta t$ can no longer guarantee stability.
Left boundary is held near zero depth (no inflow), and depth is clamped
non-negative after every step.

**Units:** meters and minutes throughout.

**Output:** running the script simulates to `T_final`, saves a before/after plot to
`data/linear_advection.png`, and saves the full depth-vs-time table (recorded every
`record_interval` minutes, default 1) to `data/linear_advection_timeseries.csv`. Use
`src/tools/animate_depth.py` to animate that table and watch depth evolve over time.

## Development History

The flood model is being built incrementally, with each stage adding one new piece
of physics on top of the last, verified before moving on. All stages so far live in
the same file (`linear_advection.py`), rewritten in place rather than split into
separate scripts — so the file's current behavior reflects only the *latest* stage.
Earlier stages are preserved as dated documents under `README/`.

| Stage | Commits | What changed |
|---|---|---|
| **0 — Basic linear advection** | `50c0d1c` | Initial upwind solver for the plain linear advection equation, constant wave speed, no source term. |
| **1 — Source term** | `a1ae414`, `d5ab4f4`, `0b8b7e6` | Added a source term $g(x,t)$ to the PDE, fixed a `dx` computation bug, and added an analytical verification test (method of characteristics) confirming the numerical solution against a closed-form solution. Documented in [`README/README_stage1.md`](README/README_stage1.md). |
| **2 — Nonlinear kinematic wave** | `536bf04` | Replaced the constant advection speed with the nonlinear Manning's-equation closure $c(h)$, turning the model into the kinematic wave equation and introducing genuinely nonlinear behavior (wave steepening, no more closed-form solution). Parameters (`S0`, `n0`) adjusted to physically realistic overland-flow values. |
| **2.1 — Adaptive time stepping** | `b15e0e9` | Added CFL-based adaptive $\Delta t$, recomputed each step from the current max wave speed, to prevent numerical oscillations now that $c$ can grow with $h$. General code cleanup. |
| **2.2 — Plot output** | `5ba28e8` | Switched from an interactive `plt.show()` to saving the comparison plot under `graphs/`. |
| **3 — 1D Saint-Venant (full dynamic wave)** | *(this session)* | New file `src/floods/saint_venant_1d.py`. Upgraded from kinematic wave to full dynamic Saint-Venant equations: added momentum equation tracking unit discharge $q = h \cdot \text{vel}$, pressure-gradient term $\partial(gh^2/2)/\partial x$, inertia, and Manning's friction slope $S_f$. Lax-Friedrichs scheme, adaptive CFL time stepping, operator-split source terms. |


## Repository Layout

```
src/floods/linear_advection.py   # current flood model solver (kinematic wave)
src/floods/saint_venant_1d.py    # 1D Saint-Venant (full dynamic wave) solver
src/tools/animate_depth.py       # animates a saved depth-vs-time table
tests/test_linear_advection.py   # mass conservation + analytical verification
data/                             # simulation output: plot + time series CSV
```

## How to Run

```
python src/floods/linear_advection.py
python src/tools/animate_depth.py                    # animate the most recent run
python src/tools/animate_depth.py path/to/other.csv  # or a specific table
```

Requires `numpy` and `matplotlib`. The solver produces `data/linear_advection.png`
and `data/linear_advection_timeseries.csv`.

## Next Steps

- Add spatial variation to the rainfall source in the active test case (the closure
  already supports it).
- Move from a single evolving script toward a structure that supports multiple
  models as the repository grows beyond the flood model.
