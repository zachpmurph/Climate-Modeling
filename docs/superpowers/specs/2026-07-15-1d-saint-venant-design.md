# 1D Saint-Venant (Full Dynamic Wave) — Design Spec

**Date:** 2026-07-15
**Branch:** feature-claude
**File to create:** `src/floods/saint_venant_1d.py`

---

## Overview

Replace the kinematic wave simplification with the full 1D Saint-Venant (shallow water) equations, adding a momentum equation that captures inertia and pressure-gradient effects alongside the existing continuity equation. This is a new file rather than an in-place rewrite, preserving the working kinematic wave model.

---

## Governing Equations

Conservation form with two state variables, depth $h$ [m] and unit discharge $q = h \cdot \text{vel}$ [m²/min]:

$$\frac{\partial}{\partial t}\begin{bmatrix}h\\q\end{bmatrix} + \frac{\partial}{\partial x}\begin{bmatrix}q\\\frac{q^2}{h}+\frac{gh^2}{2}\end{bmatrix} = \begin{bmatrix}r(x,t)\\gh(S_0-S_f)\end{bmatrix}$$

Manning's friction slope:

$$S_f = \frac{n_0^2\, \text{vel}\,|\text{vel}|}{h^{4/3}}$$

**Units:** meters and minutes throughout. Gravity converted: $g = 9.81\ \text{m/s}^2 = 35316\ \text{m/min}^2$.

---

## File & Architecture

- **New file:** `src/floods/saint_venant_1d.py`
- Mirrors `linear_advection.py` structure: module-level parameters, plain module-level functions (no classes), a `run_model()` returning a dict, a `__main__` block saving outputs to `data/`.
- Module-level `r`, `S0`, `n0` are monkeypatchable for tests (same pattern as existing model).
- No `src/floods/__init__.py` needed — namespace package, import as `from floods import saint_venant_1d`.

**Parameters (same as existing model):**
- `L = 10.0` — domain length [m]
- `T_final = 300.0` — simulation duration [min]
- `S0 = 0.05` — bed slope
- `n0 = 0.05` — Manning's roughness
- `g = 35316.0` — gravity [m/min²]

---

## Numerical Method

**Scheme:** Lax-Friedrichs finite volume, explicit Euler time integration, operator splitting.

Each time step:

1. **Lax-Friedrichs flux update** (interior cells only):
$$U_j^{*} = \frac{U_{j+1}^n + U_{j-1}^n}{2} - \frac{\Delta t}{2\Delta x}\left(F(U_{j+1}^n) - F(U_{j-1}^n)\right)$$
where $U = [h, q]^T$ and $F(U) = [q,\ q^2/h + gh^2/2]^T$.

2. **Source terms** (operator split, applied after flux update):
$$h \mathrel{+}= \Delta t \cdot r(x, t)$$
$$q \mathrel{+}= \Delta t \cdot g h (S_0 - S_f)$$

3. **Safeguards:**
   - Depth clamped: `h = maximum(h, 1e-10)`
   - Discharge zeroed in dry cells: `q[h <= floor] = 0`

**CFL time step** (recomputed every iteration):
$$\Delta t = \text{CFL} \cdot \frac{\Delta x}{\max(|\text{vel}| + c)}, \quad c = \sqrt{gh}, \quad \text{CFL} = 0.5$$

**Boundary conditions:**
- Left (upstream): `h[0] = 1e-10`, `q[0] = 0` — no inflow, held each step.
- Right (downstream): zero-gradient extrapolation — `h[-1] = h[-2]`, `q[-1] = q[-2]` — free outflow.

**Grid:** same as existing model — `Nx = int(L * 10)` cells, cell-centred, `dx = L / Nx`.

---

## `run_model()` Interface

```python
def run_model(L, T_final, record_interval=1.0) -> dict:
```

Return dict keys:
- `x` — cell centres [Nx]
- `times` — recorded snapshot times [Nt]
- `h_history`, `q_history` — depth and discharge snapshots [Nt × Nx]
- `h_initial`, `h_final`, `q_initial`, `q_final` — first/last states [Nx]
- `mass_source` — cumulative rainfall added to interior cells (1..Nx-1) [m²]
- `mass_outflow` — cumulative discharge through right boundary [m²]

Snapshot recording follows the same exact-landing logic as the existing model (cap `dt` to hit each recording mark rather than interpolating).

---

## Outputs

Written by the `__main__` block:
- `data/saint_venant_1d.png` — before/after depth comparison plot
- `data/saint_venant_1d_timeseries.csv` — CSV with one row per recorded time, one column per cell (same format as `linear_advection_timeseries.csv`, compatible with `animate_depth.py`)

---

## Initial Conditions

Same Gaussian bump as the existing model (provides non-trivial dynamics from the start):
- `h = 0.01 * exp(-((x - 3.0)² / 0.2))`
- `q = 0` everywhere (fluid at rest initially)

---

## Tests

New file: `tests/test_saint_venant_1d.py`

| Test | What it checks |
|---|---|
| `test_mass_conservation` | $\Delta(\text{stored}\ h) \approx \text{mass\_source} - \text{mass\_outflow}$ over interior cells |
| `test_still_water_at_rest` | Flat $h$, $q=0$, no rainfall, no slope — nothing should move (lake at rest) |

The still-water test is a necessary sanity check for any shallow-water scheme: a scheme that doesn't preserve still water will generate spurious velocities even with no forcing.

---

## Out of Scope (Future Stages)

- HLL/Roe Riemann solver upgrade
- Spatially varying bed (non-flat topography)
- Specified inflow hydrograph at left boundary
- 2D extension
