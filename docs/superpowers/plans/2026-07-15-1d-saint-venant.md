# 1D Saint-Venant (Full Dynamic Wave) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full 1D Saint-Venant equations in `src/floods/saint_venant_1d.py` using a Lax-Friedrichs scheme, tracking depth `h` and unit discharge `q`.

**Architecture:** New file mirroring `linear_advection.py` structure — module-level parameters, plain functions (no classes), adaptive CFL time stepping, Lax-Friedrichs flux update with operator-split source terms. Two state variables: `h` (depth, m) and `q = h * vel` (unit discharge, m²/min). Tests use monkeypatching on module-level `r`, `S0`, `n0`, same as the existing model.

**Tech Stack:** NumPy, Matplotlib, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/floods/saint_venant_1d.py` | Create | Solver: parameters, `r()`, `run_model()`, `save_time_series_csv()`, `__main__` |
| `tests/test_saint_venant_1d.py` | Create | Tests: still water at rest, mass conservation |
| `README.md` | Modify | Add new model to Development History table and Repository Layout |

---

## Background: Units and Key Formulas

All units are **meters and minutes**. Gravity must be converted:
$$g = 9.81\ \text{m/s}^2 = 9.81 \times 3600\ \text{m/min}^2 = 35316\ \text{m/min}^2$$

The two governing equations in conservation form:

$$\frac{\partial h}{\partial t} + \frac{\partial q}{\partial x} = r(x,t)$$

$$\frac{\partial q}{\partial t} + \frac{\partial}{\partial x}\!\left(\frac{q^2}{h} + \frac{gh^2}{2}\right) = gh(S_0 - S_f)$$

Manning's friction slope:
$$S_f = \frac{n_0^2\, \text{vel}\,|\text{vel}|}{h^{4/3}}, \qquad \text{vel} = q/h$$

Lax-Friedrichs update for interior cell $j$ (written for both $h$ and $q$, i.e., both components of $U = [h,q]^T$):

$$U_j^{*} = \frac{U_{j+1}^n + U_{j-1}^n}{2} - \frac{\Delta t}{2\Delta x}\bigl(F(U_{j+1}^n) - F(U_{j-1}^n)\bigr)$$

where the flux vector is $F(U) = \bigl[q,\ q^2/h + gh^2/2\bigr]^T$.

CFL wave speed uses both characteristics: $\Delta t = \text{CFL} \cdot \Delta x / \max(|\text{vel}| + c)$ where $c = \sqrt{gh}$.

---

### Task 1: Create skeleton file

**Files:**
- Create: `src/floods/saint_venant_1d.py`

- [ ] **Step 1: Write the skeleton**

Create `src/floods/saint_venant_1d.py` with this exact content:

```python
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
    raise NotImplementedError


def save_time_series_csv(result, path):
    raise NotImplementedError


if __name__ == "__main__":
    pass
```

- [ ] **Step 2: Verify the import works**

```bash
python -c "from floods import saint_venant_1d; print('ok')"
```

Expected output: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/floods/saint_venant_1d.py
git commit -m "feat: add saint_venant_1d skeleton"
```

---

### Task 2: Still water at rest — write test, implement run_model, verify

**Files:**
- Create: `tests/test_saint_venant_1d.py`
- Modify: `src/floods/saint_venant_1d.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_saint_venant_1d.py`:

```python
import numpy as np
import pytest

from floods import saint_venant_1d as sv


def test_still_water_at_rest(monkeypatch):
    # Flat h=0.5, q=0, no rain, no slope — cells far from the left BC
    # should remain exactly undisturbed.
    #
    # The left BC pins h[0] ~ 0, which creates a step gradient that
    # propagates rightward. Each LxF step spreads this disturbance at
    # most 1 cell (±1 stencil), so after n steps only cells 1..n are
    # affected. With h=0.5 the CFL dt ≈ 3.76e-4 min; T_final=1e-3 min
    # executes ~3 steps, touching at most cells 1..3. Cells 5..Nx-2 are
    # exactly unaffected and should equal the initial condition to machine
    # precision.
    monkeypatch.setattr(sv, "S0", 0.0)

    def no_rain(x, t):
        return np.zeros_like(x)

    monkeypatch.setattr(sv, "r", no_rain)

    Nx = int(sv.L * 10)
    h0 = 0.5 * np.ones(Nx)
    q0 = np.zeros(Nx)

    result = sv.run_model(sv.L, 1e-3, h_init=h0, q_init=q0)

    h_final = result["h_final"]
    q_final = result["q_final"]

    assert np.allclose(h_final[5:-1], 0.5, rtol=1e-10), \
        f"Still water disturbed: max deviation = {np.max(np.abs(h_final[5:-1] - 0.5))}"
    assert np.allclose(q_final[5:-1], 0.0, atol=1e-15), \
        f"Discharge non-zero: max = {np.max(np.abs(q_final[5:-1]))}"
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
python -m pytest tests/test_saint_venant_1d.py::test_still_water_at_rest -v
```

Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement run_model**

Replace the `run_model` stub in `src/floods/saint_venant_1d.py` with:

```python
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

        # Mass bookkeeping (outflow through right face, source over interior)
        source = r(x, t_current)
        mass_outflow += q[-1] * dt
        mass_source += np.sum(source[1:]) * dx * dt

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

        # Operator-split source terms
        h_new += dt * source

        vel_new = np.where(h_new > h_floor, q_new / h_new, 0.0)
        Sf = (n0 ** 2 * vel_new * np.abs(vel_new)
              / np.maximum(h_new, h_floor) ** (4 / 3))
        q_new += dt * g * h_new * (S0 - Sf)

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
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
python -m pytest tests/test_saint_venant_1d.py::test_still_water_at_rest -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/floods/saint_venant_1d.py tests/test_saint_venant_1d.py
git commit -m "feat: implement run_model with Lax-Friedrichs; add still water test"
```

---

### Task 3: Mass conservation test

**Files:**
- Modify: `tests/test_saint_venant_1d.py`

- [ ] **Step 1: Add the mass conservation test**

Append to `tests/test_saint_venant_1d.py`:

```python
def test_mass_conservation():
    # Run long enough that outflow is non-trivial (~40 min same as kinematic wave test).
    # Tolerance is 1% rather than 0.1%: Lax-Friedrichs numerical diffusion at the
    # left BC creates a small mass leakage not captured by outflow tracking alone.
    result = sv.run_model(sv.L, 40.0)
    x = result["x"]
    dx = x[1] - x[0]

    stored_initial = np.sum(result["h_initial"][1:]) * dx
    stored_final = np.sum(result["h_final"][1:]) * dx
    delta_mass = stored_final - stored_initial

    expected_delta = result["mass_source"] - result["mass_outflow"]

    assert delta_mass == pytest.approx(expected_delta, rel=1e-2)
```

- [ ] **Step 2: Run both tests**

```bash
python -m pytest tests/test_saint_venant_1d.py -v
```

Expected: both PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_saint_venant_1d.py
git commit -m "test: add mass conservation test for saint_venant_1d"
```

---

### Task 4: CSV output and `__main__` block

**Files:**
- Modify: `src/floods/saint_venant_1d.py`

- [ ] **Step 1: Replace the `save_time_series_csv` stub and `__main__` block**

Replace the two stubs at the bottom of `src/floods/saint_venant_1d.py` with:

```python
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
```

- [ ] **Step 2: Run the model and verify outputs**

```bash
python src/floods/saint_venant_1d.py && ls data/saint_venant_1d*
```

Expected:
```
data/saint_venant_1d.png
data/saint_venant_1d_timeseries.csv
```

- [ ] **Step 3: Run the full test suite to confirm nothing broke**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/floods/saint_venant_1d.py
git commit -m "feat: add CSV output and __main__ block to saint_venant_1d"
```

---

### Task 5: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add row to the Development History table**

In `README.md`, add this row after the `| **2.2 — Plot output** | ...` line:

```markdown
| **3 — 1D Saint-Venant (full dynamic wave)** | *(this session)* | New file `src/floods/saint_venant_1d.py`. Upgraded from kinematic wave to full dynamic Saint-Venant equations: added momentum equation tracking unit discharge $q = h \cdot \text{vel}$, pressure-gradient term $\partial(gh^2/2)/\partial x$, inertia, and Manning's friction slope $S_f$. Lax-Friedrichs scheme, adaptive CFL time stepping, operator-split source terms. |
```

- [ ] **Step 2: Add entry to Repository Layout**

In the `## Repository Layout` section, add this line:

```
src/floods/saint_venant_1d.py    # 1D Saint-Venant (full dynamic wave) solver
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README for 1D Saint-Venant model"
```
