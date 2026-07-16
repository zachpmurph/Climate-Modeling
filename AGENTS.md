# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this repo is

A growing collection of climate-related numerical models. Currently the only model
under development is a 1D flood model (kinematic wave overland flow), living in
`src/general/solvers/linear_advection.py`. See [README.md](README.md) for the physics, the
governing equation, and a stage-by-stage development history.

Solvers now live under `src/general/solvers/` and share a common contract defined in
`src/general/solvers/contract.py`. A unified harness in `src/rivers/simulations/`
dispatches to any registered solver by name.

## Commands

There is no build system or linter configured. Dependencies are `numpy`, `matplotlib`,
and `pytest` (no requirements file exists — install them directly if missing).

Run the solver:

```
python src/general/solvers/linear_advection.py
```

This runs the simulation to completion and writes a plot to `data/linear_advection.png`
plus a per-minute depth table to `data/linear_advection_timeseries.csv` (no window is
shown — the script saves rather than displays). To watch that table animate:

```
python src/general/viz/animate_depth.py
```

Run the tests:

```
python -m pytest tests/
python -m pytest tests/test_linear_advection.py::test_mass_conservation  # single test
```

Unified CLI (solver-agnostic harness):

```
python src/rivers/simulations/run_simulation.py real_world_rivers/tools/example_river_profile.csv \
    --solver river_kinematic_wave --t-final 10 --left-inflow 0.0006
```

`pytest.ini` sets `pythonpath = src`, so tests import model code as
`from general.solvers import linear_advection` without an installed package or `__init__.py`
(there is no `src/general/solvers/__init__.py` — `general.solvers` is a namespace package). Model
functions (`run_model`, `r`, `c`, `q`, ...) are plain module-level functions in
`linear_advection.py`, not wrapped in a class, so tests monkeypatch them directly
(e.g. `monkeypatch.setattr(la, "r", other_func)`) to substitute things like a
different rainfall source rather than passing them as parameters.

## Architecture: staged-development-in-place

The most important thing to know before editing `linear_advection.py`: **each
development stage rewrites the same file rather than adding a new one.** There is no
`stage2.py` alongside `stage1.py` — stage 2 (nonlinear kinematic wave) replaced stage
1's code (linear advection) entirely in the same file. The stage-by-stage history lives
only in the [README.md](README.md) history table and commit messages, not as runnable
per-stage code — don't assume an old `README/README_stageN.md` still exists or still
matches current behavior; check the top-level README.md history table first.

When you finish a new stage, update that history table rather than leaving the record
only in commit messages.

## Solver contract

Each solver in `src/general/solvers/` exposes a module-level `SOLVER` singleton that implements the `Solver` protocol from `src/general/solvers/contract.py`. The protocol requires:
- `name: str` — registry key
- `supports: frozenset[str]` — which `Scenario` knobs this solver honours
- `run(domain: Domain, scenario: Scenario) -> SimulationResult`

The back-compat `run_model(...)` function is preserved in each solver file so existing tests and `__main__` demos continue to work unchanged.

`src/rivers/simulations/registry.py` maps solver names to instances. `src/rivers/simulations/run_simulation.py` is the unified CLI entry point.

## Model conventions

- **Units:** meters and minutes throughout (not SI seconds) — keep this consistent
  when adding parameters or new source terms.
- **Naming:** `u`/`h` = state variable (flow depth), `q` = flux, `c` = wave speed
  (`dq/dh`), `S0` = bed slope, `n0` = Manning's roughness coefficient, `r(x, t)` =
  source term (rainfall), `CFL` = Courant number target.
- **Numerical scheme:** conservative finite-volume upwind in space, explicit Euler in
  time with operator splitting (flux update, then source addition). Time step is
  recomputed every iteration from the CFL condition against the current max wave
  speed — do not hardcode `dt`, since `c(h)` is nonlinear and a fixed step can go
  unstable as `h` grows.
- Depth is clamped non-negative after every update, and the left boundary cell is
  pinned near zero (no-inflow condition) each step — preserve both when restructuring
  the update loop.
- `run_model(L, T_final, record_interval=1.0)` returns a dict (`x`, `u_initial`,
  `u_final`, `times`, `u_history`, `mass_source`, `mass_outflow`) rather than a bare
  array. `mass_source`/`mass_outflow` are cumulative totals tracked over the interior
  control volume (cells `1..Nx-1`) during the time loop — cell 0 is a
  boundary-condition cell, not physical storage, so it's excluded from both. This is
  what `tests/test_linear_advection.py::test_mass_conservation` checks against.
- `times`/`u_history` are snapshots taken every `record_interval` minutes (always
  including `t=0` and `t=T_final`), not every adaptive `dt` — the loop caps `dt` so it
  lands exactly on each recording mark rather than overshooting it, so snapshots are
  exact, not interpolated. `save_time_series_csv()` writes this table to disk (one row
  per recorded time, one column per cell) for `src/general/viz/animate_depth.py` to read back.
