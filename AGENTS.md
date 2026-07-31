# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this repo is

A growing collection of climate-related numerical models. Current flood solvers
include a 1-D kinematic wave, 1-D Saint-Venant, and 2-D Saint-Venant model under
`src/general/solvers/`. See [README.md](README.md) for the physics, governing
equations, and stage-by-stage development history.

Solvers now live under `src/general/solvers/` and share a common contract defined in
`src/general/solvers/contract.py`. A unified harness in `src/rivers/simulations/`
dispatches to any registered solver by name.

## Commands

There is no build system or linter configured. Dependencies are pinned in
`requirements.txt`.

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
python -m pytest tests/test_linear_advection.py::test_upstream_inflow_mass_balance  # single test
python src/general/verification/verify_saint_venant_2d.py  # quantitative 2-D matrix
```

Unified CLI (solver-agnostic harness):

```
python src/rivers/simulations/run_simulation.py real_world_rivers/tools/example_river_profile.csv \
    --solver kinematic_wave --t-final 10 --left-inflow 0.0006
```

For a rectangular 2-D extrusion of the same profile:

```
python src/rivers/simulations/run_simulation.py real_world_rivers/tools/example_river_profile.csv \
    --solver saint_venant_2d --width 100 --cross-cells 20 --t-final 10
```

`linear_advection.py` can also be run standalone on a profile (or with no argument
for a built-in demo):

```
python src/general/solvers/linear_advection.py real_world_rivers/tools/example_river_profile.csv
```

`pytest.ini` sets `pythonpath = src`, so tests import model code as
`from general.solvers import linear_advection` without an installed package or `__init__.py`
(there is no `src/general/solvers/__init__.py` — `general.solvers` is a namespace package). Model
functions (`run_model`, `q`, `c`, `make_profile`, `load_profile`, ...) are plain
module-level functions in `linear_advection.py`, not wrapped in a class. Tests build a
`RiverProfile` with `make_profile(...)` and call `run_model(profile, t_final_min,
left_inflow_flux, ...)` directly rather than passing solver internals as parameters.

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
- `run(domain: Domain | Domain2D, scenario: Scenario) -> SimulationResult`

Each solver file also keeps a plain `run_model(...)` function used by its tests and
`__main__`. For `linear_advection.py` this is the profile-based
`run_model(profile, t_final_min, left_inflow_flux, ...)`.

`src/rivers/simulations/registry.py` maps solver names to instances. `src/rivers/simulations/run_simulation.py` is the unified CLI entry point.

Observed-event validation under `src/rivers/validation/` must use
`saint_venant_2d` exclusively. The canonical validation runner rejects missing
`validation_2d` settings and must never dispatch or silently fall back to a 1-D
solver. The 1-D solvers remain available for numerical verification, reference,
and non-validation simulation workflows.

All registered solvers run on the supplied grid and must honor its spatial cell
widths, slopes, and `manning_n`. `scenario_from_profile(...)` transfers
the optional profile initial depth, rainfall rates, and labels into `Scenario`.
`Scenario.rainfall` is evaluated as `rainfall(x_m, t_min)` during time stepping;
do not reduce it to a single spatial or temporal sample.

`saint_venant_2d` requires `Domain2D`; the CLI constructs one by extruding the
longitudinal profile across `--width`. Programmatic cases may supply full
`slope_x`, `slope_y`, and `manning_n` fields. `Scenario.rainfall` is broadcast
across y, while `Scenario.rainfall_2d(x_m, y_m, t_min)` can provide lateral
variation. The harness saves full 2-D fields to `<run>_fields.npz` and a
cross-channel-mean CSV for compatibility.

`Domain2D.bed_elevation_m` is the authoritative topography used by the
well-balanced hydrostatic reconstruction. Profile extrusion derives it from
longitudinal slope. Do not replace the hydrostatic face corrections with a
cell-centred `g*h*slope` source: that would break the non-flat lake-at-rest
invariant. The 2-D draining limiter must scale a donor's flux consistently on
the shared face; do not reintroduce a mass-adding depth floor.

Any numerical-core change to `saint_venant_2d.py` must pass both
`tests/test_saint_venant_2d_verification.py` and the standalone verification
matrix. If an intentional method change alters quantitative reference values,
document the reason, rerun the full grid study, and update
`docs/validation/saint_venant_2d_results.json`.

## Model conventions

- **Units:** meters and minutes throughout (not SI seconds) — keep this consistent
  when adding parameters or new source terms.
- **Naming:** `depth`/`h` = state variable (flow depth), `q` = flux (per unit width),
  `c` = wave speed (`dq/dh`), `slope`/`S0` = bed slope, `manning_n`/`n0` = Manning's
  roughness, `cfl` = Courant number target. `q(depth, slope, manning_n)` and
  `c(depth, slope, manning_n)` take per-cell slope and roughness (not module globals).
- **Numerical scheme:** conservative finite-volume upwind in space, explicit Euler in
  time with operator splitting (flux update, then source addition). Time step is
  recomputed every iteration from the CFL condition against the current max wave
  speed — do not hardcode `dt`, since `c(h)` is nonlinear and a fixed step can go
  unstable as `h` grows.
- Depth is clamped non-negative after every update. The **left boundary is a flux
  boundary**: the left interface carries the constant `left_inflow_flux` (0 for no
  inflow), and interior interfaces carry the upwind cell's Manning flux — preserve
  this when restructuring the update loop.
- `run_model(profile, t_final_min, left_inflow_flux, record_interval_min=1.0, ...)`
  returns a dict with `station_m`, `dx_m`, `slope`, `manning_n`, `times`,
  `depth_history`, `depth_initial`, `depth_final`, and cumulative `mass_inflow`,
  `mass_source`, `mass_outflow`. Mass balance is
  `Δstorage == mass_inflow + mass_source − mass_outflow`, checked by
  `tests/test_linear_advection.py::test_upstream_inflow_mass_balance` and
  `::test_rainfall_source_mass_balance`.
- `times`/`u_history` are snapshots taken every `record_interval` minutes (always
  including `t=0` and `t=T_final`), not every adaptive `dt` — the loop caps `dt` so it
  lands exactly on each recording mark rather than overshooting it, so snapshots are
  exact, not interpolated. `save_time_series_csv()` writes this table to disk (one row
  per recorded time, one column per cell) for `src/general/viz/animate_depth.py` to read back.
