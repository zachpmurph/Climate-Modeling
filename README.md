# Climate-Modeling

A repository for climate-related numerical models. Currently the active work is a 1D
flood model built up incrementally from a simple linear advection solver to a
full dynamic-wave solver running on real river geometry. A solver-agnostic dispatch
harness lets any solver run on any ingested river profile through a single CLI.

---

## Physics: what the solvers solve

### Kinematic wave (`linear_advection.py`, `river_kinematic_wave.py`)

Solves the 1D kinematic wave equation for water depth $h(x, t)$:

$$\frac{\partial h}{\partial t} + \frac{\partial q(h)}{\partial x} = r(x, t)$$

The flux $q$ and wave speed $c$ are related to depth through Manning's equation:

$$q(h) = \frac{1}{n} h^{5/3} \sqrt{S_0}, \qquad c(h) = \frac{dq}{dh} = \frac{5}{3n} h^{2/3} \sqrt{S_0}$$

where $S_0$ is bed slope, $n$ is Manning's roughness coefficient, and $r(x, t)$ is
a rainfall source. The kinematic approximation assumes the friction slope equals the
bed slope, which is valid for slowly-varying flows and gentle slopes but neglects
inertia and pressure-gradient acceleration.

**Numerical scheme:** conservative finite-volume upwind in space, explicit forward
Euler in time with operator splitting (flux update then source addition). Time step
is recomputed every iteration from the CFL condition $\Delta t = \text{CFL} \cdot
\Delta x / c_\text{max}$. Depth is clamped non-negative after every step.

### Saint-Venant (full dynamic wave) (`saint_venant_1d.py`)

Solves the 1D Saint-Venant equations for depth $h$ and unit discharge $q$ together:

$$\frac{\partial h}{\partial t} + \frac{\partial q}{\partial x} = r$$

$$\frac{\partial q}{\partial t} + \frac{\partial}{\partial x}\!\left(\frac{q^2}{h} + \frac{g h^2}{2}\right) = g h (S_0 - S_f)$$

where $S_f = n^2 q^2 / h^{10/3}$ is the Manning friction slope and $g = 9.8 \times
60^2\ \text{m/min}^2$. This adds the momentum equation, capturing pressure-gradient
forces and flow inertia that the kinematic approximation omits — important near rapid
transients, steep wetting fronts, and backwater effects.

**Numerical scheme:** Rusanov (local Lax-Friedrichs) face fluxes over a conservative
finite-volume stencil, ghost-cell boundaries, explicit forward Euler time stepping,
operator-split Manning friction (semi-implicit in a single-step sense), adaptive CFL
time step.

**Units:** meters and minutes throughout (Manning's $n$ is converted from the
conventional s/m$^{1/3}$ units before use).

---

## Development history

Each stage is implemented in the same file as its predecessor, rewritten in place.
The rationale for each transition is given alongside the change.

| Stage | Commits | What changed | Why |
|---|---|---|---|
| **0 — Linear advection** | `50c0d1c` | Upwind finite-volume solver for $\partial_t h + a \partial_x h = 0$ with constant wave speed $a$. No source term. | Starting point: verify the grid, time loop, and upwind stencil work correctly on a problem with a known solution before adding nonlinearity. |
| **1 — Rainfall source term** | `a1ae414` `d5ab4f4` `0b8b7e6` | Added $r(x, t)$ source term; fixed a `dx` computation bug; added analytical verification test via method of characteristics. | Rainfall is the primary driver of overland flow. Verification against a closed-form solution confirmed the source term was coded correctly before moving to nonlinear physics. |
| **2 — Nonlinear kinematic wave** | `536bf04` | Replaced constant wave speed with Manning's-equation closure $c(h)$. Parameters adjusted to physically realistic overland-flow values ($S_0 = 0.05$, $n = 0.05$). | Linear advection does not capture wave steepening or depth-dependent propagation speed. The Manning's closure is the standard kinematic-wave model for overland flow; no closed-form solution exists once $c$ depends on $h$. |
| **2.1 — Adaptive time stepping** | `b15e0e9` | CFL-based adaptive $\Delta t$, recomputed each step from the current $c_\text{max}$. General code cleanup; made upwind updates cleaner. | With nonlinear $c(h)$, a fixed $\Delta t$ chosen at $t = 0$ can violate the CFL condition as depth grows, producing numerical oscillations. Adaptive stepping guarantees stability throughout. |
| **2.2 — Plot output to file** | `5ba28e8` | Switched from `plt.show()` to saving figures under `graphs/`. | `plt.show()` blocks execution in headless or remote environments. Saving to file lets the script run non-interactively. |
| **2.3 — Function refactor** | `26e0568` | Wrapped the time loop in `run_model()`. | Bare module-level code is untestable. A function interface lets pytest call the solver with different inputs without running the whole script. |
| **2.4 — Pytest tests** | `b96bf1a` | Added mass conservation and steady-state analytical tests. | Tests anchor the solver against known invariants (mass balance, Manning's steady state), making it safe to refactor internals. |
| **2.5 — CSV output + animation** | `2d6c684` | `save_time_series_csv()` writes a depth-vs-time table; `animate_depth.py` reads it back and renders a frame-by-frame animation. | The before/after snapshot only shows start and end states. A recorded time series lets you watch the wave propagate and detect instabilities that would otherwise be invisible. |
| **3a — Saint-Venant (Lax-Friedrichs)** | `a43fd30` `7280489` `127a183` `8dcc3bd` `5157561` `71bb3a1` `227d70c` | New file `saint_venant_1d.py`. Two-field solver $(h, q)$ with Lax-Friedrichs face fluxes, mass conservation test, physical boundary flux accounting, CSV output. | The kinematic wave approximation neglects the momentum equation, hiding pressure-gradient and inertial effects. Lax-Friedrichs is the simplest conservative scheme for systems of conservation laws and was used to get the two-field structure correct before worrying about accuracy. |
| **3b — Rusanov rewrite** | `7dd8863` | Replaced Lax-Friedrichs face fluxes with Rusanov (local Lax-Friedrichs) fluxes using ghost-cell boundaries. Added `left_inflow` parameter for prescribed upstream discharge. | Lax-Friedrichs uses a single global wave-speed estimate for all faces, adding excessive numerical diffusion that smears sharp fronts and pollutes the left boundary. Rusanov uses the local maximum wave speed at each face, giving lower diffusion while remaining simple and unconditionally entropy-satisfying. Ghost-cell boundaries also fix the left-BC diffusion artifact and enable a prescribed-discharge upstream condition needed for real river runs. |
| **3c — Real-river kinematic wave** | `01d808b` | New file `river_kinematic_wave.py` with per-cell slope and Manning's $n$ from a `RiverProfile` dataclass. Data pipeline (`collect_river_data.py`, `src/rivers/ingest/`) ingests USGS discharge, DEM-derived slopes, and roughness estimates into a local SQLite database; `export_profile.py` writes solver-ready CSV/JSON profiles. | Uniform-slope overland-flow models cannot represent real river channels whose geometry varies along the reach. Per-cell spatial variation is essential for using observed topography. |
| **4a — Restructure** | `93c682d` | Moved all files into `src/general/` (solvers, viz) and `src/rivers/` (ingest, simulations). | As the repo grew beyond a single solver, `src/floods/` and `src/tools/` no longer described their contents. The new layout separates reusable numerical machinery (`general/`) from the river-application layer (`rivers/`). |
| **4b — Solver-agnostic harness** | `69b519d` | `contract.py` defines `Domain`, `Scenario`, `SimulationResult`, `Solver` protocol, and `UnsupportedScenario`. `profile.py` houses `RiverProfile` loaders. Each solver exposes a `SOLVER` singleton; back-compat `run_model()` wrappers preserved. `registry.py` maps names to solvers; `run_simulation.py` is the unified CLI. | Adding a new solver previously required a new runner script and bespoke output handling. The contract layer means any solver can be swapped in by name, scenario knobs are validated up-front, and output is always a canonical `SimulationResult` with a mass-balance error in the JSON summary. |

---

## How to use the model

### Quick demos (no data needed)

Run the overland-flow kinematic wave solver and save a before/after plot:

```bash
python src/general/solvers/linear_advection.py
# → data/linear_advection.png
# → data/linear_advection_timeseries.csv
```

Animate the depth field evolving over time:

```bash
python src/general/viz/animate_depth.py                    # reads data/linear_advection_timeseries.csv
python src/general/viz/animate_depth.py path/to/other.csv  # or any recorded time series
```

Run the full dynamic-wave Saint-Venant solver:

```bash
python src/general/solvers/saint_venant_1d.py
# → data/saint_venant_1d.png
# → data/saint_venant_1d_timeseries.csv
```

### Unified CLI — run any solver on a river profile

`run_simulation.py` dispatches any registered solver on a CSV or JSON river profile,
writes an animate_depth-compatible time series CSV, and prints a JSON summary
including mass-balance error.

```bash
python src/rivers/simulations/run_simulation.py PROFILE --solver SOLVER --t-final T [options]
```

| Flag | Default | Description |
|---|---|---|
| `PROFILE` | *(required)* | Path to CSV or JSON river profile |
| `--solver` | `river_kinematic_wave` | One of: `river_kinematic_wave`, `saint_venant`, `kinematic_wave` |
| `--t-final` | *(required)* | Simulation duration, minutes |
| `--record-interval` | `1.0` | Snapshot interval, minutes |
| `--left-inflow` | `0.0` | Constant upstream inflow flux, m²/min |
| `--rainfall-rate` | `0.0` | Uniform rainfall rate, m/min |
| `--cfl` | `0.5` | CFL target (0 < CFL ≤ 1) |
| `--output-dir` | `data/real_world_rivers/runs/` | Output directory |
| `--run-name` | `simulation` | Filename prefix for outputs |

**Example — kinematic wave on example profile:**
```bash
python src/rivers/simulations/run_simulation.py \
    real_world_rivers/tools/example_river_profile.csv \
    --solver river_kinematic_wave \
    --t-final 30 \
    --left-inflow 0.0006 \
    --run-name hanford_kw
```

**Example — Saint-Venant on the same profile:**
```bash
python src/rivers/simulations/run_simulation.py \
    real_world_rivers/tools/example_river_profile.csv \
    --solver saint_venant \
    --t-final 10 \
    --left-inflow 0.0006 \
    --run-name hanford_sv
```

Each solver declares which `Scenario` knobs it supports. Passing an unsupported
knob (e.g. `--left-inflow` with `--solver kinematic_wave`) raises `UnsupportedScenario`
immediately rather than silently ignoring it.

### Ingesting real river data

The data pipeline collects DEM-derived slopes, Manning's roughness, and USGS discharge
for a reach, stores them in a local SQLite database, and exports a solver-ready profile:

```bash
# Initialise the database
python src/rivers/ingest/collect_river_data.py --db data/real_world_rivers/river_inputs.sqlite init

# Import a reach centreline (CSV, JSON, or GeoJSON LineString)
python src/rivers/ingest/collect_river_data.py create-reach \
    --river "Columbia" --reach "Hanford" \
    --markers real_world_rivers/columbia/hanford_markers.csv

# Fetch DEM elevations and derive slopes
python src/rivers/ingest/collect_river_data.py fetch-elevation --reach-id 1

# Fetch USGS continuous discharge
python src/rivers/ingest/collect_river_data.py fetch-flow \
    --reach-id 1 --site 12472800 --start 2024-01-01T00:00:00Z --end 2024-01-31T00:00:00Z

# Export a solver-ready profile
python src/rivers/ingest/collect_river_data.py export-profile \
    --reach-id 1 --output data/real_world_rivers/columbia_hanford_profile.csv
```

Then run any solver on the exported profile:

```bash
python src/rivers/simulations/run_simulation.py \
    data/real_world_rivers/columbia_hanford_profile.csv \
    --solver river_kinematic_wave --t-final 120 --left-inflow 0.015
```

For programmatic use, `profile_to_domain_scenario()` in
`src/rivers/simulations/ingest_to_simulate.py` converts a profile path directly into
a `(Domain, Scenario)` pair ready for `registry.dispatch()`.

### Running tests

```bash
python -m pytest tests/                                           # full suite (35 tests)
python -m pytest tests/test_linear_advection.py -v               # kinematic wave only
python -m pytest tests/test_saint_venant_1d.py -v                # Saint-Venant only
python -m pytest tests/test_run_simulation.py -v                 # harness + dispatch tests
python -m pytest tests/test_linear_advection.py::test_mass_conservation  # single test
```

---

## Repository layout

```
src/general/solvers/contract.py                # Domain, Scenario, SimulationResult, Solver protocol
src/general/solvers/profile.py                 # RiverProfile dataclass and CSV/JSON loaders
src/general/solvers/linear_advection.py        # kinematic wave overland-flow solver
src/general/solvers/saint_venant_1d.py         # 1D Saint-Venant (full dynamic wave) solver
src/general/solvers/river_kinematic_wave.py    # kinematic wave solver for real river profiles
src/general/viz/animate_depth.py               # animates a saved depth-vs-time table
src/rivers/simulations/registry.py             # name → Solver mapping
src/rivers/simulations/run_simulation.py       # unified CLI dispatcher
src/rivers/simulations/ingest_to_simulate.py   # profile_path → (Domain, Scenario) helper
src/rivers/simulations/run_river_kinematic_wave.py  # legacy runner (pre-harness)
src/rivers/ingest/collect_river_data.py        # CLI for the real-river data pipeline
src/rivers/ingest/                             # USGS, DEM, roughness importers + SQLite helpers
tests/test_linear_advection.py                 # mass conservation + analytical steady-state
tests/test_saint_venant_1d.py                  # conservation, equilibrium, boundary, dry-state
tests/test_river_kinematic_wave.py             # profile I/O and mass balance
tests/test_river_data_tools.py                 # data-pipeline unit tests
tests/test_run_simulation.py                   # dispatch, UnsupportedScenario, result shapes
data/                                          # simulation output: plots and time series CSVs
data/real_world_rivers/                        # SQL schema, local database, run outputs
real_world_rivers/                             # example profiles and Columbia River inputs
```

---

## Solver capabilities at a glance

| Solver name | File | Left inflow | Per-cell geometry | Two-field (h+q) | Grid |
|---|---|---|---|---|---|
| `kinematic_wave` | `linear_advection.py` | No (pinned near 0) | No (uniform S₀, n) | No | Internal L×10 |
| `river_kinematic_wave` | `river_kinematic_wave.py` | Yes | Yes | No | Profile stations |
| `saint_venant` | `saint_venant_1d.py` | Yes (callable or const) | No (uniform S₀, n) | Yes | Internal L×10 |

`kinematic_wave` and `saint_venant` reconstruct their own uniform grid at 10
cells/metre from the domain length $L$. Their `SimulationResult.domain` reflects the
internal grid so mass-balance arithmetic is always self-consistent, even though it
differs from the input `Domain` cell count.

---

## Next steps

- Expose `Nx` (or `dx`) as a parameter in `linear_advection.run_model` and
  `saint_venant_1d.run_model` so those solvers can honour the input `Domain` grid
  directly, removing the L×10 internal grid.
- Add per-cell slope and Manning's $n$ to the Saint-Venant solver so it can also
  run on real river profiles with spatially varying geometry.
- Add spatial variation to the rainfall source in the overland-flow test case (the
  $r(x, t)$ closure already supports it).
- Extend the data pipeline to additional river systems beyond the Columbia River
  Hanford reach.
