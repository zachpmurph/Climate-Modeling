# Climate-Modeling

A repository for climate-related numerical models. The active flood models are
a 1-D kinematic wave, a 1-D dynamic-wave solver, and a verified 2-D
Saint-Venant solver. A solver-agnostic dispatch harness lets any solver run on
an ingested river profile through a single CLI.

---

## Physics: what the solvers solve

### Kinematic wave (`linear_advection.py`)

With reviewed rectangular-section width $B(x)$, solves the conservative
kinematic wave equation for cross-sectional area $A=Bh$:

$$\frac{\partial A}{\partial t} + \frac{\partial Q(A)}{\partial x} = B r(x, t)$$

The whole-channel flow uses Manning's equation with hydraulic radius
$R_h=A/(B+2h)$:

$$Q(h) = \frac{1}{n} A R_h^{2/3} \sqrt{S_0}.$$

Without hydraulic geometry, the backward-compatible unit-width form is
$q=h^{5/3}\sqrt{S_0}/n$.

where $S_0$ is bed slope, $n$ is Manning's roughness coefficient, and $r(x, t)$ is
a rainfall source. The kinematic approximation assumes the friction slope equals the
bed slope, which is valid for slowly-varying flows and gentle slopes but neglects
inertia and pressure-gradient acceleration.

**Numerical scheme:** conservative finite-volume upwind in space, explicit forward
Euler in time with operator splitting (flux update then source addition). Time step
is recomputed every iteration from the CFL condition $\Delta t = \text{CFL} \cdot
\Delta x / c_\text{max}$. Depth is clamped non-negative after every step.

### Saint-Venant (full dynamic wave) (`saint_venant_1d.py`)

Solves the 1D Saint-Venant equations for rectangular cross-sectional area
$A=Bh$ and whole-channel discharge $Q$:

$$\frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = B r$$

$$\frac{\partial Q}{\partial t} + \frac{\partial}{\partial x}\!\left(\frac{Q^2}{A} + \frac{g B h^2}{2}\right) = g A (S_0 - S_f) + S_B$$

where $S_f=n^2u|u|/R_h^{4/3}$ is the Manning friction slope, $S_B$ is the
non-prismatic width source, and $g = 9.8 \times 60^2\ \text{m/min}^2$. This
adds the momentum equation, capturing pressure-gradient
forces and flow inertia that the kinematic approximation omits — important near rapid
transients, steep wetting fronts, and backwater effects.

**Numerical scheme:** Rusanov (local Lax-Friedrichs) face fluxes over a conservative
finite-volume stencil, hydrostatic reconstruction for well-balanced non-flat beds,
a conservative draining limiter for wet/dry cells, ghost-cell boundaries, explicit
forward Euler time stepping, operator-split Manning friction (semi-implicit in a
single-step sense), and adaptive CFL time steps. The solver runs on the supplied
profile grid and applies bed elevation, Manning roughness, and rainfall independently
in every cell. A common-width face reconstruction plus a discrete width source
preserves a non-flat lake at rest even when channel width changes by cell.

**Units:** meters and minutes throughout (Manning's $n$ is converted from the
conventional s/m$^{1/3}$ units before use).

### 2-D Saint-Venant (`saint_venant_2d.py`)

Solves the Cartesian shallow-water system for
$U = [h, hu, hv]^T$:

$$
\frac{\partial U}{\partial t}
+ \frac{\partial F(U)}{\partial x}
+ \frac{\partial G(U)}{\partial y}
= S(U,z_b,R).
$$

The implementation uses first-order Rusanov face fluxes, hydrostatic
reconstruction for well-balanced non-flat topography, a conservative draining
limiter for wet/dry positivity, adaptive two-dimensional CFL stepping, and
semi-implicit Manning friction. It supports reflecting walls,
inflow/open-outflow x boundaries, and periodic verification domains.

The Tier 3 verification matrix includes exact axial and diagonal wave
convergence studies, a variable-depth manufactured pressure wave, non-flat lake
at rest, quantitative 1-D reduction, wet radial dam-break symmetry, strict mass
conservation, and a dry-bed dam break. See
[the numerical method and verification report](docs/saint_venant_2d_numerics.md)
and the machine-readable
[`docs/validation/saint_venant_2d_results.json`](docs/validation/saint_venant_2d_results.json).

---

## Development history

Each stage is implemented in the same file as its predecessor, rewritten in place.
The rationale for each transition is given alongside the change.

| Stage | Commits | What changed | Why |
|---|---|---|---|
| **0 — Linear advection** | `50c0d1c` | Upwind finite-volume solver for $\partial_t h + a \partial_x h = 0$ with constant wave speed $a$. No source term. | Starting point: verify the grid, time loop, and upwind stencil work correctly on a problem with a known solution before adding nonlinearity. |
| **1 — Rainfall source term** | `a1ae414` `d5ab4f4` `0b8b7e6` | Added $r(x, t)$ source term; fixed a `dx` computation bug; added analytical verification test via method of characteristics. | Rainfall is the primary driver of overland flow. Verification against a closed-form solution confirmed the source term was coded correctly before moving to nonlinear physics. |
| **2 — Nonlinear kinematic wave** | `536bf04` | Replaced constant wave speed with Manning's-equation closure $c(h)$. Parameters use $S_0 = 0.05$ and conventional SI $n = 0.05$ (stored as $0.05/60$ in the current minutes-based model). | Linear advection does not capture wave steepening or depth-dependent propagation speed. The Manning's closure is the standard kinematic-wave model for overland flow; no closed-form solution exists once $c$ depends on $h$. |
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
| **4c — Kinematic wave consolidated** | `a282b4f` `e1ec579` | Folded the real-river kinematic wave capability (per-cell slope and Manning's $n$, upstream inflow, rainfall) into `linear_advection.py` and removed the duplicate `river_kinematic_wave.py` and its pre-harness runner. `linear_advection.py` now runs standalone on a profile (or a built-in demo) and is the `kinematic_wave` solver in the harness. `--solver river_kinematic_wave` is replaced by `--solver kinematic_wave`. | The overland-flow file and the real-river file had diverged into near-duplicate kinematic wave solvers. Consolidating to one implementation removes the redundancy and the need for a separate file to run a real-profile simulation. |
| **4d — Model-neutral flood reporting** | `6116048` | Added a reporting consumer for saved time-series CSV and summary JSON artifacts. It produces a self-contained interactive HTML report and versioned outcomes JSON with peak depth, timing, reach-threshold exceedance, and mass-balance diagnostics. | Reporting should evolve independently from numerical model development. Consuming saved artifacts prevents visualization code from coupling to solver internals and makes the interpretation boundary explicit. |
| **4e — Profile-grid dynamic wave and forcing** | Current branch | Saint-Venant now runs on the supplied nonuniform profile grid with per-cell bed slope and Manning roughness. Both solvers accept spatially and temporally varying rainfall callables. Profile initial depth, rainfall, and labels are transferred into `Scenario`. | Reconstructing a uniform grid and using module-level coefficients discarded real-reach variation. Sampling rainfall only once also prevented event functions from changing through time. |
| **4f — Integrated 2-D shallow water** | Current branch | Added `Domain2D` and a registered `saint_venant_2d` solver. The unified runner builds a terrain-backed channel and floodplain from reviewed width/bankfull geometry, applies spatial terrain, roughness, and rainfall, saves complete fields to NPZ, and produces plan-view area-based flood reports. | The standalone 2-D solver could not consume ingested profiles or participate in shared simulation and reporting workflows; a flat cross-channel extrusion could not represent overbank inundation. |
| **4g — Tier 3 numerical verification** | Current branch | Added explicit bed elevation, hydrostatic reconstruction, a conservative draining limiter, finite-state diagnostics, periodic verification boundaries, analytic convergence, non-flat equilibrium, 1-D reduction, radial symmetry, strict mass, and wet/dry gates. Pinned dependencies and clean-checkout CI preserve evidence. | Stability and visual plausibility do not establish PDE accuracy. The solver now has quantitative, reproducible evidence for first-order convergence, well-balancedness, positivity, multidimensional symmetry, and machine-precision conservation within its documented scope. |
| **4h — Geographic flood screening** | Current branch | Added an interactive topographic map that animates canonical saved depth time series along a reviewed river centerline. The runner can record portable marker and geometry paths in its summary so the map command auto-discovers them. | Existing reports quantify outcomes but do not place a 1-D result in geographic context. The map makes scenario review easier while explicitly retaining the distinction between estimated cross-section width and a terrain-resolving 2-D inundation boundary. |
| **4i — Observed baseline and well-balanced 1-D dynamics** | Current branch | Added an approved-USGS two-gauge validation case and rebuilt 1-D Saint-Venant bed coupling with hydrostatic reconstruction and a conservative draining limiter. | Exact flat-bed and synthetic tests hid spurious currents over real topography. The observed case now quantifies field error, while the 1-D solver preserves a non-flat lake at rest without mass-adding depth floors. |
| **4j — Hydraulic cross-sections** | Current branch | Added reviewed per-cell channel width and bankfull depth to the 1-D domain. Both 1-D solvers now conserve whole-channel volume and discharge when geometry is supplied; Saint-Venant includes rectangular hydraulic radius and a well-balanced non-prismatic width source. | Unit-width flow cannot reproduce real storage, wetted perimeter, friction, rainfall volume, or gauge discharge without ad hoc conversions. |
| **4k — Event forcing and consistent startup** | Current branch | Added linearly interpolated inflow/rainfall CSV forcing, preserved callable hydrographs in every solver, aligned time steps with forcing knots, and initialized dynamic-wave discharge from the boundary flow. Whole-channel 2-D flow is distributed only across initially wet upstream cells. | Collapsing a hydrograph to its first value and starting a flowing boundary from zero momentum create physically false transients and erase the event being simulated. |
| **4l — Provenance-safe solver grids** | Current branch | Added optional longitudinal resampling that linearly interpolates reviewed fields onto a derived numerical grid while preserving reach length and labeling the source/solver cell counts in every summary. | A five-station measurement profile is too coarse for routing, but silently treating interpolated cells as observations overstates the evidence. |
| **4m — Conservative dry kinematic states** | Current branch | Removed the artificial minimum depth from profile loading and kinematic updates, added a conservative draining limiter, retained time-varying hydrographs, and exposed any roundoff floor correction in mass accounting. | A hidden depth floor creates water in every dry cell and can make a mass balance appear better than it is. |
| **4n — Downstream hydraulic boundaries** | Current branch | Added free-outflow, reflecting-wall, and prescribed-stage downstream boundaries to 1-D Saint-Venant, including stage-driven backflow and signed boundary mass accounting. | A zero-gradient outlet cannot represent a dam, closed gate, lake level, tide, or downstream backwater control. |
| **4o — Limited second-order reconstruction** | Current branch | Added optional minmod-limited second-order reconstruction of water surface, bed, and velocity/momentum to both Saint-Venant solvers (plus width and unit discharge in 1-D). It retains hydrostatic well-balancing and conservative draining limiters while reducing smooth-wave diffusion. | First-order piecewise-constant Rusanov fluxes smear hydrographs and wetting fronts, especially on long reaches. |
| **4p — Manning fixture units** | Current branch | Converted the shipped example profile, reviewed-roughness example, standalone kinematic demo, and ingestion fixtures from conventional SI seconds-based Manning values to the repository's minutes-based values (`n_model = n_SI / 60`). | Supplying `0.035` directly to a minutes-based solver makes roughness 60× too large and mislabels a seconds-scale Manning flux as per-minute flow. |

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
| `--solver` | `saint_venant` | One of: `kinematic_wave`, `saint_venant`, `saint_venant_2d` |
| `--t-final` | *(required)* | Simulation duration, minutes |
| `--record-interval` | `1.0` | Snapshot interval, minutes |
| `--left-inflow` | `0.0` | Constant upstream flow: m³/min with hydraulic geometry, legacy m²/min otherwise |
| `--inflow-series` | — | CSV with `t_min,left_inflow`; mutually exclusive with nonzero `--left-inflow` |
| `--rainfall-rate` | `0.0` | Uniform rainfall rate, m/min |
| `--rainfall-series` | — | CSV with `t_min,rainfall_rate_m_per_min`; added to profile and constant rainfall |
| `--downstream-boundary` | `outflow` | 1-D Saint-Venant: `outflow`, `wall`, or `stage` |
| `--downstream-stage` | — | Fixed water-surface elevation for a `stage` boundary |
| `--spatial-order` | `1` | Saint-Venant reconstruction order: robust first-order or less-diffusive second-order |
| `--cfl` | `0.5` | CFL target (0 < CFL ≤ 1) |
| `--longitudinal-cells` | — | Derived solver-cell count; linearly interpolates reviewed fields without creating observations |
| `--width` | — | Total channel-plus-floodplain domain width in metres; required for `saint_venant_2d` |
| `--cross-cells` | `10` | Number of cells across a 2-D domain |
| `--hydraulic-geometry` | — | Reviewed `station_m,width_m,bankfull_depth_m` CSV; optional for physical 1-D sections and required for 2-D |
| `--floodplain-slope` | `0.02` | Lateral rise/run outside the reviewed bankfull channel |
| `--output-dir` | `data/real_world_rivers/runs/` | Output directory |
| `--run-name` | `simulation` | Filename prefix for outputs |

**Example — kinematic wave on example profile:**
```bash
python src/rivers/simulations/run_simulation.py \
    real_world_rivers/tools/example_river_profile.csv \
    --solver kinematic_wave \
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

**Example — 2-D Saint-Venant on a terrain-backed channel and floodplain:**
```bash
python src/rivers/simulations/run_simulation.py \
    real_world_rivers/tools/example_river_profile.csv \
    --solver saint_venant_2d \
    --width 100 \
    --cross-cells 20 \
    --hydraulic-geometry real_world_rivers/tools/example_geometry.csv \
    --t-final 10 \
    --run-name hanford_sv2
```

The 2-D runner uses the reviewed channel width and bankfull depth to construct a
centred parabolic channel, then raises the floodplain beyond each bank at
`--floodplain-slope`. Profile initial depth is applied as a level water surface,
so higher bank and floodplain cells start dry. It writes a full
`<run>_fields.npz`, a summary JSON, and a cross-channel-mean CSV for
compatibility with 1-D tools. This synthetic cross-section is safer than a flat
extrusion but is not a DEM or surveyed cross-section; build measured terrain
cases programmatically by passing a `Domain2D` directly.

Each solver declares which `Scenario` knobs it supports. Passing a knob a solver
doesn't support raises `UnsupportedScenario` immediately rather than silently
ignoring it.

Optional profile fields are applied automatically: `initial_depth_m` becomes the
scenario initial condition, `rainfall_rate_m_per_min` becomes a spatial rainfall
function, and `label` values are retained in `Scenario.labels`. The CLI
`--rainfall-rate` is added to any rainfall already stored in the profile.

For programmatic scenarios, `Scenario.rainfall` may be any callable with the
signature `rainfall(x_m, t_min) -> rates_m_per_min`; all solvers evaluate it
during time stepping. A 2-D case may instead set `Scenario.rainfall_2d` with
`rainfall_2d(x_m, y_m, t_min) -> rates_m_per_min`.

For a physically scaled 1-D run, pass `--hydraulic-geometry`. The solver then
interprets initial and boundary discharge as whole-channel m³/min, uses the
interpolated width in storage and friction, and reports mass in m³. Omitting
geometry retains the historical unit-width mode for verification and backward
compatibility. `bankfull_depth_m` is retained as a reviewed reference; the 1-D
section itself is currently rectangular.

Time-varying forcing CSVs must start at `t_min=0`, contain at least two strictly
increasing times, and have finite non-negative values. Values are linearly
interpolated between rows and held constant outside the supplied range. Solver
time steps land exactly on every forcing row, preventing a sharp forcing change
from being skipped. The summary records portable paths to both forcing files.

Use `--longitudinal-cells` when a reviewed profile is too sparse for numerical
routing. The source CSV/JSON remains unchanged. Slope, roughness, optional
depth, rainfall, and hydraulic geometry are interpolated onto a derived grid;
labels survive only at exact reviewed stations. Every summary records both
counts, the interpolation method, and `creates_observations: false`.

### Reporting saved flood outcomes

Generate a self-contained interactive report from any harness time-series CSV:

```bash
python src/rivers/reporting/generate_flood_report.py \
    data/real_world_rivers/runs/example_timeseries.csv \
    --depth-threshold 0.5
```

For reviewed reach geometry, replace the uniform threshold with
`--geometry PATH`, where the CSV contains `station_m` and
`bankfull_depth_m`. The reporter writes both HTML and a versioned
`.outcomes.json` artifact. For a 2-D run it auto-discovers the full field,
displays a plan-view depth map, and reports threshold-exceedance area. See
`docs/reporting_contract.md` for the stable model-to-report boundary and
interpretation limits.

### Geographic flood animation

To pair a run with reviewed map inputs, record its ordered centerline and channel
geometry in the run summary:

```bash
python src/rivers/simulations/run_simulation.py \
    real_world_rivers/tools/example_river_profile.csv \
    --solver kinematic_wave --t-final 30 --run-name example_map \
    --map-markers real_world_rivers/tools/example_markers.csv \
    --map-geometry real_world_rivers/tools/example_geometry.csv

python src/rivers/visualization/animate_flood_map.py \
    data/real_world_rivers/runs/example_map_timeseries.csv
```

The second command writes a neighboring `_flood_map.html` with playback,
scrubbing, depth tooltips, and an OpenTopoMap basemap. Leaflet and map tiles
require an internet connection when the HTML is opened. The polygons are
screening estimates: 1-D depths are spread across an estimated cross-section,
and a 2-D run's canonical CSV contains its cross-channel mean. Use the standard
2-D flood report and full NPZ field for area-based 2-D outcomes.

### Ingesting real river data

The data pipeline turns authoritative provider data (DEM-derived slopes, Manning's
roughness, USGS discharge, Open-Meteo rainfall) into a validated, provenance-rich,
model-ready profile. Full reference: [docs/real_world_ingestion.md](docs/real_world_ingestion.md).

**Config-driven (recommended).** A curated JSON reach definition under
`real_world_rivers/curated/` declares the river/reach identifiers, marker centreline,
provider windows, roughness/geometry sources, and export options. `run_ingestion`
runs the whole pipeline — collect → validate → export — for one reach or a whole
directory, exiting non-zero if any reach fails or a declared export is missing (so it
is safe to gate a pipeline on):

```bash
# One curated reach
python -m rivers.ingest.run_ingestion real_world_rivers/curated/columbia_hanford.json

# Every *.json definition in a directory
python -m rivers.ingest.run_ingestion --all real_world_rivers/curated
```

Each export writes the profile plus a `.metadata.json` provenance sidecar in which
every value is classified `observed` / `derived` / `estimated` / `fallback`, alongside
validation findings — errors block export; warnings are retained. SQLite databases and
generated profiles are reproducible and git-ignored; curated definitions and the
reviewed source CSVs are committed.

**Manual (component-level).** The same stages can be driven one at a time with
`collect_river_data.py`, e.g. when building or inspecting a reach interactively:

```bash
# Initialise the database
python src/rivers/ingest/collect_river_data.py --db data/real_world_rivers/river_inputs.sqlite init

# Import a reach centreline (CSV, JSON, or GeoJSON LineString)
python src/rivers/ingest/collect_river_data.py create-reach \
    --river "Columbia" --reach "Hanford" \
    --markers real_world_rivers/columbia_hanford_markers.csv

# Fetch DEM elevations and derive slopes; fetch USGS continuous discharge
python src/rivers/ingest/collect_river_data.py fetch-elevation --reach-id 1
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
    --solver kinematic_wave --t-final 120 --left-inflow 0.015
```

For programmatic use, `profile_to_domain_scenario()` in
`src/rivers/simulations/ingest_to_simulate.py` converts a profile path directly into
a `(Domain, Scenario)` pair ready for `registry.dispatch()`.

### Running tests

```bash
python -m pytest tests/                                           # full suite
python -m pytest tests/test_linear_advection.py -v               # kinematic wave only
python -m pytest tests/test_saint_venant_1d.py -v                # Saint-Venant only
python -m pytest tests/test_saint_venant_2d_verification.py -v   # 2-D Tier 3 gates
python -m pytest tests/test_run_simulation.py -v                 # harness + dispatch tests
python -m pytest tests/test_linear_advection.py::test_upstream_inflow_mass_balance  # single test
```

Run the standalone verification matrix and emit machine-readable evidence:

```bash
python src/general/verification/verify_saint_venant_2d.py \
    --output docs/validation/saint_venant_2d_results.json
```

Dependencies are pinned in `requirements.txt`. The GitHub Actions verification
workflow runs the complete suite and matrix from a clean checkout.

---

## Repository layout

```
src/general/solvers/contract.py                # Domain, Scenario, SimulationResult, Solver protocol
src/general/solvers/profile.py                 # RiverProfile dataclass and CSV/JSON loaders
src/general/solvers/linear_advection.py        # kinematic wave solver (per-cell profile; standalone + harness)
src/general/solvers/saint_venant_1d.py         # 1D Saint-Venant (full dynamic wave) solver
src/general/solvers/saint_venant_2d.py         # verified 2D Saint-Venant solver
src/general/verification/verify_saint_venant_2d.py # quantitative benchmark matrix
src/general/viz/animate_depth.py               # animates a saved depth-vs-time table
src/rivers/simulations/registry.py             # name → Solver mapping
src/rivers/simulations/run_simulation.py       # unified CLI dispatcher
src/rivers/simulations/ingest_to_simulate.py   # profile_path → (Domain, Scenario) helper
src/rivers/reporting/generate_flood_report.py  # saved artifacts → HTML + outcomes JSON
src/rivers/visualization/animate_flood_map.py  # saved time series → geographic HTML animation
src/rivers/ingest/run_ingestion.py             # config-driven ingestion CLI (one reach or --all)
src/rivers/ingest/orchestrator.py              # runs a curated reach definition end to end
src/rivers/ingest/collect_river_data.py        # low-level per-step data pipeline CLI
src/rivers/ingest/validation.py                # profile validation gate (error/warning/info)
src/rivers/ingest/                             # USGS, DEM, roughness/geometry importers + SQLite helpers
real_world_rivers/curated/                     # reviewed curated reach definitions (JSON)
docs/real_world_ingestion.md                   # ingestion guide: sources, config, validation, provenance
tests/test_linear_advection.py                 # profile I/O, mass balance, analytical equilibrium
tests/test_saint_venant_1d.py                  # conservation, equilibrium, boundary, dry-state
tests/test_saint_venant_2d.py                  # stability, sources, boundaries, edge cases
tests/test_saint_venant_2d_verification.py     # quantitative Tier 3 gates
tests/test_river_data_tools.py                 # data-pipeline unit tests
tests/test_ingestion_orchestrator.py           # config-driven ingest: single + batch, exit status
tests/test_ingestion_validation.py             # validation severities and export gating
tests/test_ingestion_export.py                 # atomic export + provenance sidecar
tests/test_ingestion_reliability.py            # dedup, retries, credential redaction
tests/test_run_simulation.py                   # dispatch, UnsupportedScenario, result shapes
tests/test_flood_reporting.py                  # report validation, outcomes, HTML, CLI
data/                                          # simulation output: plots and time series CSVs
data/real_world_rivers/                        # SQL schema, local database, run outputs
real_world_rivers/                             # example profiles and Columbia River inputs
```

---

## Solver capabilities at a glance

| Solver name | File | Left inflow | Spatial geometry | Momentum | Grid |
|---|---|---|---|---|---|
| `kinematic_wave` | `linear_advection.py` | Yes | Yes | No | Profile stations |
| `saint_venant` | `saint_venant_1d.py` | Yes (callable or const) | Yes | Yes | Profile stations |
| `saint_venant_2d` | `saint_venant_2d.py` | Yes (callable or const) | Yes, per 2-D cell | x and y | Profile × cross-channel cells |

All solvers run on the profile's longitudinal stations, honouring spatially
varying slope, Manning's $n$, initial depth, and rainfall. Reviewed hydraulic
geometry adds physical 1-D channel width or constructs the synthetic 2-D
channel/floodplain terrain.

---

## Next steps

- Extend rectangular 1-D sections to surveyed compound/trapezoidal geometry.
- Save Saint-Venant discharge histories through the unified CLI alongside depth.
- Extend the data pipeline to additional river systems beyond the Columbia River
  Hanford reach.
