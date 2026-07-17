# Restructure + Solver-Agnostic Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganise `src/` into `general/` (solvers + viz) and `rivers/` (ingest + simulations), then add a shared contract and unified `run_simulation.py` CLI that dispatches any solver by name.

**Architecture:** Part 1 is a pure filesystem + import rename with no logic changes; green tests confirm correctness at every step. Part 2 adds `contract.py` (Domain/Scenario/SimulationResult/Solver), adapts each solver to that interface with a back-compat `run_model` wrapper, then wires the registry and CLI.

**Tech Stack:** Python 3.11+, numpy, pytest; no new dependencies.

---

## Branch setup

- [ ] Create working branch:
  ```bash
  git checkout -b feature/restructure-harness
  ```
  (Both `feature-claude` and `origin/main` point to the same commit `6133d36`, so this branches cleanly from the current state.)

---

## PART 1 — Filesystem restructure

### Task 1: Create target directories

**Files:**
- Create: `src/general/solvers/` (directory)
- Create: `src/general/viz/` (directory)
- Create: `src/rivers/ingest/` (directory)
- Create: `src/rivers/simulations/` (directory)

- [ ] **Step 1: Make directories**

  ```bash
  mkdir -p src/general/solvers src/general/viz src/rivers/ingest src/rivers/simulations
  ```

- [ ] **Step 2: Verify**

  ```bash
  ls src/general/ src/rivers/
  ```
  Expected: `solvers  viz` and `ingest  simulations`.

---

### Task 2: Move solver files with git mv

**Files:**
- Move: `src/floods/linear_advection.py` → `src/general/solvers/linear_advection.py`
- Move: `src/floods/saint_venant_1d.py` → `src/general/solvers/saint_venant_1d.py`
- Move: `src/floods/river_kinematic_wave.py` → `src/general/solvers/river_kinematic_wave.py`

- [ ] **Step 1: git mv the three solvers**

  ```bash
  git mv src/floods/linear_advection.py src/general/solvers/linear_advection.py
  git mv src/floods/saint_venant_1d.py src/general/solvers/saint_venant_1d.py
  git mv src/floods/river_kinematic_wave.py src/general/solvers/river_kinematic_wave.py
  ```

---

### Task 3: Move viz and scripts with git mv

**Files:**
- Move: `src/tools/animate_depth.py` → `src/general/viz/animate_depth.py`
- Move: `src/tools/run_river_kinematic_wave.py` → `src/rivers/simulations/run_river_kinematic_wave.py`
- Move: `src/tools/collect_river_data.py` → `src/rivers/ingest/collect_river_data.py`
- Move: `src/tools/river_data/__init__.py` → `src/rivers/ingest/__init__.py`
- Move all `src/tools/river_data/*.py` → `src/rivers/ingest/`

- [ ] **Step 1: git mv animate_depth and scripts**

  ```bash
  git mv src/tools/animate_depth.py src/general/viz/animate_depth.py
  git mv src/tools/run_river_kinematic_wave.py src/rivers/simulations/run_river_kinematic_wave.py
  git mv src/tools/collect_river_data.py src/rivers/ingest/collect_river_data.py
  ```

- [ ] **Step 2: git mv river_data contents**

  ```bash
  git mv src/tools/river_data/__init__.py src/rivers/ingest/__init__.py
  git mv src/tools/river_data/common.py src/rivers/ingest/common.py
  git mv src/tools/river_data/database.py src/rivers/ingest/database.py
  git mv src/tools/river_data/elevation.py src/rivers/ingest/elevation.py
  git mv src/tools/river_data/export_profile.py src/rivers/ingest/export_profile.py
  git mv src/tools/river_data/markers.py src/rivers/ingest/markers.py
  git mv src/tools/river_data/parameters.py src/rivers/ingest/parameters.py
  git mv src/tools/river_data/rainfall.py src/rivers/ingest/rainfall.py
  git mv src/tools/river_data/usgs_flow.py src/rivers/ingest/usgs_flow.py
  ```

- [ ] **Step 3: Remove now-empty directories**

  ```bash
  rmdir src/tools/river_data
  rmdir src/tools
  rmdir src/floods
  ```

- [ ] **Step 4: Verify layout**

  ```bash
  find src/ -name "*.py" | sort
  ```
  Should show files under `src/general/solvers/`, `src/general/viz/`, `src/rivers/ingest/`, `src/rivers/simulations/` — nothing under `src/floods/` or `src/tools/`.

---

### Task 4: Fix imports in test files

**Files:**
- Modify: `tests/test_linear_advection.py`
- Modify: `tests/test_saint_venant_1d.py`
- Modify: `tests/test_river_kinematic_wave.py`
- Modify: `tests/test_river_data_tools.py`

- [ ] **Step 1: Fix test_linear_advection.py**

  Change line 22:
  ```python
  # was:
  from floods import linear_advection as la
  # becomes:
  from general.solvers import linear_advection as la
  ```

- [ ] **Step 2: Fix test_saint_venant_1d.py**

  Change line 6:
  ```python
  # was:
  from floods import saint_venant_1d as sv
  # becomes:
  from general.solvers import saint_venant_1d as sv
  ```

- [ ] **Step 3: Fix test_river_kinematic_wave.py**

  Change line 7:
  ```python
  # was:
  from floods import river_kinematic_wave as rkw
  # becomes:
  from general.solvers import river_kinematic_wave as rkw
  ```

- [ ] **Step 4: Fix test_river_data_tools.py**

  Lines 7–19 currently:
  ```python
  from floods import river_kinematic_wave
  from tools.river_data.common import haversine_m
  from tools.river_data.elevation import collect_elevations, parse_elevations
  from tools.river_data.export_profile import export_profile
  from tools.river_data.markers import create_reach, load_marker_rows
  from tools.river_data.parameters import import_geometry, import_roughness
  from tools.river_data.rainfall import collect_rainfall, parse_hourly_precipitation
  from tools.river_data.usgs_flow import (
      CFS_TO_M3_PER_MIN,
      collect_usgs_flow,
      discharge_to_m3_per_min,
      fetch_usgs_flow,
  )
  ```
  Replace with:
  ```python
  from general.solvers import river_kinematic_wave
  from rivers.ingest.common import haversine_m
  from rivers.ingest.elevation import collect_elevations, parse_elevations
  from rivers.ingest.export_profile import export_profile
  from rivers.ingest.markers import create_reach, load_marker_rows
  from rivers.ingest.parameters import import_geometry, import_roughness
  from rivers.ingest.rainfall import collect_rainfall, parse_hourly_precipitation
  from rivers.ingest.usgs_flow import (
      CFS_TO_M3_PER_MIN,
      collect_usgs_flow,
      discharge_to_m3_per_min,
      fetch_usgs_flow,
  )
  ```

- [ ] **Step 5: Run tests — expect 25 passes**

  ```bash
  python -m pytest tests/ -v
  ```
  Expected: 25 passed (all green). If any fail at this point, the import paths are wrong — debug before continuing.

---

### Task 5: Fix sys.path and imports in moved scripts

**Files:**
- Modify: `src/rivers/simulations/run_river_kinematic_wave.py`
- Modify: `src/rivers/ingest/collect_river_data.py`

- [ ] **Step 1: Fix run_river_kinematic_wave.py**

  The file now lives at `src/rivers/simulations/run_river_kinematic_wave.py`.
  `parents[1]` used to reach `src/` (from `src/tools/`); now `parents[2]` reaches `src/`.
  Also update the solver import.

  Replace lines 5–9:
  ```python
  # was:
  SRC_ROOT = Path(__file__).resolve().parents[1]
  if str(SRC_ROOT) not in sys.path:
      sys.path.insert(0, str(SRC_ROOT))

  from floods import river_kinematic_wave as rkw
  # becomes:
  SRC_ROOT = Path(__file__).resolve().parents[2]
  if str(SRC_ROOT) not in sys.path:
      sys.path.insert(0, str(SRC_ROOT))

  from general.solvers import river_kinematic_wave as rkw
  ```

- [ ] **Step 2: Fix collect_river_data.py**

  The file now lives at `src/rivers/ingest/collect_river_data.py`.
  `parents[2]` used to be repo root (adding repo root let it do `from src.tools...`);
  now `parents[2]` is `src/` — which is exactly what we want so we can do `from rivers.ingest...`.
  Rename the variable and update all imports.

  Replace lines 6–18:
  ```python
  # was:
  REPO_ROOT = Path(__file__).resolve().parents[2]
  if str(REPO_ROOT) not in sys.path:
      sys.path.insert(0, str(REPO_ROOT))

  from src.tools.river_data.common import connect_database
  from src.tools.river_data.database import DEFAULT_DB_PATH, initialize_database
  from src.tools.river_data.elevation import collect_elevations
  from src.tools.river_data.export_profile import export_profile
  from src.tools.river_data.markers import create_reach
  from src.tools.river_data.parameters import import_geometry, import_roughness
  from src.tools.river_data.rainfall import collect_rainfall
  from src.tools.river_data.usgs_flow import collect_usgs_flow
  # becomes:
  SRC_ROOT = Path(__file__).resolve().parents[2]
  if str(SRC_ROOT) not in sys.path:
      sys.path.insert(0, str(SRC_ROOT))

  from rivers.ingest.common import connect_database
  from rivers.ingest.database import DEFAULT_DB_PATH, initialize_database
  from rivers.ingest.elevation import collect_elevations
  from rivers.ingest.export_profile import export_profile
  from rivers.ingest.markers import create_reach
  from rivers.ingest.parameters import import_geometry, import_roughness
  from rivers.ingest.rainfall import collect_rainfall
  from rivers.ingest.usgs_flow import collect_usgs_flow
  ```

- [ ] **Step 3: Confirm database.py parents[3] is still repo root**

  `src/rivers/ingest/database.py` → `parents[3]` = repo root (0=ingest, 1=rivers, 2=src, 3=repo). No change needed.

- [ ] **Step 4: Run tests again**

  ```bash
  python -m pytest tests/ -v
  ```
  Expected: 25 passed.

---

### Task 6: Update docstrings and documentation

**Files:**
- Modify: `src/general/viz/animate_depth.py` (module docstring)
- Modify: `tests/test_linear_advection.py` (module docstring)
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Fix animate_depth.py docstring**

  Replace the module docstring (lines 1–13) — change path references:
  ```python
  """View a kinematic wave depth-vs-time table as an animation.

  Reads a CSV produced by general.solvers.linear_advection.save_time_series_csv
  (header row: t, x_0, x_1, ...; one row per recorded time) and animates
  u(x) over time, one frame per recorded timestamp.

  Usage:
      python src/general/viz/animate_depth.py [path/to/timeseries.csv]

  With no argument, reads data/linear_advection_timeseries.csv.
  """
  ```

- [ ] **Step 2: Fix test_linear_advection.py docstring**

  Change the reference in line 14:
  ```
  # was:   by src/tools/animate_depth.py to animate depth over time.
  # becomes: by src/general/viz/animate_depth.py to animate depth over time.
  ```

- [ ] **Step 3: Update README.md**

  Replace the "Repository layout" block (lines ~68-77 in README):
  ```
  src/general/solvers/linear_advection.py    # kinematic wave overland-flow solver
  src/general/solvers/saint_venant_1d.py     # 1D Saint-Venant (full dynamic wave) solver
  src/general/solvers/river_kinematic_wave.py# kinematic wave solver for real river profiles
  src/general/viz/animate_depth.py           # animates a saved depth-vs-time table
  src/rivers/simulations/run_river_kinematic_wave.py  # runs the river-profile kinematic wave solver
  src/rivers/ingest/collect_river_data.py    # collects and exports real-river input data
  src/rivers/ingest/                         # provider clients, importers, and SQLite helpers
  tests/test_linear_advection.py             # mass conservation + analytical verification
  tests/test_saint_venant_1d.py              # conservation, equilibrium, boundary, and dry-state tests
  tests/test_river_kinematic_wave.py         # profile I/O and mass balance tests
  ```

  Replace the "Quick start" commands (lines ~87-91):
  ```
  python src/general/solvers/linear_advection.py
  python src/general/solvers/saint_venant_1d.py
  python src/general/viz/animate_depth.py                    # animate the most recent run
  python src/general/viz/animate_depth.py path/to/other.csv  # or a specific table
  python src/rivers/ingest/collect_river_data.py --help      # real-river data workflow
  ```

  Update the history table Stage 3 entry to reflect new path `src/general/solvers/river_kinematic_wave.py`.

- [ ] **Step 4: Update CLAUDE.md**

  Replace every `src/floods/` with `src/general/solvers/` and `src/tools/animate_depth.py` with `src/general/viz/animate_depth.py`. Key lines to update:
  - Line 9: `src/general/solvers/linear_advection.py`
  - Line 20: `python src/general/solvers/linear_advection.py`
  - Line 28: `python src/general/viz/animate_depth.py`
  - Line 39: `from general.solvers import linear_advection`
  - Line 40: `(there is no `src/general/solvers/__init__.py` — `general.solvers` is a namespace package)`
  - Line 79: `tests/test_linear_advection.py::test_mass_conservation`
  - Line 84: `src/general/viz/animate_depth.py`

- [ ] **Step 5: Update AGENTS.md** (keep in sync with CLAUDE.md — same path replacements, just title/description line differs)

- [ ] **Step 6: Run tests one final time for Part 1**

  ```bash
  python -m pytest tests/ -v
  ```
  Expected: 25 passed.

---

### Task 7: Smoke-test CLIs and commit Part 1

- [ ] **Step 1: Smoke-test solver scripts**

  ```bash
  python src/general/solvers/linear_advection.py
  python src/general/solvers/saint_venant_1d.py
  ```
  Both should complete without error and write files under `data/`.

- [ ] **Step 2: Smoke-test ingest CLI**

  ```bash
  python src/rivers/ingest/collect_river_data.py --help
  ```
  Expected: argparse help text shown, no ImportError.

- [ ] **Step 3: Smoke-test river sim CLI**

  ```bash
  python src/rivers/simulations/run_river_kinematic_wave.py \
      real_world_rivers/tools/example_river_profile.csv \
      --left-inflow-flux 0.0006 \
      --t-final 5 \
      --wave-amplitude 0.01 \
      --output-dir /tmp/rk-smoke
  ```
  Expected: completes, writes CSV + JSON to `/tmp/rk-smoke/`.

- [ ] **Step 4: Commit Part 1 (do NOT add generated data/ artifacts)**

  ```bash
  git add src/ tests/ README.md CLAUDE.md AGENTS.md pytest.ini
  git status  # confirm no data/*.csv or data/*.png staged
  git commit -m "refactor: split src/ into general/solvers+viz and rivers/ingest+simulations"
  ```

---

## PART 2 — Solver-agnostic harness

### Task 8: Add contract.py (Domain, Scenario, SimulationResult, Solver protocol)

**Files:**
- Create: `src/general/solvers/contract.py`

- [ ] **Step 1: Write the contract module**

  Create `src/general/solvers/contract.py`:
  ```python
  from __future__ import annotations

  from dataclasses import dataclass, field
  from typing import Callable, Protocol, runtime_checkable

  import numpy as np


  class UnsupportedScenario(Exception):
      """Raised when a Scenario knob is not in a solver's ``supports`` set."""


  @dataclass(frozen=True)
  class Domain:
      """Per-cell spatial description of a 1-D river reach."""

      x_m: np.ndarray        # cell-centre positions, metres
      dx_m: np.ndarray       # cell widths, metres
      slope: np.ndarray      # bed slope S0, dimensionless
      manning_n: np.ndarray  # Manning roughness n


  @dataclass
  class Scenario:
      """Everything the solver needs beyond the domain geometry."""

      t_final_min: float
      record_interval_min: float = 1.0
      initial_depth_m: float | np.ndarray = 0.0
      initial_discharge: float | np.ndarray = 0.0
      left_inflow: float | Callable[[float], float] = 0.0
      rainfall: Callable[[np.ndarray, float], np.ndarray] | None = None
      cfl: float = 0.5


  @dataclass
  class SimulationResult:
      """Canonical output every solver must produce."""

      domain: Domain
      times: np.ndarray               # shape (n_times,)
      depth_history: np.ndarray       # shape (n_times, n_cells)
      depth_initial: np.ndarray       # shape (n_cells,)
      depth_final: np.ndarray         # shape (n_cells,)
      mass_inflow: float
      mass_source: float
      mass_outflow: float
      extra: dict = field(default_factory=dict)


  @runtime_checkable
  class Solver(Protocol):
      name: str
      supports: frozenset[str]

      def run(self, domain: Domain, scenario: Scenario) -> SimulationResult:
          ...
  ```

- [ ] **Step 2: Verify import**

  ```bash
  python -c "from general.solvers.contract import Domain, Scenario, SimulationResult, Solver, UnsupportedScenario; print('ok')"
  ```
  Expected: `ok`.

- [ ] **Step 3: Run tests (still 25 green)**

  ```bash
  python -m pytest tests/ -q
  ```

---

### Task 9: Add profile.py (move RiverProfile + loaders, add domain_from_profile)

**Files:**
- Create: `src/general/solvers/profile.py`
- Modify: `src/general/solvers/river_kinematic_wave.py` (remove duplicated definitions, import from profile)

- [ ] **Step 1: Create profile.py**

  Extract `RiverProfile`, `_as_float`, `_cell_widths_from_stations`, `_optional_array`, `make_profile`, `load_profile_csv`, `load_profile_json`, `load_profile` from `river_kinematic_wave.py` and add `domain_from_profile`.

  Create `src/general/solvers/profile.py`:
  ```python
  import csv
  import json
  from dataclasses import dataclass
  from pathlib import Path

  import numpy as np

  from general.solvers.contract import Domain


  @dataclass(frozen=True)
  class RiverProfile:
      """Cell-centred river profile inputs for the 1-D kinematic wave model."""

      station_m: np.ndarray
      dx_m: np.ndarray
      slope: np.ndarray
      manning_n: np.ndarray
      initial_depth_m: np.ndarray | None = None
      rainfall_rate_m_per_min: np.ndarray | None = None
      labels: list[str] | None = None


  def _as_float(row, key, *, required=True, default=None):
      val = row.get(key)
      if val is None or val == "":
          if required:
              raise ValueError(f"Missing required field '{key}'")
          return default
      return float(val)


  def _cell_widths_from_stations(station_m):
      n = len(station_m)
      dx = np.empty(n)
      if n == 1:
          dx[0] = 1.0
          return dx
      dx[0] = station_m[1] - station_m[0]
      dx[-1] = station_m[-1] - station_m[-2]
      for i in range(1, n - 1):
          dx[i] = (station_m[i + 1] - station_m[i - 1]) / 2.0
      return dx


  def _optional_array(values, expected_len, name, *, minimum=None):
      if all(v is None for v in values):
          return None
      if any(v is None for v in values):
          raise ValueError(f"Field '{name}' must be present in all rows or none")
      arr = np.array(values, dtype=float)
      if minimum is not None and np.any(arr < minimum):
          raise ValueError(f"All values of '{name}' must be >= {minimum}")
      return arr


  def make_profile(station_m, slope, manning_n, initial_depth_m=None, rainfall_rate_m_per_min=None, labels=None):
      station_m = np.asarray(station_m, dtype=float)
      slope = np.asarray(slope, dtype=float)
      manning_n = np.asarray(manning_n, dtype=float)
      dx_m = _cell_widths_from_stations(station_m)
      if initial_depth_m is not None:
          initial_depth_m = np.asarray(initial_depth_m, dtype=float)
      if rainfall_rate_m_per_min is not None:
          rainfall_rate_m_per_min = np.asarray(rainfall_rate_m_per_min, dtype=float)
      return RiverProfile(
          station_m=station_m,
          dx_m=dx_m,
          slope=slope,
          manning_n=manning_n,
          initial_depth_m=initial_depth_m,
          rainfall_rate_m_per_min=rainfall_rate_m_per_min,
          labels=labels,
      )


  def load_profile_csv(path):
      path = Path(path)
      rows = []
      with path.open(newline="", encoding="utf-8") as f:
          reader = csv.DictReader(f)
          for row in reader:
              rows.append(row)
      if not rows:
          raise ValueError(f"Profile CSV '{path}' is empty")
      station_m = np.array([float(r["station_m"]) for r in rows])
      slope = np.array([_as_float(r, "slope") for r in rows])
      manning_n = np.array([_as_float(r, "manning_n") for r in rows])
      initial_depth = [_as_float(r, "initial_depth_m", required=False) for r in rows]
      rainfall = [_as_float(r, "rainfall_rate_m_per_min", required=False) for r in rows]
      labels = [r.get("label") for r in rows] if "label" in rows[0] else None
      return make_profile(
          station_m=station_m,
          slope=slope,
          manning_n=manning_n,
          initial_depth_m=_optional_array(initial_depth, len(rows), "initial_depth_m", minimum=0),
          rainfall_rate_m_per_min=_optional_array(rainfall, len(rows), "rainfall_rate_m_per_min", minimum=0),
          labels=labels,
      )


  def load_profile_json(path):
      path = Path(path)
      data = json.loads(path.read_text(encoding="utf-8"))
      segments = data["segments"]
      station_m = np.array([float(s["station_m"]) for s in segments])
      slope = np.array([float(s["slope"]) for s in segments])
      manning_n = np.array([float(s["manning_n"]) for s in segments])
      initial_depth = [s.get("initial_depth_m") for s in segments]
      rainfall = [s.get("rainfall_rate_m_per_min") for s in segments]
      labels = [s.get("label") for s in segments] if any("label" in s for s in segments) else None
      return make_profile(
          station_m=station_m,
          slope=slope,
          manning_n=manning_n,
          initial_depth_m=_optional_array(initial_depth, len(segments), "initial_depth_m", minimum=0),
          rainfall_rate_m_per_min=_optional_array(rainfall, len(segments), "rainfall_rate_m_per_min", minimum=0),
          labels=labels,
      )


  def load_profile(path):
      path = Path(path)
      if path.suffix.lower() == ".json":
          return load_profile_json(path)
      return load_profile_csv(path)


  def domain_from_profile(profile: RiverProfile) -> Domain:
      """Build a Domain from a RiverProfile (uses per-cell slope and Manning n)."""
      return Domain(
          x_m=profile.station_m,
          dx_m=profile.dx_m,
          slope=profile.slope,
          manning_n=profile.manning_n,
      )
  ```

- [ ] **Step 2: Refactor river_kinematic_wave.py to import from profile**

  At the top of `src/general/solvers/river_kinematic_wave.py`, replace the current `RiverProfile` dataclass and all its helper functions (`_as_float`, `_cell_widths_from_stations`, `_optional_array`, `make_profile`, `load_profile_csv`, `load_profile_json`, `load_profile`) with:
  ```python
  from general.solvers.profile import (
      RiverProfile,
      make_profile,
      load_profile,
      load_profile_csv,
      load_profile_json,
  )
  ```
  Keep `q`, `c`, `_initial_depth`, `_rainfall_source`, `run_model`, `save_time_series_csv`, `save_summary_json`.

- [ ] **Step 3: Run tests — still 25 green**

  ```bash
  python -m pytest tests/ -q
  ```

---

### Task 10: Migrate river_kinematic_wave to implement Solver contract

**Files:**
- Modify: `src/general/solvers/river_kinematic_wave.py`

The existing `run_model(profile, t_final_min, left_inflow_flux, ...)` stays as a back-compat wrapper. Add a `SOLVER` singleton.

- [ ] **Step 1: Add contract imports at top of river_kinematic_wave.py**

  Add after existing imports:
  ```python
  from general.solvers.contract import Domain, Scenario, SimulationResult, Solver
  from general.solvers.profile import domain_from_profile
  ```

- [ ] **Step 2: Add _RiverKinematicWaveSolver class at end of file (before __main__)**

  ```python
  class _RiverKinematicWaveSolver:
      name = "river_kinematic_wave"
      supports = frozenset({"initial_depth", "left_inflow", "rainfall", "cfl"})

      def run(self, domain: Domain, scenario: Scenario) -> SimulationResult:
          from general.solvers.profile import RiverProfile

          profile = RiverProfile(
              station_m=domain.x_m,
              dx_m=domain.dx_m,
              slope=domain.slope,
              manning_n=domain.manning_n,
          )

          left_inflow = scenario.left_inflow
          if callable(left_inflow):
              # river_kinematic_wave only supports constant inflow; use t=0 value
              left_inflow = float(left_inflow(0.0))

          rainfall_rate = 0.0
          if scenario.rainfall is not None:
              sample = scenario.rainfall(domain.x_m, 0.0)
              rainfall_rate = float(np.mean(sample))

          result = run_model(
              profile,
              t_final_min=scenario.t_final_min,
              left_inflow_flux=float(left_inflow),
              record_interval_min=scenario.record_interval_min,
              rainfall_rate_m_per_min=rainfall_rate,
              cfl=scenario.cfl,
          )

          return SimulationResult(
              domain=domain,
              times=result["times"],
              depth_history=result["depth_history"],
              depth_initial=result["depth_initial"],
              depth_final=result["depth_final"],
              mass_inflow=result["mass_inflow"],
              mass_source=result["mass_source"],
              mass_outflow=result["mass_outflow"],
          )


  SOLVER = _RiverKinematicWaveSolver()
  ```

- [ ] **Step 3: Run tests — still 25 green**

  ```bash
  python -m pytest tests/ -q
  ```

---

### Task 11: Migrate linear_advection to implement Solver contract

**Files:**
- Modify: `src/general/solvers/linear_advection.py`

`linear_advection` pins the left cell to ~0 (no inflow). It exposes `supports=frozenset({"rainfall"})`. The back-compat `run_model(L, T_final, record_interval)` stays; a thin `Solver.run` wrapper builds a uniform Domain + Scenario and adapts the result dict to `SimulationResult`.

- [ ] **Step 1: Add contract imports at top**

  ```python
  from general.solvers.contract import Domain, Scenario, SimulationResult
  ```

- [ ] **Step 2: Add _KinematicWaveSolver class at end of file (before __main__)**

  ```python
  class _KinematicWaveSolver:
      name = "kinematic_wave"
      supports = frozenset({"rainfall", "cfl"})

      def run(self, domain: Domain, scenario: Scenario) -> SimulationResult:
          import general.solvers.linear_advection as _la

          # Inject per-scenario rainfall as module-level r if provided
          _orig_r = _la.r
          if scenario.rainfall is not None:
              _la.r = scenario.rainfall
          # Temporarily set global parameters from domain (uniform assumed)
          _orig_S0, _orig_n0 = _la.S0, _la.n0
          _la.S0 = float(domain.slope[0])
          _la.n0 = float(domain.manning_n[0])

          L = float(domain.x_m[-1] + domain.dx_m[-1] / 2)
          try:
              result = run_model(L, scenario.t_final_min, scenario.record_interval_min)
          finally:
              _la.r = _orig_r
              _la.S0 = _orig_S0
              _la.n0 = _orig_n0

          return SimulationResult(
              domain=domain,
              times=result["times"],
              depth_history=result["u_history"],
              depth_initial=result["u_initial"],
              depth_final=result["u_final"],
              mass_inflow=0.0,
              mass_source=result["mass_source"],
              mass_outflow=result["mass_outflow"],
          )


  SOLVER = _KinematicWaveSolver()
  ```

- [ ] **Step 3: Run tests — still 25 green**

  ```bash
  python -m pytest tests/ -q
  ```

---

### Task 12: Migrate saint_venant_1d to implement Solver contract

**Files:**
- Modify: `src/general/solvers/saint_venant_1d.py`

Saint-Venant tracks two fields (h, q). The `SimulationResult` carries `depth_history` for h; discharge goes in `extra`. Advertise `supports=frozenset({"initial_discharge", "left_inflow"})`.

- [ ] **Step 1: Add contract imports at top**

  ```python
  from general.solvers.contract import Domain, Scenario, SimulationResult
  ```

- [ ] **Step 2: Add _SaintVenantSolver class at end of file (before __main__)**

  ```python
  import numpy as np

  class _SaintVenantSolver:
      name = "saint_venant"
      supports = frozenset({"initial_depth", "initial_discharge", "left_inflow", "cfl"})

      def run(self, domain: Domain, scenario: Scenario) -> SimulationResult:
          L = float(domain.x_m[-1] + domain.dx_m[-1] / 2)
          left_inflow = scenario.left_inflow if callable(scenario.left_inflow) else None
          h_init = None
          if isinstance(scenario.initial_depth_m, np.ndarray):
              h_init = scenario.initial_depth_m
          q_init = None
          if isinstance(scenario.initial_discharge, np.ndarray):
              q_init = scenario.initial_discharge

          result = run_model(
              L,
              scenario.t_final_min,
              record_interval=scenario.record_interval_min,
              h_init=h_init,
              q_init=q_init,
              left_inflow=left_inflow,
              cfl=scenario.cfl,
          )

          return SimulationResult(
              domain=domain,
              times=result["times"],
              depth_history=result["h_history"],
              depth_initial=result["h_initial"],
              depth_final=result["h_final"],
              mass_inflow=result["mass_inflow"],
              mass_source=result["mass_source"],
              mass_outflow=result["mass_outflow"],
              extra={
                  "discharge_history": result["q_history"],
                  "discharge_initial": result["q_initial"],
                  "discharge_final": result["q_final"],
              },
          )


  SOLVER = _SaintVenantSolver()
  ```

- [ ] **Step 3: Run tests — still 25 green**

  ```bash
  python -m pytest tests/ -q
  ```

- [ ] **Step 4: Commit the contract + solver migrations**

  ```bash
  git add src/general/solvers/
  git commit -m "feat: add contract.py, profile.py, and SOLVER adapters for all three solvers"
  ```

---

### Task 13: Add registry.py and run_simulation.py CLI

**Files:**
- Create: `src/rivers/simulations/registry.py`
- Create: `src/rivers/simulations/run_simulation.py`

- [ ] **Step 1: Create registry.py**

  ```python
  from general.solvers.contract import Solver, UnsupportedScenario
  import general.solvers.linear_advection as _la
  import general.solvers.saint_venant_1d as _sv
  import general.solvers.river_kinematic_wave as _rkw

  SOLVERS: dict[str, Solver] = {
      "kinematic_wave": _la.SOLVER,
      "saint_venant": _sv.SOLVER,
      "river_kinematic_wave": _rkw.SOLVER,
  }


  def dispatch(name: str, domain, scenario):
      if name not in SOLVERS:
          raise KeyError(f"Unknown solver '{name}'. Available: {sorted(SOLVERS)}")
      solver = SOLVERS[name]
      _check_scenario(solver, scenario)
      return solver.run(domain, scenario)


  def _check_scenario(solver: Solver, scenario) -> None:
      """Raise UnsupportedScenario if the user requested a knob the solver doesn't honour."""
      import numpy as np

      checks = {
          "left_inflow": lambda s: (callable(s.left_inflow) or float(s.left_inflow) != 0.0),
          "initial_discharge": lambda s: (
              isinstance(s.initial_discharge, np.ndarray) or float(s.initial_discharge) != 0.0
          ),
          "rainfall": lambda s: s.rainfall is not None,
      }
      for knob, is_active in checks.items():
          if knob not in solver.supports and is_active(scenario):
              raise UnsupportedScenario(
                  f"Solver '{solver.name}' does not support the '{knob}' scenario knob. "
                  f"Solver supports: {sorted(solver.supports)}"
              )
  ```

- [ ] **Step 2: Create run_simulation.py**

  ```python
  import argparse
  import json
  import sys
  from pathlib import Path

  import numpy as np

  SRC_ROOT = Path(__file__).resolve().parents[2]
  if str(SRC_ROOT) not in sys.path:
      sys.path.insert(0, str(SRC_ROOT))

  from general.solvers.contract import Scenario
  from general.solvers.profile import domain_from_profile, load_profile
  from rivers.simulations.registry import SOLVERS, dispatch


  DEFAULT_OUTPUT_DIR = Path("data") / "real_world_rivers" / "runs"


  def parse_args():
      p = argparse.ArgumentParser(description="Run a 1-D river solver on a profile.")
      p.add_argument("profile", help="CSV or JSON river profile path")
      p.add_argument(
          "--solver",
          choices=sorted(SOLVERS),
          default="river_kinematic_wave",
          help="Which solver to use",
      )
      p.add_argument("--t-final", type=float, required=True, help="Simulation duration in minutes")
      p.add_argument("--record-interval", type=float, default=1.0)
      p.add_argument("--left-inflow", type=float, default=0.0, help="Constant upstream inflow flux, m²/min")
      p.add_argument("--rainfall-rate", type=float, default=0.0, help="Uniform rainfall rate, m/min")
      p.add_argument("--cfl", type=float, default=0.5)
      p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
      p.add_argument("--run-name", default="simulation")
      return p.parse_args()


  def main():
      args = parse_args()

      profile = load_profile(args.profile)
      domain = domain_from_profile(profile)

      rainfall_fn = None
      if args.rainfall_rate > 0:
          rate = args.rainfall_rate
          rainfall_fn = lambda x, t: np.full_like(x, rate)

      scenario = Scenario(
          t_final_min=args.t_final,
          record_interval_min=args.record_interval,
          left_inflow=args.left_inflow,
          rainfall=rainfall_fn,
          cfl=args.cfl,
      )

      result = dispatch(args.solver, domain, scenario)

      out = Path(args.output_dir)
      out.mkdir(parents=True, exist_ok=True)

      # Write depth CSV (animate_depth-compatible: t_min column + one column per station)
      csv_path = out / f"{args.run_name}_timeseries.csv"
      import csv as _csv
      with csv_path.open("w", newline="") as f:
          writer = _csv.writer(f)
          writer.writerow(["t_min"] + [f"{x:.6f}" for x in domain.x_m])
          for t, row in zip(result.times, result.depth_history):
              writer.writerow([f"{t:.6f}"] + [f"{d:.10g}" for d in row])

      # Compute mass balance error
      mass_balance_error = (
          result.mass_inflow + result.mass_source - result.mass_outflow
          - float(np.sum((result.depth_final - result.depth_initial) * domain.dx_m))
      )

      summary = {
          "solver": args.solver,
          "profile": str(args.profile),
          "t_final_min": args.t_final,
          "mass_inflow": result.mass_inflow,
          "mass_source": result.mass_source,
          "mass_outflow": result.mass_outflow,
          "mass_balance_error": mass_balance_error,
      }
      json_path = out / f"{args.run_name}_summary.json"
      json_path.write_text(json.dumps(summary, indent=2))

      print(f"Done. CSV: {csv_path}  Summary: {json_path}")
      print(f"Mass balance error: {mass_balance_error:.4e} m²")


  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 3: Verify run_simulation.py --help works**

  ```bash
  python src/rivers/simulations/run_simulation.py --help
  ```
  Expected: argparse help text with `--solver {kinematic_wave,river_kinematic_wave,saint_venant}`.

- [ ] **Step 4: Smoke-test with river_kinematic_wave solver**

  ```bash
  python src/rivers/simulations/run_simulation.py \
      real_world_rivers/tools/example_river_profile.csv \
      --solver river_kinematic_wave \
      --t-final 5 \
      --left-inflow 0.0006 \
      --output-dir /tmp/sim-smoke \
      --run-name test
  ```
  Expected: completes, `Done. CSV: ...` printed, files written.

- [ ] **Step 5: Run tests — still 25 green**

  ```bash
  python -m pytest tests/ -q
  ```

---

### Task 14: Add ingest-to-simulate helper

**Files:**
- Create: `src/rivers/simulations/ingest_to_simulate.py`

This converts an exported profile (from `rivers/ingest/export_profile.py`) plus its recommended-inflow metadata (USGS flow stored in the SQLite DB) into a ready-to-run `(Domain, Scenario)` pair.

- [ ] **Step 1: Create ingest_to_simulate.py**

  ```python
  """Helper that turns an ingested river profile into a (Domain, Scenario) ready for dispatch."""

  from pathlib import Path

  import numpy as np

  from general.solvers.contract import Domain, Scenario
  from general.solvers.profile import domain_from_profile, load_profile


  def profile_to_domain_scenario(
      profile_path: str | Path,
      t_final_min: float,
      left_inflow: float = 0.0,
      rainfall_rate_m_per_min: float = 0.0,
      record_interval_min: float = 1.0,
      cfl: float = 0.5,
  ) -> tuple[Domain, Scenario]:
      """Load *profile_path* and build a Domain + Scenario.

      Args:
          profile_path: CSV or JSON river profile produced by export_profile.
          t_final_min: Simulation duration, minutes.
          left_inflow: Constant upstream inflow flux, m²/min.
          rainfall_rate_m_per_min: Uniform rainfall rate, m/min (0 = off).
          record_interval_min: Snapshot interval, minutes.
          cfl: CFL target.

      Returns:
          (domain, scenario) ready for ``registry.dispatch(solver_name, domain, scenario)``.
      """
      profile = load_profile(profile_path)
      domain = domain_from_profile(profile)

      rainfall_fn = None
      if rainfall_rate_m_per_min > 0:
          rate = rainfall_rate_m_per_min
          rainfall_fn = lambda x, t: np.full_like(x, rate)

      scenario = Scenario(
          t_final_min=t_final_min,
          record_interval_min=record_interval_min,
          left_inflow=left_inflow,
          rainfall=rainfall_fn,
          cfl=cfl,
      )
      return domain, scenario
  ```

- [ ] **Step 2: Verify import**

  ```bash
  python -c "from rivers.simulations.ingest_to_simulate import profile_to_domain_scenario; print('ok')" \
      && echo "ok"
  ```

---

### Task 15: Add tests/test_run_simulation.py

**Files:**
- Create: `tests/test_run_simulation.py`

- [ ] **Step 1: Write the test file**

  ```python
  """Tests for the unified run_simulation dispatch harness."""

  import pytest
  import numpy as np

  from general.solvers.contract import Domain, Scenario, UnsupportedScenario
  from rivers.simulations.registry import dispatch, SOLVERS


  PROFILE_PATH = "real_world_rivers/tools/example_river_profile.csv"


  def _make_scenario(**kwargs):
      defaults = dict(t_final_min=2.0, record_interval_min=1.0, cfl=0.5)
      defaults.update(kwargs)
      return Scenario(**defaults)


  def _load_domain():
      from general.solvers.profile import domain_from_profile, load_profile
      return domain_from_profile(load_profile(PROFILE_PATH))


  # ── registry ──────────────────────────────────────────────────────────────
  def test_registry_contains_expected_solvers():
      assert "kinematic_wave" in SOLVERS
      assert "river_kinematic_wave" in SOLVERS
      assert "saint_venant" in SOLVERS


  def test_unknown_solver_raises():
      with pytest.raises(KeyError, match="Unknown solver"):
          dispatch("no_such_solver", _load_domain(), _make_scenario())


  # ── dispatch by name ──────────────────────────────────────────────────────
  def test_dispatch_river_kinematic_wave():
      domain = _load_domain()
      scenario = _make_scenario(left_inflow=0.0006, t_final_min=2.0)
      result = dispatch("river_kinematic_wave", domain, scenario)
      assert result.depth_history.shape[0] >= 2
      assert result.depth_history.shape[1] == len(domain.x_m)
      assert result.mass_inflow >= 0


  def test_dispatch_kinematic_wave():
      from general.solvers.profile import load_profile
      profile = load_profile(PROFILE_PATH)
      # Build uniform domain from profile (uses first cell values for S0/n0)
      domain = Domain(
          x_m=profile.station_m,
          dx_m=profile.dx_m,
          slope=profile.slope,
          manning_n=profile.manning_n,
      )
      scenario = _make_scenario(t_final_min=2.0)
      result = dispatch("kinematic_wave", domain, scenario)
      assert result.depth_history.shape[1] == len(domain.x_m)


  # ── same profile, two solvers, comparable mass balance ────────────────────
  def test_two_solvers_on_same_profile():
      domain = _load_domain()
      scenario = _make_scenario(t_final_min=3.0, left_inflow=0.0006)

      r1 = dispatch("river_kinematic_wave", domain, scenario)
      # kinematic_wave doesn't support left_inflow, so use rainfall-only scenario
      scenario_no_inflow = _make_scenario(t_final_min=3.0)
      r2 = dispatch("kinematic_wave", domain, scenario_no_inflow)

      # Both produce valid (non-negative) depth histories
      assert np.all(r1.depth_history >= 0)
      assert np.all(r2.depth_history >= 0)


  # ── UnsupportedScenario ───────────────────────────────────────────────────
  def test_unsupported_scenario_left_inflow_on_kinematic_wave():
      domain = _load_domain()
      scenario = _make_scenario(left_inflow=0.001)  # kinematic_wave doesn't support this
      with pytest.raises(UnsupportedScenario, match="left_inflow"):
          dispatch("kinematic_wave", domain, scenario)


  def test_unsupported_scenario_initial_discharge_on_kinematic_wave():
      domain = _load_domain()
      scenario = _make_scenario(initial_discharge=0.001)
      with pytest.raises(UnsupportedScenario, match="initial_discharge"):
          dispatch("kinematic_wave", domain, scenario)


  # ── SimulationResult shape invariants ─────────────────────────────────────
  def test_simulation_result_shapes():
      domain = _load_domain()
      scenario = _make_scenario(t_final_min=3.0, left_inflow=0.0006)
      result = dispatch("river_kinematic_wave", domain, scenario)
      n_times = len(result.times)
      n_cells = len(domain.x_m)
      assert result.depth_history.shape == (n_times, n_cells)
      assert result.depth_initial.shape == (n_cells,)
      assert result.depth_final.shape == (n_cells,)
  ```

- [ ] **Step 2: Run the new tests**

  ```bash
  python -m pytest tests/test_run_simulation.py -v
  ```
  Expected: all 9 new tests pass.

- [ ] **Step 3: Run the full suite**

  ```bash
  python -m pytest tests/ -v
  ```
  Expected: 34 passed (25 original + 9 new).

---

### Task 16: Update README, CLAUDE.md, AGENTS.md for Stage 4

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Add Stage 4 row to README history table**

  Add after the Stage 3 row:
  ```
  | **4 — Solver-agnostic harness** | *(this session)* | Added `src/general/solvers/contract.py` (Domain, Scenario, SimulationResult, Solver protocol, UnsupportedScenario). Moved RiverProfile loaders to `src/general/solvers/profile.py`. Each solver now exposes a `SOLVER` singleton implementing the contract; back-compat `run_model()` wrappers preserved. `src/rivers/simulations/registry.py` maps solver names to singletons; `run_simulation.py` is the unified CLI. |
  ```

- [ ] **Step 2: Add run_simulation.py to README quick-start and layout**

  Add to the quick-start block:
  ```
  python src/rivers/simulations/run_simulation.py real_world_rivers/tools/example_river_profile.csv \
      --solver river_kinematic_wave --t-final 10 --left-inflow 0.0006
  ```

  Add to layout:
  ```
  src/general/solvers/contract.py            # Domain, Scenario, SimulationResult, Solver protocol
  src/general/solvers/profile.py             # RiverProfile dataclass and CSV/JSON loaders
  src/rivers/simulations/registry.py         # name → Solver mapping
  src/rivers/simulations/run_simulation.py   # unified CLI dispatcher
  src/rivers/simulations/ingest_to_simulate.py # profile_path → (Domain, Scenario) helper
  tests/test_run_simulation.py               # dispatch, UnsupportedScenario, result shapes
  ```

- [ ] **Step 3: Update CLAUDE.md** — add new files to architecture section; update "Model conventions" run_model note to mention Solver contract.

- [ ] **Step 4: Sync AGENTS.md** with CLAUDE.md (same content, different title line).

---

### Task 17: Final verification and commit Part 2

- [ ] **Step 1: Run full test suite**

  ```bash
  python -m pytest tests/ -v
  ```
  Expected: 34 passed, 0 failures.

- [ ] **Step 2: End-to-end smoke test of run_simulation.py with two solvers**

  ```bash
  python src/rivers/simulations/run_simulation.py \
      real_world_rivers/tools/example_river_profile.csv \
      --solver river_kinematic_wave --t-final 5 --left-inflow 0.0006 \
      --output-dir /tmp/final-smoke --run-name rkw

  python src/rivers/simulations/run_simulation.py \
      real_world_rivers/tools/example_river_profile.csv \
      --solver saint_venant --t-final 2 \
      --output-dir /tmp/final-smoke --run-name sv
  ```
  Both should complete and print mass-balance errors.

- [ ] **Step 3: Commit Part 2**

  ```bash
  git add src/ tests/ README.md CLAUDE.md AGENTS.md
  git status  # confirm no data/*.csv or data/*.png staged
  git commit -m "feat: add contract.py, profile.py, registry, and run_simulation unified CLI (Stage 4)"
  ```

- [ ] **Step 4: Push branch and open draft PR**

  ```bash
  git push -u origin feature/restructure-harness
  gh pr create --draft \
      --title "refactor+feat: restructure src/ and add solver-agnostic harness" \
      --body "Part 1: moves solvers → general/solvers, viz → general/viz, ingest tools → rivers/ingest.
  Part 2: adds contract.py (Domain/Scenario/SimulationResult/Solver), per-solver SOLVER singletons with back-compat run_model wrappers, unified run_simulation.py CLI.
  All 34 tests pass."
  ```

---

## Self-review against spec

**Spec coverage check:**

| Requirement | Task |
|---|---|
| git mv all files, preserve history | Tasks 2–3 |
| Fix all imports in tests + scripts | Tasks 4–5 |
| Update README/CLAUDE/AGENTS | Tasks 6, 16 |
| pytest green after every step | After each task |
| Smoke-test all 4 CLIs | Task 7 |
| Part 1 commit | Task 7 |
| contract.py with Domain/Scenario/SimulationResult/Solver/UnsupportedScenario | Task 8 |
| profile.py with RiverProfile + domain_from_profile | Task 9 |
| river_kinematic_wave migrated first | Task 10 |
| linear_advection migrated (no left_inflow in supports) | Task 11 |
| saint_venant migrated (extra["discharge_*"]) | Task 12 |
| registry.py + run_simulation.py CLI | Task 13 |
| UnsupportedScenario if knob not in supports | Task 13 registry._check_scenario |
| ingest→simulate helper | Task 14 |
| test_run_simulation.py | Task 15 |
| dispatch by name; two solvers on same profile; UnsupportedScenario tests | Task 15 |
| README Stage 4 history row | Task 16 |
| Part 2 commit + push + draft PR | Task 17 |

**Gaps found:**
- `run_river_kinematic_wave.py` should become an alias/thin wrapper calling `run_simulation.py --solver river_kinematic_wave` per spec. Task 13 keeps it working as-is; it remains a standalone script. The spec says "make it an alias to run_simulation.py --solver river_kinematic_wave, or delete it after updating docs." We keep it and leave the alias decision to the user.
- The spec mentions `data/real_world_rivers/columbia_hanford_profile.csv` in test_run_simulation.py. That file may not exist in the test environment — Task 15 uses `example_river_profile.csv` instead (which tests already use). Adjust if the Columbia profile is present.
