# Real-world river ingestion

This pipeline turns public river data into **validated, provenance-rich,
model-ready profiles** that load through the existing
`general.solvers.profile.load_profile` interface. It lives entirely under
`src/rivers/ingest/` and writes to a SQLite database plus CSV/JSON profile files
with a `.metadata.json` provenance sidecar.

Its responsibility ends at producing a profile. It does **not** run the numerical
models, interpret flood outcomes, or calibrate anything.

---

## Data sources and their limitations

| Quantity | Provider | Notes / limitations |
|---|---|---|
| Terrain elevation → slope | Open-Meteo Elevation API (Copernicus **GLO-90** DEM) | ~90 m horizontal resolution; a **point sample at each marker**, not a surveyed thalweg. Slopes are endpoint differences between markers and can be noisy or locally negative — see the slope floor below. |
| Discharge (flow) | **USGS** Water Data OGC API, parameter `00060` (continuous) | US gauges only. A gauge reading is applied at one marker; extending it along the reach is an assumption. |
| Precipitation (rainfall) | Open-Meteo Historical Weather API (reanalysis) | Hourly, preceding-hour sum. Sampled at a **single marker** unless a `marker_order` is given per call; reanalysis is a model product, not a rain gauge. |
| Channel geometry (width, depth) | Reviewed input CSV | Not fetched; supplied and reviewed by a human. Often a **literature estimate** — classify accordingly. |
| Roughness (Manning's *n*) | Reviewed input CSV | Not fetched; expert/literature value. **Unit convention is currently unsettled — see [ingestion_integration_requests.md](ingestion_integration_requests.md).** |

Guardrails: the pipeline never invents missing observations, and estimated or
fallback values are labelled as such (see *Provenance classification*).

---

## Curated reach definitions (configuration format)

Reaches are declared as JSON files under `real_world_rivers/curated/`. One file
per reach. Paths inside a definition are resolved **relative to the definition
file's own directory**.

```json
{
  "river":    {"name": "Columbia", "region": "Washington", "country": "US", "notes": "..."},
  "reach":    {"name": "Hanford Reach", "notes": "upstream to downstream"},
  "markers":  "../columbia_hanford_markers.csv",
  "elevation":{"provider": "open-meteo-dem"},
  "roughness":{"file": "../columbia_hanford_roughness.csv"},
  "geometry": {"file": "../columbia_hanford_geometry.csv"},
  "flow":     {"site": "12472800", "start": "2020-01-01T00:00:00Z",
               "end": "2020-01-07T00:00:00Z", "marker_order": 0},
  "rainfall": {"start_date": "2020-01-01", "end_date": "2020-01-07", "marker_order": 1},
  "export":   {"output": "columbia_hanford.profile.csv", "minimum_slope": 1e-6,
               "initial_depth_m": null,
               "rainfall_window": ["2020-01-01T00:00", "2020-01-07T23:59:59"],
               "flow_window": ["2020-01-01T00:00:00Z", "2020-01-07T00:00:00Z"]}
}
```

- **Required sections:** `river`, `reach`, `markers`, `export`.
- **Optional sections:** `elevation`, `roughness`, `geometry`, `flow`, `rainfall`.
  A section's *presence* is what triggers that step. Without `elevation` there are
  no slopes and export will fail (a profile needs slope); without `flow` there is
  no inflow recommendation; and so on.
- JSON is used deliberately — the repo depends only on numpy/matplotlib/pytest,
  so no YAML dependency is introduced.

Marker files are CSV, JSON, or GeoJSON LineString with `lat`/`lon` (and optional
`station_m`, `label`). If `station_m` is omitted it is derived from great-circle
(haversine) distances. Roughness/geometry files are reviewed CSVs; both accept an
optional `classification` column (see below).

---

## Commands

Run as a script (adds `src/` to the path itself), like the other ingest tools:

```bash
# One reach
python src/rivers/ingest/run_ingestion.py real_world_rivers/curated/columbia_hanford.json

# Every *.json definition in a directory (batch)
python src/rivers/ingest/run_ingestion.py --all real_world_rivers/curated

# Options
#   --db <path>     SQLite database (default: data/real_world_rivers/river_inputs.sqlite)
#   --replace       replace existing reach data instead of erroring on conflict
```

**Exit status** is non-zero if any reach failed *or* any declared export file is
missing from disk afterwards — so the command is safe to gate a pipeline on.
Batch runs continue past a failing reach and report every reach's status.

The lower-level per-step CLI (`collect_river_data.py`: `create-reach`,
`fetch-elevation`, `fetch-flow`, `fetch-rainfall`, `import-roughness`,
`import-geometry`, `export-profile`) still exists for manual/one-off work.

---

## Database and unit conventions

- **Units at the model-facing boundary are meters and minutes.** Providers'
  original units are preserved in provenance (e.g. USGS ft³/s and its converted
  m³/min are both stored; rainfall keeps its original mm value).
- The schema is `data/real_world_rivers/schema.sql`. It separates raw provider
  values, normalized values, and derived quantities, and every sample/observation
  carries a `source_id` and a `classification`.
- **Manning's *n* unit convention is currently ambiguous** and is emitted as
  provided — do not assume it is SI. See
  [ingestion_integration_requests.md](ingestion_integration_requests.md).

### Schema changes require a re-init

`initialize_database` uses `CREATE TABLE IF NOT EXISTS`, so changing `schema.sql`
does **not** migrate an existing `.sqlite` file. Because databases are
reproducible and gitignored, delete and rebuild rather than migrate:

```bash
rm data/real_world_rivers/river_inputs.sqlite
python src/rivers/ingest/run_ingestion.py --all real_world_rivers/curated
```

---

## Retry, caching, and replacement behavior

- **Retries:** provider HTTP requests (`common.request_json`) have a bounded
  timeout and retry 5xx/network errors with exponential backoff; 4xx errors are
  not retried. Credentials in the URL (`api_key`, `token`, …) are redacted from
  stored URLs, logs, **and exception messages**.
- **Deduplication:** re-running the same ingestion does **not** create duplicate
  logical observations. Each observation has a run-independent natural key
  (reach + gauge + timestamp, reach + station, reach + interval, …) and providers
  upsert on it. `add_source` is idempotent on its metadata, so re-runs don't mint
  fresh source rows.
- **Re-running is idempotent.** Running a reach (or `--all`) again *without*
  `--replace` reuses the existing reach and re-runs the data steps, which upsert
  on natural keys — no duplicates, no error. `--all` also skips generated
  sidecars (`*.profile.metadata.json`, `*.profile.json`) so they are never
  mistaken for definitions.
- **Replacement** is explicit and per-invocation via `--replace`, which rebuilds
  the reach and its markers from the definition. It is never baked into a
  curated definition.
- **Caching:** no on-disk provider cache is shipped; `cache/` and `downloads/`
  directories under `real_world_rivers/` are gitignored if you add one.
- **Atomicity & partial failure:** profile and metadata are written atomically
  (temp file + `os.replace`), sidecar first, so a reader never sees a profile
  without its metadata. A provider failure rolls back its transaction and closes
  the connection, leaving previously stored data intact and the DB writable.

---

## Validation rules

Every export runs `validation.py` on the assembled rows **before writing
anything**. Findings have three severities:

- **error** — blocks export; nothing is written. Covers: empty profile, fewer
  than two stations, non-increasing or duplicate stations, non-finite/missing
  values, non-positive slope or roughness, negative rainfall, invalid WGS84
  coordinates.
- **warning** — retained in metadata and the batch summary; export proceeds.
  Covers: suspicious slope/roughness/rainfall ranges, station gaps, incomplete or
  unknown provenance, no observed-classified values, and temporal coverage gaps.
- **info** — neutral context: the provenance-classification summary and each
  adjustment record.

Validation findings are stored in the metadata sidecar under `validation`.

### Adjustments

When a value is altered — currently the `minimum_slope` floor applied to
non-positive DEM-derived slopes — the pipeline records the **original value, the
adjusted value, the rule, and the affected-value count** (in `adjustments` and
`slope_values_adjusted`). Floored slopes are reclassified `fallback`.

---

## Provenance classification

Every exported value is classified so it can be traced and trusted:

- **observed** — measured/provider-reported (DEM elevation, gauge discharge,
  reanalysis rainfall, surveyed geometry, reviewed marker stations).
- **derived** — computed from stored values (slope from elevations, rainfall
  *rate* averaged from observations, the inflow recommendation).
- **estimated** — expert/literature value not measured for this reach (Manning's
  *n*, literature geometry).
- **fallback** — a default substituted for a missing/invalid value (floored
  slope, a supplied `initial_depth_m`).

The metadata sidecar records a per-row `classification` for `station_m`, `slope`,
`manning_n` (and `rainfall_rate_m_per_min`/`initial_depth_m` when present), a
coverage summary, and the reach's data sources.

---

## Generated vs. reviewed artifacts

| Reviewed (committed) | Generated (gitignored, reproducible) |
|---|---|
| `real_world_rivers/curated/*.json` definitions | `real_world_rivers/curated/*.profile.csv` / `.profile.json` |
| Source `*_markers.csv`, `*_roughness.csv`, `*_geometry.csv` | `*.profile.metadata.json` sidecars |
| `data/real_world_rivers/schema.sql` | `data/real_world_rivers/river_inputs.sqlite` |
| | provider `cache/` / `downloads/` |

Never hand-edit a generated file; change the curated definition or source CSV and
re-run.

---

## Reproducing an export

From a fresh checkout (needs network for elevation/flow/rainfall providers):

```bash
python src/rivers/ingest/run_ingestion.py real_world_rivers/curated/columbia_hanford.json
python -m pytest tests/test_river_data_tools.py
```

The generated `real_world_rivers/curated/columbia_hanford.profile.csv` then loads
through `general.solvers.profile.load_profile`. The full offline pipeline
(curated → mocked providers → db → validation → export → `load_profile`) is
covered without network by `tests/test_ingestion_orchestrator.py` and
`tests/test_river_data_tools.py`.

---

## Assumptions that still require calibration

A profile passing validation is **not** calibrated. Known assumptions to revisit:

1. **Manning's *n* convention and values** — unresolved unit convention (see
   [ingestion_integration_requests.md](ingestion_integration_requests.md)) and
   literature/estimated magnitudes, not calibrated against observed stage/flow.
2. **DEM slopes** — GLO-90 point samples over long marker spacings; endpoint
   differencing smooths real bed variation and can require the slope floor.
3. **Single-point forcing** — rainfall (and often flow) sampled at one marker and
   applied more broadly; spatial variation is not captured unless configured.
4. **Literature geometry** — width/depth estimates, not surveyed cross-sections.
5. **Slope floor** — a numerical fallback, not a physical measurement.

Do not treat model output built on these inputs as validated against reality
until these are calibrated.
