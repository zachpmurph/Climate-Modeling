# Flood reporting contract

The reporting layer is a consumer of saved model artifacts. It does not import,
run, tune, or modify a solver.

## Required artifact

The required input is a depth time-series CSV:

```text
t_min,0,100,200
0,0.1,0.1,0.1
10,0.2,0.4,0.3
```

- The first column is time in minutes.
- Remaining headers are strictly increasing river stations in meters.
- Each row contains non-negative water depth in meters.
- Times must be strictly increasing.

This is the format already written by
`src/rivers/simulations/run_simulation.py`.

For a 1-D Saint-Venant run, the runner also writes
`<run>_discharge.csv` on the identical time/station grid. The run summary's
`discharge_path` points to it and `discharge_unit` distinguishes
whole-channel `m3_per_min` (reviewed width supplied) from unit-width
`m2_per_min`. This optional companion artifact supports observed-flow
validation; the flood-outcome report remains depth based.

For a 2-D run, the summary additionally points to `<run>_fields.npz`. That
artifact contains `x_m`, `y_m`, `dx_m`, `dy_m`, `bed_elevation_m`,
`slope_x`, `slope_y`, `manning_n`, `times_min`, `depth_m`,
`discharge_x_m2_per_min`, and `discharge_y_m2_per_min`. Time-dependent fields have shape
`(snapshot, x, y)`. The CSV remains a cross-channel mean for compatibility;
the reporter automatically uses the NPZ for 2-D outcomes.

## Optional run summary

An adjacent `<run>_summary.json` is discovered automatically for a
`<run>_timeseries.csv` input. It may also be supplied explicitly.

The reporter recognizes these numerical diagnostic fields when present:

- `mass_inflow`
- `mass_source`
- `mass_outflow`
- `mass_correction` (for explicitly tracked numerical floor corrections)
- `mass_balance_error`

Other scalar summary fields are displayed as run context. Unknown fields do not
break the report.

## Optional flood threshold

A water-depth result alone does not define flooding. Threshold outcomes are
therefore calculated only when one of these is supplied:

1. A geometry CSV containing strictly increasing `station_m` and positive
   `bankfull_depth_m`, covering the full modeled reach.
2. An explicit uniform depth threshold in meters.

If neither is supplied, the report still shows modeled depths but marks flood
threshold outcomes as not assessed.

## Produced artifacts

The command writes:

- A self-contained interactive HTML report.
- A machine-readable `.outcomes.json` file with `schema_version: 1`.

For 1-D runs, outcomes include peak modeled depth, its time and station, maximum
depth change, and threshold-exceedance length. For 2-D runs, the interactive
plot is a plan-view depth map and outcomes include peak x/y location and
threshold-exceedance area.

These are screening outcomes. A 1-D result is not an inundation boundary, and a
2-D result is only as reliable as its terrain, boundary conditions, roughness,
forcing, calibration, and validation.

## 2-D uncertainty artifact

`run_2d_ensemble.py` writes `<run>_ensemble.npz` and
`<run>_ensemble_summary.json`. The numeric artifact contains:

- sampled parameter names and values, plus every member mass residual;
- requested quantile probabilities;
- time-dependent depth and water-surface-elevation quantiles;
- terrain, peak-depth, and peak-water-surface quantiles;
- probability that each cell exceeds the configured wet-depth threshold; and
- member and quantile maximum wet areas.

Most sampled parameters are multiplicative scales. Downstream-stage uncertainty
is an additive offset in metres and is available only when the base 2-D run uses
a stage boundary in the same vertical datum as its terrain.

`generate_uncertainty_report.py` consumes those saved artifacts without
importing a solver and writes a self-contained spatial report. Its probabilities
are conditional on the configured parameter ranges and model structure. They
must not be described as forecast probabilities unless the input distributions
have been independently estimated and the ensemble has been probabilistically
validated.

The ensemble configuration accepts either reproducible Latin-hypercube marginal
ranges or explicit joint parameter samples. Use joint samples when inputs are
correlated; independent ranges can otherwise create physically inconsistent
cross-sections or forcing combinations.

## Command

With reviewed bankfull geometry:

```text
python src/rivers/reporting/generate_flood_report.py \
    data/real_world_rivers/runs/example_timeseries.csv \
    --geometry real_world_rivers/tools/example_geometry.csv
```

With an explicit screening threshold:

```text
python src/rivers/reporting/generate_flood_report.py \
    data/real_world_rivers/runs/example_timeseries.csv \
    --depth-threshold 0.5
```

The reporting contract is intentionally separate from `SimulationResult`.
Model development can change internally as long as the saved artifact boundary
remains compatible, or a versioned exporter is provided.
