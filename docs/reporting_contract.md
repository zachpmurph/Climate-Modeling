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

## Optional run summary

An adjacent `<run>_summary.json` is discovered automatically for a
`<run>_timeseries.csv` input. It may also be supplied explicitly.

The reporter recognizes these numerical diagnostic fields when present:

- `mass_inflow`
- `mass_source`
- `mass_outflow`
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

The outcomes include peak modeled depth, its time and station, maximum depth
change, and—when a threshold exists—first exceedance time, maximum exceedance
depth, and the maximum length and fraction of the modeled reach exceeding the
threshold.

These are screening outcomes from a 1-D model. They are not a 2-D inundation
boundary and must not be presented as calibrated flood-risk conclusions.

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
