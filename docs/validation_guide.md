# Uncalibrated validation guide

## Scope

This branch tests whether the 2-D Saint-Venant model reproduces observed downstream
hydrographs when driven by an observed upstream hydrograph. It does not build
arbitrary river profiles and it does not tune model parameters.

The canonical runner refuses 1-D fallback. Each case is an offline, committed evidence package:

- upstream and downstream observations;
- a fixed event window and warm-up;
- hydraulic-data provenance;
- fixed geometry, roughness, boundaries, and numerical settings; and
- a saved result containing scores and diagnostic decomposition.

## Dataset selection

Use the first available defensible source:

1. surveyed cross-sections or topobathymetric lidar;
2. 1 m USGS 3DEP or reviewed NOAA/USGS topobathymetry;
3. ANADEM 30 m in South America;
4. GEDTM30 elsewhere as screening terrain.

Record survey/DEM date, horizontal and vertical datum, resolution, processing,
hydrologic conditioning, and known gaps. Never infer submerged bed shape from
a bare-earth land DEM. Do not use gauge datum as streambed elevation.

The registry is in `src/rivers/validation/datasets.py`. A case with an unknown
dataset ID, missing `calibration: none`, or missing `validation_2d` settings
fails before a solver runs.

## Cross-region test design

Use regions that force different source products, but interpret them in two
layers:

1. **Across-region transfer:** reveals where overall performance changes under
   different rivers, forcing regimes, controls, and datasets. It cannot assign
   the difference to terrain alone.
2. **Within-event dataset substitution:** runs independently prepared datasets
   on the same event while holding all other inputs fixed. This is the useful
   terrain-error isolation test.

For each new dataset tier, predeclare the event and preparation rules before
viewing downstream scores. Keep failed variants.

## No-calibration rule

Do not select width, slope, roughness, lateral gain, solver order, or a terrain
variant using NSE or correlation. Downstream discharge is a validation target,
not a fitting target. The validation runner rejects calibration scale fields.

One-at-a-time sensitivity is allowed only to answer questions such as “could
grid diffusion plausibly explain the attenuation?” Its output is diagnostic
evidence, not permission to adopt the best result.

## Reading the diagnostics

The case result includes:

- NSE, RMSE, mean and percent bias, and Pearson correlation;
- observed/predicted integrated volume ratio;
- hydrograph amplitude ratio;
- observed/predicted peak time and lag; and
- symptom-based likely source classes.

Interpret combinations:

- **Large volume error:** first inspect upstream forcing, tributaries,
  diversions, rainfall/runoff, and storage/geometry.
- **Volume near one but poor NSE:** net water is plausible; timing,
  attenuation, boundary controls, or model structure are more likely.
- **Low amplitude ratio:** inspect excessive storage, roughness/geometry,
  floodplain connectivity, and first-order numerical diffusion.
- **Peak lag:** inspect wave speed, bed/cross-section geometry, roughness, and
  boundary timing.
- **Low correlation after volume/timing checks:** investigate missing controls,
  terrain connectivity, or structural river processes.

No one metric uniquely identifies the cause. Use conservation and manufactured
solution tests to separate numerical defects from field-input/model-structure
errors.

## Commands

Run one case:

```bash
python src/rivers/validation/run_case.py \
  real_world_rivers/validation/glen_canyon_lees_ferry.json
```

Run all fixed observed cases:

```bash
python src/rivers/validation/run_suite.py \
  real_world_rivers/validation/validation_suite.json
```

Rebuild the five-river descriptive error assessment from committed case
evidence:

```bash
python src/rivers/validation/analyze_expanded_events.py \
  real_world_rivers/validation/expanded_river_error_assessment.json
```

The expanded set adds Delaware, Connecticut, Potomac, Russian, and Snoqualmie
floods. These were selected before scoring to span a roughly 6% to 270%
increase in contributing drainage area between gauges. That is a forcing
stress test: because the current model supplies only upstream flow, worsening
volume error with drainage-area growth elevates omitted tributary and
rainfall-runoff forcing as a hypothesis. It does not prove that drainage area
alone predicts event runoff.

Run the cross-region diagnostic subset:

```bash
python src/rivers/validation/run_diagnostic_suite.py \
  real_world_rivers/validation/diagnostic_suite.json
```

Run a 2-D screening representation and rebuild the error-source assessment:

```bash
python src/rivers/validation/run_case_2d.py CASE.json \
  --representation ribbon --x-cells 31 --y-cells 1
python src/rivers/validation/assess_error_sources.py \
  real_world_rivers/validation/error_source_experiments.json
```

The 2-D runner scores the finite-volume downstream boundary flux. Do not score
the last cell's internal momentum when stage is prescribed. A ribbon or
idealized shelf is a controlled structural experiment, not a geographic flood
map. Only continuous, datum-reviewed topobathymetry can support a terrain-based
inundation validation.

Do not interpret volume over a truncated flood window as permanent gain or
loss. Water may still be stored in the reach at the end of the window, and
reservoirs, hydropower operations, diversions, tributaries, and gauge
uncertainty can all contribute. Extend through recession or model those
controls before assigning the residual to solver error.

### Reach storage, ribbon, and shelf

**Reach storage** is water temporarily held inside the modeled river section
between the two gauges. It includes main-channel water and, when connected,
water on banks, side channels, or floodplain areas. During a rising hydrograph,
storage can delay and flatten downstream flow; some water may return later. It
is not necessarily a permanent loss or diversion.

A **ribbon** is a straight, constant-width 2-D strip with wall boundaries along
its sides. It uses the 2-D equations and boundary fluxes but intentionally
suppresses realistic lateral storage. It is a routing control experiment, not
a flood map.

A **shelf** terrain adds raised cells beside the central channel. When the
assigned bank elevation is overtopped, water can spread sideways, remain
temporarily, and possibly drain back. Ribbon-versus-shelf differences measure
sensitivity to lateral storage/connectivity, but become geographic evidence
only when elevations come from reviewed terrain.

Explicitly refresh a committed USGS fixture only when source review requires it:

```bash
python src/rivers/validation/fetch_event.py CASE.json
python src/rivers/validation/fetch_stage_control.py CASE.json
python src/rivers/validation/fetch_channel_geometry.py CASE.json
```

The refresh utilities have no database or generic ingestion/export path.
