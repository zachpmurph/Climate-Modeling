# Real-basin model validation

**Validation** measures how well the model reproduces *real observations*, as
opposed to **verification** (`src/general/verification/`, `docs/tier3_assessment.md`),
which measures agreement with *exact solutions*. A profile that passes ingestion
validation and a solver that passes Tier-3 verification are still **not** validated
against a real river until this step is done.

The reusable harness lives in `src/rivers/validation/` and is fully offline-tested
(`tests/test_validation.py`); the live run below additionally needs network access
to the data providers.

## Committed observed-data suite

The repository includes seven reproducible, offline two-gauge events on three rivers.
Four cover the Colorado River from below Glen Canyon Dam (USGS 09379910) to
Lees Ferry (USGS 09380000): 2002-01-25, 2002-07-02, 2004-01-06, and
2004-07-01. Two independent structural tests cover the Truckee River from Reno
(USGS 10348000) to near Sparks (USGS 10348200): 2017-01-08 and 2017-02-10.
The third-river transfer test covers the Rio Grande from Alameda
(USGS 08329918) to Albuquerque (USGS 08330000): 2023-05-12 through
2023-05-14. Its event window and all hydraulic assumptions were committed and
pushed before the hydrographs were fetched or scored.
All approved USGS observations and exact query URLs are committed.

- Upstream observations cover a 12-hour pre-event spin-up plus the event
  (433 samples for Colorado; 145 for Truckee; 241 for Rio Grande).
- Downstream observations are held out and used only for scoring (97 samples
  for each 24-hour event; 193 for the 48-hour Rio Grande event).
- Geometry, roughness, grid, warm-up, and numerical settings are fixed within
  each reach and explicitly marked estimated. The Colorado calibration manifest
  remains separate and none of its fitted multipliers are transferred to Truckee
  or the Rio Grande.

Run it with:

```bash
python src/rivers/validation/run_case.py \
    real_world_rivers/validation/glen_canyon_lees_ferry.json
python src/rivers/validation/run_suite.py \
    real_world_rivers/validation/validation_suite.json
```

Observed-event configurations can now use the same stage-dependent geometry as
the production 1-D solver. Set `reach.cross_section_shape` to:

- `rectangular` and retain `upstream_width_m` / `downstream_width_m`;
- `trapezoidal` with `hydraulic_geometry` and an optional
  `bottom_width_fraction`;
- `compound` with `compound_cross_sections`; or
- `surveyed` with `surveyed_cross_sections`.

Geometry paths are resolved relative to the event JSON. Compound files use
`station_m,depth_m,top_width_m`; raw surveyed files use
`station_m,offset_m,elevation_m`. Startup depth is solved independently in each
cell from its cross-section, local roughness, slope, and initial flow before the
observed warm-up begins. This matters because applying a rectangular
normal-depth estimate to a surveyed section creates a geometry-dependent
startup transient that can falsely raise or lower NSE.

The committed events remain rectangular: no reviewed cross-sections are
currently available for those reaches. Enabling the code path is not permission
to invent geometry or choose a shape from downstream scores.

Where repeated measurements span multiple flow depths, a case may also set
`reach.stage_dependent_manning` to a reviewed
`station_m,depth_m,manning_n` CSV. Roughness is interpolated at the evolving
depth and used consistently in normal-depth startup and time stepping.
Calibration then requires `manning_scale=[1.0]`; a reviewed curve cannot be
scaled to improve its downstream score. At least two common depth levels and
two stations covering the reach are required. The current Rio Grande evidence
has only one usable field visit per gauge, so it does not identify a
roughness-versus-depth curve and this option is deliberately not applied there.

Measured event controls use optional top-level `downstream_stage_series` and
`point_flow_series` paths in the case JSON. Paths are relative to that JSON.
Stage rows contain `downstream_stage_m`; point-flow rows contain
`station_m,discharge_m3_per_min`. Both formats accept either `t_min` relative
to the scored event start or an absolute UTC `observed_at` timestamp. Every
series must cover the complete warm-up and scored window—endpoint
extrapolation is rejected.

Point discharge is signed: positive values are tributaries or returns and
negative values are diversions or withdrawals. A withdrawal removes
proportional local momentum and is capped by the water physically available in
its cell. Measured points cannot be combined with the calibrated uniform
`lateral_inflow_fraction`; otherwise the same missing flow could be counted
twice. Downstream stage must already be converted to the model's vertical datum,
not supplied as raw gage height. The cross-event optimizer enforces the same
separation: `lateral_inflow_fraction` must be fixed at zero when measured point
flows are present, and `width_scale` must be fixed at one for reviewed
stage-dependent sections. `manning_scale` must likewise remain one for
field-inferred or stage-dependent reviewed roughness.

The current hydraulically scaled, hydrostatically well-balanced, uncalibrated
July 2004 baseline is NSE `0.1188`, RMSE `4376.9 m³/min`, percent bias
`-12.53%`, and Pearson `r = 0.7559`. It uses
a 12-hour observed-upstream dynamic spin-up before the scored event. The tracked
results JSON is regenerated by the command and guarded by
`tests/test_real_validation_case.py`. These scores establish an honest baseline
for subsequent geometry changes; they do not make the estimated width, slope,
or roughness calibrated.

Across the four Colorado fixed-parameter events, NSE ranges from `-0.2718` to `0.3838`
with a median of `0.0434`; percent bias is consistently negative from `-21.48%`
to `-12.53%`, while Pearson correlation remains `0.7283–0.8117`. This combination
suggests the model often captures the timing pattern but has a shared
geometry/storage/conveyance bias. It is stronger evidence than one event, but
all four events use the same regulated reach.

The independent Truckee events use the same unfitted reach assumptions and
score NSE `0.9668` and `0.7967`, with correlation `0.9980` and `0.9198`.
Percent bias is `9.39%` and `0.70%`. This is encouraging transfer evidence,
not calibration: width and roughness remain estimates, both events have now
been inspected during development, and a third river or future event is still
needed for a pristine prospective test.

That prospective-style test is now recorded for the Rio Grande. Its untouched
first run scores NSE `-2.8482`, Pearson correlation `0.5701`, and percent bias
`+4.55%`. The small volume bias but poor NSE is diagnostically important: the
observed hydrograph spans about `1,155 m³/min`, while the model spans only about
`691 m³/min`, and the modeled maximum occurs at the end of the window rather
than near the observed peak. The result is retained rather than tuned away.
It shows that a constant estimated rectangular width, generic roughness, free
outlet, and zero unmeasured reach exchange do not transfer reliably to this
sand-bed reach.

A separate post-baseline experiment replaces only the free outlet with 241
approved downstream gage-height observations converted from the Albuquerque
NAVD88 gage datum into the model's upstream-datum coordinate system. It is
stage-conditioned discharge validation, not an independent forecast. The
correct gauge observable is the finite-volume downstream boundary flux, not
`q_history[:, -1]`, which is the last cell's internal discharge and can differ
when stage is prescribed. With the corrected observable, NSE is `-3.3217`,
RMSE is `450.4 m³/min`, percent bias is `+4.66%`, and correlation is `0.4025`.
The stage boundary therefore does not improve on the untouched baseline.

A second post-baseline experiment uses approved USGS field visits at both
gauges. It sums only channel measurements with reported width and area, derives
effective bed elevation as water-surface elevation minus hydraulic mean depth,
infers Manning roughness under an explicit rectangular wetted-perimeter
assumption, and interpolates width, bed, and roughness between the gauges. It
scores NSE `-3.4120`, RMSE `455.1 m³/min`, percent bias `+4.81%`, and
correlation `0.4566`. This is negative structural evidence: independently
derived static geometry and observed tailwater do not explain the event's
routing dynamics.

The gage datum itself is only the vertical reference used to turn gage height
into water-surface elevation. It is not a measured bed elevation. Consequently,
the difference between two gage datums must not be divided by reach length and
used as bed slope. The field experiment instead estimates the bed independently
at each visit from water surface minus measured hydraulic mean depth.

The observed spin-up scores are worse than the former constant-flow warm-up.
That is useful evidence: repeating the first event flow created an optimistic
initial state that did not represent the preceding release history.

Refresh an event from its approved USGS sources with:

```bash
python src/rivers/validation/fetch_event.py \
    real_world_rivers/validation/glen_canyon_lees_ferry_2002-01-25.json
```

The fetcher rejects non-approved observations and requires both gauges to cover
the complete configured window.

Fetch and datum-normalize a configured stage boundary with:

```bash
python src/rivers/validation/fetch_stage_control.py \
    real_world_rivers/validation/rio_grande_alameda_albuquerque_2023-05-12_stage.json
```

The stage fetcher rejects non-approved values, unsupported units, incomplete
warm-up coverage, and provider URLs that differ from the committed source.

Fetch the configured field measurements and rebuild the two-section geometry
evidence with:

```bash
python src/rivers/validation/fetch_channel_geometry.py \
    real_world_rivers/validation/rio_grande_alameda_albuquerque_2023-05-12_stage_geometry.json
```

This fetcher requires approved mean gage height for the same visit, records the
resolved provider URLs, and rejects geometry that does not imply a positive
downstream effective-bed slope.

For repeated surveys, give every upstream/downstream set the same
`geometry_snapshot` value in `field_measurement_sources`. The generated catalog
retains all dates. An event can then select only information that existed before
its start:

```json
{
  "reach": {
    "field_measurement_geometry": "channel_geometry_catalog.csv",
    "field_measurement_time_policy": "latest_not_after_event_start",
    "field_measurement_max_age_days": 365
  }
}
```

When multiple named snapshots are present, selection uses the newest complete
snapshot whose last visit predates the event; it never mixes stations from
different snapshots. For catalogs without snapshot names, selection falls back
to the newest pre-event row at each station. The result records the selected
snapshot, every chosen timestamp, and its age at event start. A station with no
pre-event measurement, or only a measurement older than the configured limit,
causes the run to fail rather than silently borrowing future or stale geometry.
The committed Rio Grande geometry experiment is explicitly retrospective: its
first available visits occurred after the event, so it remains a separate
post-baseline structural experiment and does not claim prospective selection.

Run the committed one-at-a-time structural sensitivity matrix with:

```bash
python src/rivers/validation/run_sensitivity.py \
    real_world_rivers/validation/glen_canyon_lees_ferry.json
```

Across the updated tested variants, NSE ranges from `-1.19` to `0.50`. A 20% roughness
change has the largest input-parameter effect; halving or doubling the grid
changes NSE by less than `0.006`. The less-diffusive second-order reconstruction
performs poorly on this case despite passing analytic tests, indicating strong
sensitivity to the estimated geometry and uniform-flow warm-up. This is a
model-risk finding, not permission to select the best-scoring roughness.

## Constrained multi-event calibration

Run the global optimizer with:

```bash
python src/rivers/validation/calibrate_suite.py \
    real_world_rivers/validation/calibration_suite.json
```

Only the two 2002 events select parameters. January 2004 is validation and July
2004 is a historical test. One global parameter set applies to every event; the
objective is `0.7 × mean NSE + 0.3 × mean correlation`, less penalties of
`0.1 × NSE standard deviation`, `0.05 × correlation standard deviation`, and
`0.15 × (mean NSE - worst-event NSE)`. Event-specific parameter fitting is
prohibited.

For a multi-river manifest, set `objective.balance_by` to `river` and identify
each event with `case.river`. The objective first averages NSE and correlation
within each river, then weights the river means equally. This prevents a reach
with many conveniently available events from dominating a reach with only one.
Set `cross_group_validation.enabled` to `true` for leave-one-river-out refits:
all events from one river are removed from parameter selection, the remaining
rivers are fitted, and every event from the omitted river is scored once.
This is a stronger transfer test than leave-one-event-out when events on the
same reach share geometry, controls, and measurement biases.

The selected effective parameters are Manning roughness `0.8×`, rectangular
width `1.2×`, slope `1.2×`, and uniformly distributed reach gain equal to `7.5%`
of observed upstream flow. Scores are:

- training NSE `0.725–0.762`, correlation `0.865–0.888`;
- validation NSE `0.760`, correlation `0.918`;
- historical-test NSE `0.722`, correlation `0.871`.

The tracked evidence also performs two leave-one-training-event-out refits.
Each fold omits one 2002 event from parameter selection, fits only the other,
then scores the omitted event once. Both folds independently select the same
global parameter set as the joint fit. Omitted-event NSE is `0.725` and `0.762`;
correlation is `0.865` and `0.888`. This shows stability across these two
historical training events, but it does not make either event prospective.

Roughness, width, and slope hit the tested parameter bounds. This is an explicit
identifiability warning: the variables can compensate for one another, and the
selected values must not be described as measured channel properties. The
distributed reach gain is likewise an effective missing-flow term until it is
replaced by tributary, groundwater, or local-runoff observations. All four
events were inspected during development, so a future event remains necessary
for a genuinely prospective test.

## Historical multi-river stress test

The separately committed
`real_world_rivers/validation/multi_river_calibration_suite.json` fixes its
split before fitting:

- training: two 2002 Colorado events and January 2017 Truckee;
- validation: January 2004 Colorado and February 2017 Truckee;
- test: July 2004 Colorado and the predeclared Rio Grande event.

The objective weights Colorado and Truckee equally regardless of event count.
It selects the same effective boundary values as the Colorado-only fit:
roughness `0.8×`, width `1.2×`, slope `1.2×`, and reach gain `7.5%`.
Training group means are NSE `0.744` for Colorado and `0.897` for Truckee.
However, transfer is not robust:

- the second Truckee event falls from baseline NSE `0.797` to `0.276`;
- the test-only Rio Grande event falls from `-2.848` to `-23.419`;
- fitting Truckee alone and holding out Colorado produces mean held-Colorado
  NSE `-1.060`;
- fitting Colorado while holding out Truckee gives January Truckee NSE `0.897`.

All four selected parameters hit search bounds. The result therefore rejects a
single transferable correction factor across these reaches; it does not justify
widening the search until reviewed geometry and measured controls replace the
compensating estimates. The failure is retained as evidence that higher
training NSE and correlation do not guarantee spatial transfer.

## What you need

A reach with **two gauges**:

- an **upstream** gauge whose observed discharge drives the model (the boundary
  condition), and
- a **downstream** gauge whose observed discharge is the **validation target** the
  model tries to reproduce.

Plus reviewed channel **width** along the reach so the model can represent
whole-channel storage, hydraulic radius, and discharge. A trapezoidal run also
needs defensible bankfull depth and bottom width (or a documented bottom-width
fraction); surveyed cross-section coordinates remain preferable.

Where available, also collect downstream water-surface stage and every material
tributary, return flow, and diversion between the gauges. The unified runner
accepts stage as `--downstream-stage-series` and signed spatial point flows as
`--lateral-inflow-points`. Positive point values add flow; negative values
withdraw it. These observations should replace, not accompany, a calibrated
uniform reach-gain fraction.

## Workflow

1. **Ingest** the reach with both gauges (a curated definition with two `flow`
   sections, or two `fetch-flow` calls at the upstream and downstream markers), then
   export a profile — see [real_world_ingestion.md](real_world_ingestion.md).

2. **Drive the model with the observed upstream hydrograph.** Pass the observed
   whole-channel series `Q_up(t)` [m³/min] as a time-varying `left_inflow`
   callable together with `channel_width_m` (or use `--left-inflow` and
   `--hydraulic-geometry` for a constant approximation).

3. **Extract the predicted downstream hydrograph** from the run. At a gauge on
   the model boundary, use `downstream_flux_history`; this is the signed
   finite-volume boundary discharge in whole-channel m³/min. For an interior
   station, `q_history[:, station]` is whole-channel m³/min for Saint-Venant
   when width is supplied. For the kinematic solver, evaluate its
   rectangular-section Manning discharge.

4. **Score it** against the observed downstream series:

   ```python
   from rivers.validation.compare import evaluate_series

   scores = evaluate_series(obs_times, obs_downstream_q,
                            model_times, model_downstream_q)
   print(scores["nse"], scores["rmse"], scores["percent_bias"])
   ```

   `evaluate_series` interpolates the model series onto the observation timestamps
   and returns Nash-Sutcliffe efficiency (`nse`), `rmse`, mean `bias`,
   `percent_bias`, and Pearson `pearson_r`.

## Interpreting the scores

- **NSE = 1** perfect; **> 0.75** good; **0.5–0.75** fair; **≤ 0** the model is no
  better than predicting the observed mean.
- **percent_bias** near 0 is unbiased; large magnitude means systematic over/under
  prediction (often a geometry or roughness issue).

## Caveats

- This is **validation, not calibration.** Do **not** tune Manning's *n* to maximise
  NSE and then call the model "calibrated" — that fits *n* to absorb structural and
  unit errors. Roughness is emitted in the meters-and-minutes convention
  (see [ingestion_integration_requests.md](ingestion_integration_requests.md)); a
  large bias is a signal to investigate geometry/inputs, not to re-fit *n*.
- The 1-D solver supports rectangular, trapezoidal, and tabulated compound
  stage–width sections, but the
  committed Colorado calibration remains rectangular because its bottom width
  and stage–width curve have not been independently measured or reviewed. Do
  not invent those values and
  rerun calibration: doing so would add another compensating parameter to an
  already boundary-limited fit.
- Raw offset/elevation survey sections retain asymmetric polyline wetted
  perimeter; stage–width-only sections still assume symmetric banks. Neither
  representation supports disconnected floodplain pools or horizontal
  above-bottom benches without preprocessing. Measured tributary hydrographs
  are also absent from the committed Colorado case. The calibrated
  lateral-flow fraction is an effective distributed gain, not a substitute for
  those observations.
- USGS parameter `00065` is gage height, but it becomes a model water-surface
  elevation only after applying the station's gage datum consistently with the
  model bed datum. Never pass raw gage height as absolute stage without that
  conversion.
- A clean single-event window (a rising/falling limb between two gauges with no major
  tributary in between) is the most interpretable first validation case.
