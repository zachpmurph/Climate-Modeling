# Cross-river fit investigation

## Question and test design

This investigation asks why the same uncalibrated Saint-Venant 2-D setup fits
the Potomac, Connecticut, Delaware, and Truckee events well, but underestimates
the Colorado, Snoqualmie, and Russian events. The Rio Grande is explicitly
excluded.

The comparison uses the committed downstream observations only for scoring.
It does not change roughness, width, slope, grid, or boundary conditions in
response to a score. Three additional rivers were predeclared before model
scoring:

- Eel River, California: tributary-rich flood during the same regional storm
  as the Russian River event;
- Willamette River, Oregon: tributary-rich Pacific Northwest flood; and
- Chattahoochee River below Buford Dam: regulated pulse-routing analogue for
  Glen Canyon Dam to Lees Ferry.

All new observations are approved `00060` discharge from the
[USGS Water Data API](https://api.waterdata.usgs.gov/ogcapi/v0/), and reach
lengths use the [USGS NLDI network](https://api.water.usgs.gov/docs/nldi/).
The geometries remain screening proxies, not surveyed flood terrain.

## River-level evidence

Multiple events are summarized by their median. `Down/up volume` is observed
downstream event volume divided by observed upstream event volume. A value
above one means water enters between gauges or storage is released; a value
below one can mean storage, withdrawal, regulation, or a truncated window.

| River | Events | NSE | Pearson r | Bias | Predicted/observed volume | Down/up volume |
|---|---:|---:|---:|---:|---:|---:|
| Potomac | 1 | 0.964 | 0.993 | −8.8% | 0.912 | 1.025 |
| Connecticut | 1 | 0.951 | 0.997 | −10.9% | 0.891 | 0.838 |
| Delaware | 1 | 0.757 | 0.949 | −13.3% | 0.925 | 1.003 |
| Truckee | 2 | 0.873 | 0.956 | +5.5% | 1.055 | 0.923 |
| Colorado | 4 | 0.382 | 0.801 | −11.5% | 0.886 | 1.005 |
| Snoqualmie | 1 | 0.230 | 0.798 | −32.6% | 0.675 | 1.467 |
| Russian | 1 | −2.423 | 0.620 | −76.1% | 0.239 | 3.977 |
| **Eel holdout** | 1 | −0.105 | 0.977 | −40.8% | 0.592 | 1.636 |
| **Willamette holdout** | 1 | −0.270 | 0.996 | −58.0% | 0.420 | 1.900 |
| **Chattahoochee holdout** | 1 | 0.589 | 0.839 | −9.5% | 0.901 | 0.494 |

## Why the apparently excellent cases fit

The present validation model is driven by the upstream hydrograph. It performs
best where that boundary already contains approximately the amount and shape
of water seen downstream:

- Potomac and Delaware downstream event volumes are within 2.5 percent of
  their upstream volumes.
- The stronger Truckee event differs by less than one percent in volume.
- Connecticut has less downstream than upstream event volume, so it does not
  require the model to create missing tributary water.

These are comparatively favorable routing tests. Their high scores show that
the solver can transport a supplied hydrograph; they do **not** validate the
constant width, gauge-altitude slope proxy, common roughness, or floodplain
storage. The January Truckee run illustrates the distinction: NSE is 0.964 and
Pearson r is 0.998 even though modeled routing lag is 375 minutes too short.
Hydrograph similarity can dominate an aggregate score while travel time is
physically wrong.

## Why Russian and Snoqualmie underestimate

The model supplies no intervening runoff in their committed baselines. The
observed downstream event contains 47 percent more water than upstream on the
Snoqualmie and nearly four times as much on the Russian River. A conservative
upstream-only solver cannot reproduce that volume.

The independent tests reproduce the same symptom:

- Eel downstream volume is 1.636 times upstream volume; prediction is 0.592
  of observed volume, while shape correlation remains 0.977.
- Willamette downstream volume is 1.900 times upstream volume; prediction is
  0.420 of observed volume, while shape correlation remains 0.996.

High shape agreement combined with a large negative volume bias is the key
diagnostic: the upstream storm signal is recognizable, but important
tributaries and rainfall-runoff are absent. Across the seven natural-river
tests with drainage metadata, drainage-area growth correlates with observed
volume gain at `r=0.977`, model volume deficit at `r=0.802`, and declining NSE
at `r=−0.962`. These purposive tests rank mechanisms; they do not define a
universal drainage-area correction.

The Russian and Snoqualmie shelf experiments provide a second separation.
Adding idealized lateral storage improves timing/correlation but leaves volume
ratios almost unchanged. Storage can reshape existing water; it cannot replace
omitted inflow.

## Why Colorado is different

Colorado downstream volume is approximately equal to upstream volume across
four events, so missing tributary water is not the primary explanation. The
model nevertheless passes only about 89 percent of observed downstream volume,
routes release changes 105–120 minutes too early, and correlates less well than
the favorable routing cases.

Three diagnostics narrow the cause:

1. Doubling longitudinal resolution changes predicted/observed volume from
   0.9213 to 0.9204. Grid diffusion is not the main source.
2. Extending the same July event from one to three days changes the volume
   ratio only from 0.9213 to 0.9360 and leaves routing 105 minutes too early.
   Window truncation contributes, but does not explain most of the error.
3. Adding idealized lateral shelves reduces the routing-lag error from
   −120 to −15 minutes, but lowers the volume ratio to 0.839. This shows strong
   sensitivity to storage/connectivity while demonstrating that arbitrary
   extra storage is not a valid fix.

The regulated Chattahoochee holdout has the same error class: modest volume
bias (−9.5%) but 75-minute early routing, only 0.839 shape correlation, and a
35 percent excessive amplitude. Dam-release pulses expose incorrect channel
storage, cross-section, downstream control, and initialization more strongly
than smooth flood hydrographs do.

The best-supported Colorado explanation is therefore the assumed rectangular
geometry and free-outflow/storage representation, possibly compounded by the
short observed warm-up. It is not evidence for a global roughness adjustment.

## Consequences for model development

1. Add independently observed tributary hydrographs to Russian and Snoqualmie
   first, using the conservative 2-D internal-flow interface. Never infer the
   missing series from the held-out downstream gauge.
2. Replace Colorado's assumed rectangular reach with surveyed cross-sections or
   reviewed topobathymetry and an observed downstream stage boundary.
3. Score volume, routing lag, amplitude, and correlation separately. Do not
   describe a high-NSE run as physically accurate when lag or storage fails.
4. Repeat the matched tests after those changes. The forcing fix should improve
   volume on tributary-rich rivers; the geometry/boundary fix should improve
   regulated-pulse timing without inventing or losing water.

Machine-readable evidence is in
`real_world_rivers/validation/fit_regime_analysis.results.json`.
