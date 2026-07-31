# Observed tributaries and validation-metric audit

## Observed tributary experiment

The Snoqualmie baseline was rerun with independently observed Raging River
(`USGS-12145500`) and Tolt River (`USGS-12148500`) hydrographs. The 674 approved
USGS observations cover both the 12-hour warm-up and the full scored event.
Confluences were located from the USGS NLDI network and expressed on the same
linear reference as the configured mainstem reach. No source magnitude or time
shift was inferred from the downstream validation gauge.

| Metric | Upstream-only baseline | With observed tributaries |
|---|---:|---:|
| Supplied internal volume | 0 m³ | 69.13 million m³ |
| Predicted/observed volume | 0.675 | 0.864 |
| Percent bias | −32.6% | −13.6% |
| NSE | 0.230 | 0.473 |
| Pearson r | 0.798 | 0.749 |
| KGE | not previously recorded | 0.689 |
| Volumetric efficiency | not previously recorded | 0.681 |

The measured tributaries recover about 58 percent of the baseline's unexplained
downstream volume. Approximately 49.8 million m³ remains unexplained by these
two gauges. That residual can include ungauged runoff, gauge-to-confluence
travel, floodplain storage, controls, geometry error, and measurement error; it
is not permission to fit an extra source from the downstream hydrograph.

The lower correlation alongside much better water balance is important. A
single correlation objective would reject a physically necessary forcing
improvement. Conversely, the mainstem routing lag is no longer directly
comparable after adding internal sources because downstream timing contains a
mixture of upstream and tributary travel paths.

Refresh and rerun explicitly with:

```bash
python src/rivers/validation/fetch_point_flows.py \
  real_world_rivers/validation/snoqualmie_snoqualmie_carnation_2009-01-07_observed_tributaries.json
python src/rivers/validation/run_case.py \
  real_world_rivers/validation/snoqualmie_snoqualmie_carnation_2009-01-07_observed_tributaries.json
```

## Complementary metrics

Validation now reports:

- normalized RMSE for scale-aware cross-event error magnitude;
- KGE, decomposing performance into correlation, variability, and mean ratios;
- volumetric efficiency, which directly penalizes absolute flow-volume error;
- percent bias, event-volume ratio, amplitude, peak timing, and routing lag; and
- squared-error skill over using the simultaneous upstream gauge unchanged.

KGE is included as a component diagnostic, not a replacement leaderboard.
[Gupta et al. (2009)](https://doi.org/10.1016/j.jhydrol.2009.08.003) introduced
the correlation/bias/variability decomposition, while
[Knoben et al. (2019)](https://doi.org/10.5194/hess-23-4323-2019) show why KGE
requires an explicit benchmark. A recent cross-site evaluation likewise finds
that NSE and KGE depend strongly on site flow variability and recommends
interpretable error metrics for spatial comparisons
([Liu, 2025](https://doi.org/10.1016/j.envsoft.2025.106665)).

## False-result audit

The audit covers 15 event configurations and explicitly excludes Rio Grande.
Four cases trigger at least one diagnostic flag:

1. **Truckee, January 2017:** NSE `0.964` and r `0.998` look nearly perfect,
   but routing lag is 375 minutes too short. This is a false physical-success
   conclusion even though overall hydrograph values align well.
2. **Eel and Willamette:** r is `0.977` and `0.996`, but biases are −40.8% and
   −58.0%. Correlation correctly recognizes storm shape and completely misses
   the water-balance failure.
3. **Colorado, July 2002:** r is `0.812`, but routing improves squared error by
   only 4.2% over passing the simultaneous upstream gauge through unchanged.
   Most apparent shape skill comes from the boundary hydrograph.

Potomac, Connecticut, and Delaware remain robust under this audit: all have KGE
above `0.83`, volumetric efficiency above `0.86`, no material lag flag, and add
more than 67 percent squared-error skill over upstream passthrough. Their high
scores are not classified as false positives, although their screening geometry
still does not constitute inundation validation.

Machine-readable evidence is in
`real_world_rivers/validation/metric_audit.results.json`. Thresholds are explicit
screening rules, not universal pass/fail criteria and never calibration targets.
