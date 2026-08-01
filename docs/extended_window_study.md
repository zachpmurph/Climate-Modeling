# Extended routing-window study

## Design

Every retained baseline and matched-holdout event was copied into a diagnostic
variant with exactly 48 scored hours appended to its original endpoint. The
original cases remain unchanged. Geometry, roughness, grid, warm-up, boundaries,
source mapping, and numerical settings are identical to each baseline; no
parameter or endpoint was selected from model skill.

The study contains 16 configurations across 11 rivers, including the
observed-tributary Snoqualmie variant. Approved USGS discharge was refreshed for
each longer window. Potomac lacks only the final 15-minute downstream sample;
the allowed endpoint tolerance is explicit and no value is filled or
extrapolated.

## Truckee result

| Event | Baseline window | Extended window | Baseline lag error | Extended lag error | Extended NSE |
|---|---:|---:|---:|---:|---:|
| Truckee January 2017 | 1 day | 3 days | −375 min | **−60 min** | 0.984 |
| Truckee February 2017 | 1 day | 3 days | −45 min | **−45 min** | 0.857 |

January's original window ended exactly at its observed peak. The extended
series moves the peak to minute 2070 and retains 37.5 hours of recession. Most
of the apparent 375-minute error was therefore a truncated-window artifact, but
the model still routes approximately one hour too quickly. February confirms a
smaller 45-minute fast-routing tendency.

## Is fast routing prevalent?

Yes, under the current anomaly-correlation lag diagnostic:

- 15 cases have comparable mainstem routing lags; the internal-source
  Snoqualmie case is excluded from this count.
- 13 of 15 extended cases route early.
- 10 of 15 are at least 60 minutes early.
- Delaware has zero lag error, while Willamette routes late.

The problem is therefore broader than Truckee, although its magnitude is
river-dependent. All four Colorado release events remain 105–120 minutes early.
Russian, Snoqualmie, Eel, Chattahoochee, and the retained Rio Grande diagnostic
also remain materially early. Connecticut, Potomac, and February Truckee are
early by 30–45 minutes rather than the material one-hour threshold.

Lag is descriptive rather than a fitted time correction. Multiple peaks,
regulation, tributaries, and changing event shape can affect the maximizing
correlation. It must be read alongside peak timing and the plotted hydrograph.

## Does the model handle longer periods?

Numerically, yes:

- all 16 runs completed with finite metrics;
- every run remained within the `1e-9` relative mass-balance gate;
- the worst relative balance residual was `4.35e-12`; and
- no run required a material non-negative-depth floor correction.

Predictively, performance is mixed rather than degrading systematically:

- NSE improves in 11 of 16 cases;
- absolute percent bias improves in 10 of 16;
- median NSE rises from 0.453 to 0.593; and
- median absolute percent bias falls from 11.46% to 9.08%.

Longer windows improve Colorado's volume accounting, January Truckee, Eel,
Snoqualmie, and Chattahoochee. They do not repair missing inflow on Russian or
Willamette, and Delaware accumulates additional negative volume bias. This is
the expected distinction: conservation can remain exact while omitted runoff,
storage, geometry, or controls accumulate physical error.

Fifteen cases retain at least 24 hours after the observed peak. The Rio Grande
variant retains only 13.75 hours and is included solely because this experiment
covers all committed baselines; it should remain separate from the cross-river
attribution work.

## Reproduction

```bash
python src/rivers/validation/extend_event_windows.py \
  real_world_rivers/validation/extended_window_study.json --fetch
python src/rivers/validation/run_suite.py \
  real_world_rivers/validation/extended_window_suite.json
python src/rivers/validation/analyze_extended_windows.py \
  real_world_rivers/validation/extended_window_study.json
```

Machine-readable comparison evidence is in
`real_world_rivers/validation/extended_window_study.results.json`.
