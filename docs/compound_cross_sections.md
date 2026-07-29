# Compound cross-section contract

The 1-D Saint-Venant solver can use a reviewed top-width curve at each river
station instead of reducing the channel to one rectangle or trapezoid. This
allows storage, pressure, rainfall collection, wetted perimeter, and conveyance
to change when water reaches a bench or floodplain.

## Input format

Use a CSV with one row per surveyed station and relative water-depth level:

```text
station_m,depth_m,top_width_m
0,0,8
0,1,14
0,2,32
1000,0,10
1000,1,17
1000,2,40
```

Requirements:

- At least two distinct surveyed river stations are required.
- Every station uses the same depth levels.
- Depth starts at zero and increases strictly.
- Top width is positive and non-decreasing with depth.
- Depth is relative to the local model bed, not an absolute elevation datum.
- Values must come from a reviewed source. Interpolated solver cells remain
  derived values, not new observations.

The runner interpolates top width longitudinally at each reviewed depth level.
Within a depth interval, width varies linearly. The solver integrates that
piecewise-linear curve exactly for cross-sectional area and hydrostatic
pressure.

## Explicit assumptions

Top width alone does not uniquely determine the bank shape. Wetted perimeter is
therefore calculated by splitting each width increase equally between the two
banks. Above the highest reviewed depth the solver uses vertical walls; it does
not extrapolate floodplain widening. Both assumptions are recorded in the run
summary.

This representation is appropriate for sensitivity analysis and for survey
products already reduced to stage–width curves. It is not equivalent to
retaining raw asymmetric offset/elevation coordinates, disconnected overbank
pools, culverts, levees, or structures. Use 2-D reviewed terrain when those
features control inundation.

## Run command

```text
python src/rivers/simulations/run_simulation.py PROFILE \
  --solver saint_venant \
  --cross-section-shape compound \
  --compound-cross-sections SECTIONS.csv \
  --t-final 60
```

The output summary records the source path, the interpolated depth/width table,
the vertical-wall behavior above the reviewed range, and the symmetric-bank
assumption. The discharge output is whole-channel flow in m³/min.

For multi-event calibration, keep the reviewed section curves fixed across
training, validation, and test events. Do not change bench widths event by event
to improve NSE or correlation; that would turn geometry into an untracked
calibration parameter.
