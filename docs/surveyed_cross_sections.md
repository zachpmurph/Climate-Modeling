# Surveyed asymmetric cross-section contract

The 1-D Saint-Venant solver can consume raw cross-section polylines so left and
right bank slopes influence wetted perimeter and Manning friction separately.
The input is reduced to common relative-depth hydraulic curves; the source path
and derived curves are retained in the run summary.

## Input format

```text
station_m,offset_m,elevation_m
0,0,103
0,2,100
0,6,100
0,10,101
0,13,104
1000,0,204
1000,3,200
1000,8,200
1000,12,202
1000,16,205
```

For each river `station_m`, `offset_m` must increase strictly across the
section. Elevations may use any vertical datum because they are shifted to
depth above that section's minimum before hydraulic reduction. At least two
distinct river stations and three points per section are required.

The model derives, at every source vertex elevation:

- total wetted top width;
- wetted bed-and-bank polyline length;
- cross-sectional area and hydrostatic pressure from the resulting
  piecewise-linear stage–width curve.

Width and perimeter curves are interpolated longitudinally onto solver cells.
This interpolation creates derived hydraulic geometry, not new observations.

## Representability requirements

The current conservative table form requires a positive-width horizontal
bottom. A V-shaped zero-width minimum is rejected because its dry-limit wave
speed and area inversion are singular.

Horizontal segments above the bottom are also rejected. Such a bench creates a
jump in top width at one exact elevation, while the current table stores a
continuous width curve. Reduce those sections to a reviewed stage–width table
or use reviewed 2-D terrain instead of silently smoothing the jump.

Disconnected pools, islands, culverts, levees, bridges, gates, and closed
conduits remain outside this 1-D representation.

Above a section's highest surveyed point, the model holds top width constant
and adds two vertical walls. Runs should remain inside the surveyed stage range
unless that extrapolation has been explicitly reviewed.

## Command

```text
python src/rivers/simulations/run_simulation.py PROFILE \
  --solver saint_venant \
  --cross-section-shape surveyed \
  --surveyed-cross-sections SECTIONS.csv \
  --t-final 60
```

The summary records the raw survey path, derived depth/width/perimeter curves,
vertical-wall extrapolation, and `bank_symmetry_assumption: false`.

Survey geometry should remain fixed across training, validation, and test
events. Changing a section separately for each flood to improve NSE or
correlation is calibration leakage unless independent surveys demonstrate the
physical change.
