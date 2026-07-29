# Reviewed 2-D terrain contract

The unified runner can use an externally reviewed Cartesian elevation grid
instead of constructing a parabolic channel and planar floodplain. This path is
intended for DEM, bathymetry, or merged survey products that have already been
projected, conditioned, and sampled onto the hydraulic grid.

## CSV format

Provide one row for every Cartesian cell:

```text
x_m,y_m,dx_m,dy_m,bed_elevation_m,manning_n
0,0,10,10,101.2,0.0012
0,10,10,10,100.1,0.0007
10,0,10,10,101.0,0.0012
10,10,10,10,99.9,0.0007
```

Required columns are `x_m`, `y_m`, `dx_m`, `dy_m`, and
`bed_elevation_m`. `manning_n` is optional, but it must be populated for every
cell or none. The optional Green-Ampt columns `soil_ksat_m_per_min`,
`soil_suction_head_m`, and `soil_moisture_deficit` must likewise appear and be
populated as a complete set or not at all. When optional terrain properties are
absent, longitudinal values from the river profile are interpolated to terrain
x coordinates and repeated across y.

The loader requires:

- at least two x and two y cells;
- every x/y coordinate combination exactly once;
- finite elevations and positive cell dimensions;
- one `dx_m` value across each x row and one `dy_m` value down each y column;
- terrain x coordinates inside the river-profile reach;
- positive finite Manning values when supplied.
- finite non-negative soil conductivity and suction, with moisture deficit in
  the closed interval from 0 to 1, when supplied.

The artifact is a rectilinear finite-volume grid. It is not a point cloud or an
unconditioned DEM.

## Datum and preprocessing requirements

All bed elevations and any downstream stage boundary must use the same vertical
datum. Horizontal coordinates must be projected metres oriented so increasing
x follows the modeled reach. The runner does not:

- reproject geographic coordinates;
- fill DEM voids;
- remove bridges or vegetation;
- burn a channel through a terrain model;
- infer bathymetry below the observed water surface;
- correct levees, culverts, gates, or hydraulic structures.

Those operations require documented domain review before simulation. The model
derives diagnostic x/y slopes from the supplied bed but uses the explicit bed
elevation for well-balanced hydrostatic reconstruction.

## Initial water and roughness

At each terrain x row, the profile's initial depth is applied above the
lowest bed cell as a level water surface. Cells above that surface start dry.
Positive upstream inflow therefore requires at least one wet cell on the
upstream row.

Per-cell `manning_n` values must already use the repository's minutes-based
convention (`n_model = n_seconds / 60`). The saved 2-D field artifact retains
bed elevation, derived slopes, and the exact roughness grid used by the solver.
The summary records whether terrain roughness or interpolated profile roughness
was used.

Soil conductivity is in metres per minute. Moisture deficit represents the
event-start difference between saturated and initial volumetric water content.
The solver advances cumulative infiltration during the event but does not
simulate evapotranspiration, drainage, or moisture recovery between storms.

## Command

```text
python src/rivers/simulations/run_simulation.py PROFILE \
  --solver saint_venant_2d \
  --terrain-grid TERRAIN.csv \
  --t-final 60
```

`--terrain-grid` replaces `--width` and `--hydraulic-geometry`; mixing those
inputs is rejected so a run cannot silently combine reviewed and synthetic
terrain.

## Uncertainty ensembles

The ensemble runner accepts the same `--terrain-grid` input. Reviewed-terrain
members may vary:

- `manning_scale`;
- `terrain_elevation_offset_m`, a uniform vertical-datum offset;
- `terrain_relief_scale`, which scales elevation above each x row's lowest bed
  while retaining that row's thalweg;
- inflow, rainfall, and downstream-stage uncertainty.

Synthetic geometry parameters (`longitudinal_slope_scale`,
`channel_width_scale`, `bankfull_depth_scale`, and
`floodplain_slope_scale`) must remain exactly `1` for reviewed terrain. The
runner rejects other values rather than applying a parameter with a changed
physical meaning. Saved ensemble evidence includes bed-elevation and Manning
roughness quantiles in addition to the outcome fields.

A global datum offset has no interior hydraulic effect when every boundary is
relative or open; it matters when compared with an absolute downstream stage.
Relief scaling is only defensible when its bounds come from survey/DEM vertical
error analysis. Neither range should be selected by inspecting the held-out
flood outcome.
