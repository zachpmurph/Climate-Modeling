# Soil infiltration model

## Why Green-Ampt

The Saint-Venant solvers use an event-scale Green-Ampt infiltration sink.
Green-Ampt gives the model a physical wetting-front state while requiring three
reviewable properties:

- saturated hydraulic conductivity, `soil_ksat_m_per_min`;
- wetting-front suction head, `soil_suction_head_m`;
- initial moisture deficit, `soil_moisture_deficit`.

This matches the parameterization documented in the US EPA SWMM hydrology
reference and National Stormwater Calculator guidance. USDA NRCS hydrologic
soil groups remain useful screening information, but their assignment depends
on restrictive-layer conductivity, depth to water table or impermeable layers,
and soil structure. A group letter is therefore not treated as a precise model
parameter.

Primary references:

- [US EPA, SWMM Reference Manual Volume I: Hydrology](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100NYRA.TXT)
- [US EPA, National Stormwater Calculator User Guide](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100RAYD.TXT)
- [US EPA, Infiltration Models](https://www.epa.gov/water-research/infiltration-models)
- [USDA NRCS, National Engineering Handbook Part 630, Chapter 7](https://directives.nrcs.usda.gov/sites/default/files2/1720460843/Chapter%207%20-%20Hydrologic%20Soil%20Groups.pdf)

## Numerical formulation

For cumulative infiltration `F`, conductivity `Ks`, suction `psi`, and moisture
deficit `delta_theta`, the potential ponded increment `dF` over one time step
is the positive root of

```text
Ks dt = dF - psi delta_theta
        log((F + dF + psi delta_theta) / (F + psi delta_theta)).
```

The implementation solves this monotone relation by bisection. This avoids the
instantaneous-capacity singularity at `F = 0`. Actual infiltration is the
smaller of potential infiltration and surface water available after the
hydraulic/rainfall update.

In 2-D, infiltrated depth is removed directly from each cell. In 1-D, the
vertical depth decrement is converted through the active cross-section curve,
so the removed volume remains exact for rectangular, trapezoidal, compound,
and surveyed sections. Longitudinal or planar momentum is reduced in the same
proportion as water volume, avoiding an artificial velocity increase.

`mass_source` remains the net source used by the existing mass-balance
equation: rainfall and lateral inflow minus infiltration. Positive
`mass_infiltration` is also retained separately for interpretation.

## Input contract

All three soil properties must be present together. They may be supplied:

- per longitudinal cell in a river profile;
- per Cartesian cell in reviewed 2-D terrain; or
- directly on `Domain` / `Domain2D`.

If reviewed terrain omits soil properties, complete longitudinal profile
properties are interpolated in x and repeated across y. If no complete soil
set is supplied, infiltration is disabled and historical results are
unchanged.

`Scenario.initial_cumulative_infiltration_m` optionally continues a known
wetting-front state. The default is zero.

## Limits

The current implementation is deliberately event scale. It does not yet model:

- drying, drainage, evapotranspiration, or recovery between storms;
- layered soil, perched water tables, macropores, or preferential flow;
- infiltration-excess return flow or groundwater exfiltration;
- land-cover interception or depression storage;
- spatial parameter inference from a hydrologic soil group alone.

Long simulations spanning separate storms should provide a justified initial
state for each event or wait for a future continuous soil-water balance.
