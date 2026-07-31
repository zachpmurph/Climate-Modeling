# Model shortfall roadmap

This branch improves the 2-D model in evidence-led stages. It does not tune
parameters against held-out downstream observations.

## First fix: observed internal flows

The existing Snoqualmie baseline underpredicts downstream event volume by
approximately 32.5 percent and routes the peak approximately 465 minutes too
quickly relative to the observed upstream-to-downstream lag. Its drainage area
grows by 60.8 percent between gauges, but the baseline supplies no intervening
tributary flow. That makes omitted internal flow a high-confidence volume-error
source, while the large timing error also implicates geometry and storage.

The 2-D solver now accepts signed, spatially mapped internal hydrographs:

- positive values add water without inventing momentum;
- negative values withdraw water and proportional local momentum;
- withdrawals cannot remove more water than is available in a time step;
- rainfall and internal-flow mass are reported separately; and
- forcing breakpoints constrain the adaptive time step.

This follows the physical distinction made by the
[USACE HEC-RAS 2-D internal-boundary guidance](https://www.hec.usace.army.mil/confluence/rasdocs/r2dum/6.6/boundary-and-initial-conditions-for-2d-flow-areas/internal-boundary-conditions):
internal flow hydrographs are localized sources or sinks, not precipitation.

## Next controlled experiment

Use approved USGS discharge series for the Raging River
(`USGS-12145500`) and Tolt River (`USGS-12148500`) as pre-observed tributary
inputs to a copy of the Snoqualmie case. Keep the original case unchanged as
the no-tributary baseline. Before scoring, replace the current reach-average
length proxy with a consistent routed centerline so tributary confluences and
model cells share the same linear reference.

Interpret the comparison in components:

1. Volume improvement measures how much error came from omitted tributaries.
2. Remaining peak-lag error tests the geometry/storage and wave-speed hypothesis.
3. Remaining attenuation error points to cross-section, floodplain-storage,
   roughness, or numerical-diffusion error.

Do not infer unmeasured residual lateral flow from the downstream gauge and feed
it back into the run. That would leak the validation target into the forcing.

## Following priorities

1. Replace one-cell ribbons and gauge-datum slopes with DEM-derived bed and
   floodplain geometry, preserving provenance and resolution.
2. Represent channel/floodplain exchange and reach storage explicitly.
3. Add spatial roughness classes from defensible land-cover data.
4. Add infiltration and antecedent soil-moisture forcing where rainfall-runoff
   is modeled, rather than treating river-gauge inflow as soil input.
5. Only after structural tests pass, define a training-event calibration set
   and retain separate rivers/events for untouched validation.
