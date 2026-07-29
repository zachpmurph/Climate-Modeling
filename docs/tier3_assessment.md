# Tier 3 Assessment for 2D Saint-Venant Solver

## Summary

**Overall verdict: Tier 3 achieved within the documented numerical scope.**

The solver now has two exact-solution convergence studies, including a genuinely
two-dimensional diagonal wave, plus a variable-depth manufactured pressure-wave
study; all exhibit the expected first-order rate.
It preserves a non-flat lake at rest, agrees quantitatively with the 1-D solver,
maintains radial symmetry, conserves volume at machine precision, and handles
wet/dry fronts without material floor correction. The environment is pinned,
the verification matrix is executable independently, its JSON evidence is
tracked, and clean-checkout CI runs both the matrix and complete test suite.
This is numerical PDE verification; it is not real-basin calibration or field
validation.

Authoritative quantitative evidence is in
`docs/validation/saint_venant_2d_results.json`. Method definitions and acceptance
thresholds are in `docs/saint_venant_2d_numerics.md`.

## Detailed Findings

### Check 1: Lake at rest

**Status: Pass**

**Finding:** Hydrostatic reconstruction uses explicit bed elevation and balances
the discrete pressure flux with per-face bed corrections. The verification case
uses a two-dimensional Gaussian bed bump and constant `h + z_b`. After
`0.1 min`, maximum depth error is `0 m`; maximum momenta are
`2.45e-14` and `1.16e-14 m²/min`.

**Gap:** None for fully wet Cartesian lake-at-rest states. Draining wet/dry fronts
over **non-flat** bed topography are now covered directly by
`tests/test_saint_venant_2d.py::test_wet_dry_front_over_nonflat_bed_is_positive_and_conservative`
(a frictionless dome draining over a Gaussian bed bump), which holds positivity and
periodic-watertight conservation to machine precision — closing the earlier
partially-dry-topography gap. A steady partially-dry shoreline *equilibrium* over a
slope remains the one untested sub-case.

### Check 2: 1D reduction

**Status: Pass**

**Finding:** A y-uniform 80×1 2-D run is compared directly with
`saint_venant_1d.py`. L2 differences are `2.88e-9 m` for depth and
`1.18e-7 m²/min` for longitudinal discharge; transverse discharge remains zero.
The separate y-invariance test also limits cross-channel depth spread to
`1e-10`.

**Gap:** The two implementations use slightly different multidimensional CFL
constraints, so bit-for-bit identity is neither expected nor claimed.

### Check 3: 2D radial dam break

**Status: Pass**

**Finding:** A 64² wet-background circular dam break is sampled at 144 angles
over 81 radii. Maximum angular depth range normalized by the initial depth jump
is `0.01192`, below the `0.02` gate. Quarter-turn symmetry error is `0 m`.

**Gap:** This verifies rotational symmetry, positivity, and conservation, not a
closed-form shock profile. The limitation is explicit in the numerical report.

### Check 4: Convergence study

**Status: Pass**

**Finding:** The first exact solution,
`h=H, u=U, v=A sin(k(x-Ut))`, is run on 20, 40, and 80 x cells. L2 errors
decrease from `0.253965` to `0.140807` to `0.074261`; pairwise orders are
`0.8509` and `0.9230`, with fitted order `0.8870`.

A genuinely two-dimensional exact diagonal vortex/contact wave is run on 24²,
48², and 96² cells. Vector-velocity L2 errors decrease from `0.175061` to
`0.102167` to `0.055391`; pairwise orders are `0.7769` and `0.8832`, with
fitted order `0.8301`. Both fitted rates fall inside the predefined
`0.7–1.3` first-order acceptance band.

The manufactured variable-depth pressure wave exercises continuity, advective
momentum flux, and nonlinear hydrostatic pressure. Its combined L2 errors are
`0.002216`, `0.001061`, and `0.000520` on 20, 40, and 80 cells, with fitted
order `1.0457`.

**Gap:** Smooth-wave convergence does not quantify shock-location error; the
radial dam break provides the separate discontinuous-flow symmetry check.

### Check 5: Mass conservation

**Status: Pass**

**Finding:** Conservative shared-face fluxes track physical boundary flux,
rainfall input, and any roundoff correction separately. The strict periodic
case has initial and final volume `1.2 m³`, residual `0`, and floor correction
`0`. The dry radial dam break has relative volume residual `1.78e-16` and zero
floor correction. The inflow/rain/open-outflow unit test uses a normalized
`1e-12` gate.

**Gap:** None for the tested Cartesian boundary and source combinations.

### Check 6: Positivity

**Status: Pass**

**Finding:** A donor-based draining limiter prevents any cell from exporting
more volume than it contains plus rainfall added during the step. The same
limited flux is shared by adjacent cells, so positivity does not sacrifice
conservation. Exactly dry depth is supported. Dry-bed dam-break snapshots have
minimum depth `0`, relative mass residual `1.78e-16`, and no floor correction.
Additional tests cover rainfall on a partially wet domain and a wet/dry front
crossing a periodic boundary.

**Gap:** The method is first order and diffusive at wetting fronts; that is a
documented accuracy property rather than a positivity failure.

### Check 7: CFL and stability

**Status: Pass**

**Finding:** Every step uses current local gravity-wave and velocity speeds in
both dimensions:

`dt = CFL / max[(|u|+sqrt(gh))/dx + (|v|+sqrt(gh))/dy]`.

The 2-D CFL is constrained to `(0, 0.5]` and defaults to `0.45`. Output and final
times cap the step. Updated states are checked for NaN and Inf; failures include
the time and first affected cell. A test forces non-finite dynamics and requires
an immediate diagnostic.

**Gap:** The accepted CFL range is tied to the first-order explicit method and
must be reverified if the integrator or reconstruction order changes.

### Check 8: Reproducibility

**Status: Pass**

**Finding:** `requirements.txt` pins NumPy, Matplotlib, and pytest. A fresh clone
can run all tests with `python -m pytest tests/` and the complete benchmark
matrix with one additional command. `.github/workflows/tier3-verification.yml`
installs the pinned environment, runs both commands, and uploads the JSON
evidence. No verification case uses randomness. Reference environment and
results are stored in the JSON artifact.

**Gap:** Bit-for-bit floating-point identity across operating systems is not
claimed; reproducibility is defined by the quantitative acceptance gates.

### Check 9: Documentation

**Status: Pass**

**Finding:** `README.md` states the 2-D equations, method, verification status,
and commands. `docs/saint_venant_2d_numerics.md` documents conserved variables,
Rusanov fluxes, hydrostatic reconstruction, positivity limiter, CFL condition,
friction, boundaries, exact solutions, numerical results, acceptance gates,
reproduction steps, and limitations. Test names correspond directly to the
verification matrix.

**Gap:** Real-basin calibration and validation require separate case-specific
documents and are intentionally outside this numerical-method assessment.

### Check 10: Edge case handling

**Status: Pass**

**Finding:** Tests cover exactly dry cells, a sharp dry-bed dam break, a wet
radial dam break, rainfall on a partially wet domain, a periodic wet/dry front,
reflecting y walls, periodic x/y faces, inflow, and open outflow. Masked velocity
division and dry-cell momentum reset avoid division by zero. Positivity and
mass gates require no material depth-floor addition.

**Gap:** Hydraulic structures, radiation boundaries, unstructured terrain,
infiltration, and real-data failure modes remain outside the documented solver
scope.

## Priority Gaps to Address

The gaps below are next-stage improvements, not missing Tier 3 gates for the
current documented method.

1. **Extend hydraulic boundaries and field validation.**
   - The 1-D solver now supports transmissive outflow, wall, and prescribed-stage
     downstream boundaries with regression tests. The 2-D solver still uses its
     simpler boundary set and needs case-specific open-boundary justification.
   - The committed two-gauge case and one-at-a-time sensitivity matrix establish
     an uncalibrated baseline, not transferable predictive skill.

2. **Add partially dry non-flat shoreline equilibria.**
   - This would strengthen the coupling between well-balanced topography and
     wet/dry reconstruction beyond the existing separate tests.
   - Estimated effort: 2–4 focused engineering days.

3. **Add independent events and uncertainty distributions.**
   - Numerical verification and one observed event do not establish correct
     terrain, roughness, forcing, boundary data, or flood outcomes elsewhere.
   - Calibrate only on separate events, then report out-of-sample performance
     and probabilistic input uncertainty.

## Recommendation

The solver can move from numerical-method verification to carefully scoped
real-basin application. Preserve the current benchmark matrix as a regression
gate. Before interpreting flood outcomes, add case-specific input provenance,
mesh/terrain review, boundary-condition justification, calibration data,
out-of-sample validation and uncertainty reporting. Keep the committed
one-at-a-time sensitivity matrix as a regression and model-risk artifact.
