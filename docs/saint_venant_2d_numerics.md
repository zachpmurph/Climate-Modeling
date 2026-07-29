# 2-D Saint-Venant Numerical Method and Verification

## Scope

`src/general/solvers/saint_venant_2d.py` solves the depth-averaged shallow-water
equations on a Cartesian finite-volume grid. The verification described here
establishes correctness of the implemented numerical PDE method for the listed
benchmarks. It does not constitute calibration or validation for a real river,
floodplain, rainfall product, or risk decision.

All quantities use metres and minutes. Therefore
`g = 35316 m/min²`, and a conventional Manning coefficient expressed in seconds
must be divided by 60 before it is passed to the solver.

## Governing equations

The conserved state and fluxes are

```text
U = [h, hu, hv]ᵀ

F(U) = [hu, hu²/h + gh²/2, huv/h]ᵀ
G(U) = [hv, huv/h, hv²/h + gh²/2]ᵀ
```

with

```text
∂U/∂t + ∂F/∂x + ∂G/∂y
  = [R, -gh ∂z_b/∂x - friction_x, -gh ∂z_b/∂y - friction_y]ᵀ.
```

Here `h` is water depth, `(hu, hv)` are unit-width momenta, `z_b` is bed
elevation, and `R` is rainfall depth rate.

## Spatial flux

Each face uses the Rusanov (local Lax-Friedrichs) flux

```text
F* = 0.5 [F(U_L) + F(U_R)] - 0.5 α (U_R - U_L),
α  = max(|u_L| + √(gh_L), |u_R| + √(gh_R)),
```

with the corresponding expression in y using `v`. This is a conservative,
first-order monotone flux.

## Well-balanced bed treatment

Bed elevation is represented explicitly in `Domain2D.bed_elevation_m`.
At each face, hydrostatic reconstruction uses

```text
z*   = max(z_L, z_R)
h*_L = max(0, h_L + z_L - z*)
h*_R = max(0, h_R + z_R - z*).
```

Momentum is scaled by `h*/h`, the Rusanov flux is evaluated on the reconstructed
states, and the normal momentum flux receives the hydrostatic pressure
correction

```text
g/2 (h² - h*²)
```

for the cell on each side of the face. This balances the discrete pressure flux
and bed source for `u = v = 0` and `h + z_b = constant`, including non-flat beds.

Profile-based 2-D domains derive a longitudinal bed elevation by trapezoidally
integrating the supplied positive-downhill slope. Programmatic verification
domains supply bed elevation directly. If only two slope fields are supplied,
the compatibility path integrates x and y slopes from the first cell; an
explicit elevation field is required when the slope field is not integrable.

## Positivity and wet/dry cells

Before updating a cell, the solver computes its available volume, including
rainfall during the step, and its total outward numerical flux. If the requested
outflow would drain more than that volume, a donor-cell factor

```text
θ = min(1, available volume / requested outgoing volume)
```

scales all components of that donor's face flux. The same scaled flux is used by
both adjacent cells, preserving conservation. Hydrostatic pressure corrections
remain unscaled so lake-at-rest balance is unchanged.

Depth is allowed to be exactly zero. Velocity division is masked for
`h <= 1e-10`, and momentum is zeroed in those dry cells. A final clamp handles
floating-point roundoff only. The solver records the associated volume as
`mass_floor_correction` and raises if it exceeds `1e-12` of the volume scale.
Verification requires the correction to remain below `1e-14 m³`.

## Time integration and friction

The method is explicit forward Euler. The adaptive two-dimensional time step is

```text
Δt = CFL / max[(|u| + √gh)/Δx + (|v| + √gh)/Δy],
```

with `0 < CFL <= 0.5` and default `CFL = 0.45`. Steps are shortened to land
exactly on requested output times and the final time.

Manning friction is applied after the conservative flux update:

```text
(hu, hv)ⁿ⁺¹ = (hu, hv)* /
  [1 + Δt g n² √(u*² + v*²) / h^(4/3)].
```

This semi-implicit denominator prevents the explicit friction source from
reversing momentum in one step. Friction is skipped in dry cells.

Every updated state is checked for NaN and Inf. Failure reports the time and
first affected cell instead of returning a corrupted simulation.

## Boundary conditions

- `boundary_x="inflow_outflow"`: the left ghost mirrors x momentum around the
  prescribed unit discharge; the right ghost is zero-gradient.
- `boundary_x="inflow_stage"`: the left boundary remains prescribed inflow.
  The right ghost depth is reconstructed from a downstream water-surface
  elevation in the terrain's vertical datum. Interior velocity is retained in
  the ghost state, allowing the Riemann flux to produce either outflow or
  stage-driven backflow. Stage may be constant, time varying, or vary across y.
- `boundary_y="wall"`: normal momentum is reflected and tangential momentum is
  copied, giving a free-slip reflecting wall.
- `"periodic"` is available independently in x and y for verification problems.

Mass accounting integrates the actual limited numerical flux at physical x
boundaries and classifies signed backflow as inflow. Periodic faces are internal
and are not counted as inflow or outflow.

## Quantitative verification matrix

Run all verification cases with:

```text
python src/general/verification/verify_saint_venant_2d.py \
  --output docs/validation/saint_venant_2d_results.json
```

The committed reference result was produced with Python 3.14.4 and NumPy 2.4.4.

| Case | Metric | Acceptance | Reference result |
|---|---:|---:|---:|
| Exact periodic shear wave, 20/40/80 x cells | fitted L2 order | 0.7–1.3 | 0.88698 |
| Exact periodic shear wave, 80 x cells | L2 y-velocity error | < 0.08 | 0.07426 |
| Exact diagonal vortex wave, 24/48/96² cells | fitted L2 order | 0.7–1.3 | 0.83007 |
| Exact diagonal vortex wave, 96² cells | vector-velocity L2 error | < 0.06 | 0.05539 |
| Manufactured variable-depth pressure wave, 20/40/80 x cells | fitted L2 order | 0.7–1.3 | 1.04573 |
| Manufactured pressure wave, 80 x cells | combined L2 error | < 0.001 | 0.000520 |
| Non-flat lake at rest | max depth error | < 1e-12 m | 0 |
| Non-flat lake at rest | max momentum | < 1e-11 m²/min | 2.45e-14 |
| Partially dry non-flat lake | dry shoreline cells made wet | 0 | 0 of 320 |
| Partially dry non-flat lake | max depth error | < 1e-12 m | 0 |
| Partially dry non-flat lake | max momentum | < 1e-11 m²/min | 8.65e-17 |
| 1-D reduction | depth L2 difference | < 1e-8 m | 2.88e-9 |
| 1-D reduction | discharge L2 difference | < 1e-6 m²/min | 1.18e-7 |
| Wet radial dam break | normalized angular deviation | < 0.02 | 0.01192 |
| Periodic conservation | relative volume residual | < 1e-12 | 0 |
| Dry-bed radial dam break | relative volume residual | < 1e-12 | 1.78e-16 |
| Dry-bed radial dam break | floor correction | < 1e-14 m³ | 0 |

### Analytic solution used for convergence

On a flat periodic domain with zero friction and rainfall,

```text
h = H,
u = U,
v = A sin(k(x - Ut))
```

is an exact nonlinear solution. Continuity and x momentum are constant, while
y momentum satisfies

```text
∂(Hv)/∂t + ∂(HUv)/∂x = 0.
```

The measured L2 errors are `0.253965`, `0.140807`, and `0.074261` on 20, 40,
and 80 x cells. Pairwise orders are `0.8509` and `0.9230`, consistent with the
expected first-order Rusanov method.

The second analytic solution is genuinely two-dimensional. With
`s = 2π(x + y - 2Ut)/L`,

```text
h = H,
u = U + A/√2 sin(s),
v = U - A/√2 sin(s).
```

The perturbation is perpendicular to its wave vector, so it is divergence-free
and advects with the base velocity without changing depth. Vector-velocity L2
errors are `0.175061`, `0.102167`, and `0.055391` on 24², 48², and 96² cells,
with fitted order `0.8301`.

### Manufactured pressure-flux solution

To exercise nonconstant depth and the nonlinear pressure flux, the verification
hook applies the exact x-momentum forcing required by

```text
h = H + a sin(k(x - Ut)),
u = U,
v = 0,
source_hu = g h ∂h/∂x.
```

This is a standard method-of-manufactured-solutions check; production scenarios
leave the forcing hook unset. Combined depth and scaled-momentum L2 errors are
`0.0022164`, `0.0010606`, and `0.0005201` on 20, 40, and 80 x cells. The fitted
order is `1.0457`.

## Reproducibility

The pinned environment is in `requirements.txt`. From a fresh clone:

```text
python -m venv .venv
.venv/Scripts/python -m pip install --requirement requirements.txt
.venv/Scripts/python -m pytest tests/
.venv/Scripts/python src/general/verification/verify_saint_venant_2d.py
```

On POSIX systems, use `.venv/bin/python`. GitHub Actions repeats the complete
test suite and standalone verification matrix and uploads its JSON evidence.

## Known limits

- First-order Rusanov flux is deliberately diffusive near shocks.
- The grid is Cartesian; curvilinear coordinates and unstructured meshes are
  outside the verified scope.
- Coriolis force, infiltration, spatially varying gravity, sediment transport,
  and hydraulic structures are not modeled.
- The radial dam-break case verifies symmetry and conservation, not agreement
  with a closed-form shock solution.
- The open boundary is a zero-gradient numerical boundary; a radiation or
  stage-discharge boundary has not been verified.
- Real-basin accuracy requires independently reviewed terrain, roughness,
  forcing, boundary data, calibration, and validation.
