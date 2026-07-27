# Ingestion → Model integration requests

This file records changes the ingestion pipeline would like from the model-facing
contract (`load_profile`, `Domain`, `Scenario`, the solvers, and the unit
conventions they assume). Ingestion does **not** implement these — they are
requests for the model owner to decide. Until a decision is made, ingestion
emits data under the assumptions noted below and flags the ambiguity in
validation and metadata rather than silently picking a convention.

---

## 1. Manning's *n* unit convention — RESOLVED: SI

**Decision (model owner, 2026-07-24): Manning's *n* is expressed in SI units
(s·m^(−1/3)) at the model-facing boundary.**

### What this now requires

- **Ingestion side (done):** roughness is emitted and validated as SI *n*. The
  validation plausibility range is the SI range (~0.008–0.2), and the curated
  `real_world_rivers/columbia_hanford_roughness.csv` now carries the raw SI value
  `0.028` (previously the ÷60 form `0.000467`).
- **Model side (OUTSTANDING — owned by the model developer, not ingestion):** the
  solver's `q = (1/n)·h^(5/3)·√S` (and the matching `c`) has **no seconds→minutes
  factor**, so with SI *n* it produces velocity/flux in **m/s**, while the repo's
  stated convention is meters-and-minutes. To make the model consistent with the
  SI decision, multiply `q` and `c` by **60** (s→min) in
  `src/general/solvers/river_kinematic_wave.py`. Until that factor is added, model
  output built from SI *n* will be a factor of 60 too fast. Ingestion does not and
  will not make this change.

### The original problem (for the record)

The 1D kinematic-wave solver computes discharge per unit width as

```
q = (1 / n) * h^(5/3) * sqrt(S)      # river_kinematic_wave.py
```

with **no factor converting seconds to minutes**, yet the repository states its
unit convention is *meters and minutes* (see `CLAUDE.md` → Model conventions).
Manning's equation is unit-system dependent. With a Manning's *n* taken from the
standard SI tables (e.g. gravel ≈ 0.035, in units of s·m^(−1/3)), the formula
above yields a velocity in **m/s**, not m/min. To obtain m/min from the same
formula you must either multiply the result by 60 or, equivalently, divide *n*
by 60.

The repo currently contains **both conventions at once**:

| Source | Manning's *n* value | Implied convention |
|---|---|---|
| `tests/test_river_data_tools.py` | `0.035`, `0.04` | raw SI (s-based) |
| `real_world_rivers/columbia_hanford_roughness.csv` | `0.000467` | SI ÷ 60 (m-and-min), *documented in the file itself* |

Originally the repo contained **both conventions at once** — `test_river_data_tools.py`
used raw SI *n* (`0.035`, `0.04`) while `columbia_hanford_roughness.csv` used the
÷60 form (`0.000467`). The SI decision above resolves this in favour of raw SI:
the test data was already SI, and the Columbia file has been changed to its raw SI
value `0.028`.

### Follow-up ingestion can do once the model factor lands

Once the solver's ×60 factor is in place, ingestion can optionally *derive* the
SI *n* from a documented source value and record the derivation with provenance
(original value, method, result) rather than relying on a hand-entered SI number
in the reviewed CSV. Not needed for correctness; offered if useful.

---

## 2. (Placeholder) Additional profile columns

Ingestion currently stores channel geometry (width, bankfull depth), flow
observations, and an inflow recommendation in the database and metadata sidecar,
but does **not** add them as profile columns, per the handoff rule against new
required columns. If the model later wants any of these as first-class profile
inputs (e.g. an optional `width_m` column, or a `left_inflow_flux_m2_per_min`
scenario knob), that is a contract extension to be requested here and approved
before ingestion emits it.
