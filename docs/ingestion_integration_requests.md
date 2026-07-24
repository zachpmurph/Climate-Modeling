# Ingestion → Model integration requests

This file records changes the ingestion pipeline would like from the model-facing
contract (`load_profile`, `Domain`, `Scenario`, the solvers, and the unit
conventions they assume). Ingestion does **not** implement these — they are
requests for the model owner to decide. Until a decision is made, ingestion
emits data under the assumptions noted below and flags the ambiguity in
validation and metadata rather than silently picking a convention.

---

## 1. Manning's *n* unit convention is unresolved and self-contradictory in the repo

**Severity: blocking for calibrated results (not for running the pipeline).**

### The problem

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

The Columbia roughness file explicitly annotates its value as
*"SI n=0.028 s/m^(1/3) converted to min/m^(1/3) (÷60) for meters-and-minutes
unit system."* So the curated real-world data assumes the ÷60 convention, while
the solver's own test data assumes raw SI. These cannot both be right for the
same `q` formula.

### What ingestion does in the meantime

- Ingestion **stores and emits Manning's *n* exactly as provided** in the
  reviewed roughness input — it does not convert. Whatever convention the input
  file uses is preserved verbatim, and the `method`/`notes` fields carry the
  provenance.
- The roughness **plausibility warning** in `validation.py` uses a range
  (`1e-4 … 0.2`) deliberately widened to span *both* candidate conventions, so a
  legitimately-converted value (Columbia's `0.000467`) is not flagged as
  suspicious and a raw-SI value (`0.035`) is not either. This is a stopgap; a
  single convention would let the range tighten and actually catch errors.

### Requested decision

Declare **one** convention for Manning's *n* at the model-facing boundary, and
either:

1. **Adopt m-and-min *n* (SI ÷ 60)** as the contract. Then the solver is already
   consistent, `test_river_data_tools.py` should use ÷60 values, and ingestion
   can assume/validate the ÷60 range. OR
2. **Adopt raw SI *n*** and add the missing `×60` (seconds→minutes) factor to the
   solver's `q`/`c`, so the model produces true m/min. Then ingestion assumes/
   validates the SI range and the Columbia file should carry raw SI values.

Once decided, ingestion will tighten the roughness validation range and, if
option 1 is chosen, can optionally perform and record the SI→(m·min) conversion
as a `derived` value with full provenance (original SI value, factor, result)
rather than requiring pre-converted inputs.

---

## 2. (Placeholder) Additional profile columns

Ingestion currently stores channel geometry (width, bankfull depth), flow
observations, and an inflow recommendation in the database and metadata sidecar,
but does **not** add them as profile columns, per the handoff rule against new
required columns. If the model later wants any of these as first-class profile
inputs (e.g. an optional `width_m` column, or a `left_inflow_flux_m2_per_min`
scenario knob), that is a contract extension to be requested here and approved
before ingestion emits it.
