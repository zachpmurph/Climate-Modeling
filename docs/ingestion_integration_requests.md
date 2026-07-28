# Ingestion → Model integration requests

This file records changes the ingestion pipeline would like from the model-facing
contract (`load_profile`, `Domain`, `Scenario`, the solvers, and the unit
conventions they assume). Ingestion does **not** implement these — they are
requests for the model owner to decide. Until a decision is made, ingestion
emits data under the assumptions noted below and flags the ambiguity in
validation and metadata rather than silently picking a convention.

---

## 1. Manning's *n* unit convention — RESOLVED: meters-and-minutes

**Resolution (2026-07-28): profiles carry Manning's *n* in the model's
meters-and-minutes convention, i.e. the SI value divided by 60. No solver change
is required.**

### Why meters-and-minutes (and why the earlier "SI" note was wrong)

The whole model stack is meters-and-minutes: `g = 35316` (9.81 m/s² × 3600),
times in minutes, and `CLAUDE.md` states "meters and minutes throughout". Manning's
law `q = (1/n)·h^(5/3)·√S` yields **m²/min** precisely when *n* is the
meters-and-minutes value `n = SI/60`; with a raw SI *n* the same formula yields an
m²/s magnitude mislabelled as m²/min. The Saint-Venant solvers already default to
`n0 = MANNING_N_SECONDS/60`. So the only correct model-facing convention is
meters-and-minutes.

An earlier note here claimed the *solver* owed a "×60" factor. That was mistaken:
`tests/test_solver_consistency.py` demonstrates that `kinematic_wave` and
`saint_venant` **already converge to the same Manning normal depth** for the same
profile — the two solver families interpret *n* identically. There is nothing to
change in the solvers. The only thing that was wrong was the *data*: a briefly
adopted "SI" decision had `columbia_hanford_roughness.csv` carrying `0.028`, which
the meters-and-minutes solvers would interpret 60× off.

### What this requires (all on the ingestion side, done)

- Reviewed roughness CSVs carry the **meters-and-minutes** value (`SI/60`), with
  the SI origin documented in the row's `method`/notes. `columbia_hanford_roughness.csv`
  carries `0.000467` (= SI `0.028` / 60).
- The validation plausibility range is the meters-and-minutes band
  (`~1e-4 … 4e-3`).
- `tests/test_solver_consistency.py` is the guardrail: it fails if either solver
  ever diverges from the shared Manning convention.

### Optional future nicety

Ingestion could accept a documented SI roughness and *derive* the m-and-min value
(÷60) at import, recording the original SI value, the factor, and the result as
`derived` provenance — so reviewers can enter SI directly. Not required for
correctness; the reviewed CSV convention above already yields correct model input.

---

## 2. (Placeholder) Additional profile columns

Ingestion currently stores channel geometry (width, bankfull depth), flow
observations, and an inflow recommendation in the database and metadata sidecar,
but does **not** add them as profile columns, per the handoff rule against new
required columns. If the model later wants any of these as first-class profile
inputs (e.g. an optional `width_m` column, or a `left_inflow_flux_m2_per_min`
scenario knob), that is a contract extension to be requested here and approved
before ingestion emits it.
