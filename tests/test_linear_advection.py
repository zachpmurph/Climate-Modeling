"""Tests for the kinematic wave solver in floods/linear_advection.py.

Covers:

1. Mass conservation: the change in stored depth over the interior of the
   domain must equal source added minus outflow through the right boundary
   (cell 0 is a boundary-condition cell, not a physical control volume, so
   it's excluded from the balance -- see run_model's docstring).

2. Convergence to the known analytical steady state for constant rainfall
   on the kinematic wave equation: q_eq(x) = rate * x, so
   h_eq(x) = (rate * x * n0 / sqrt(S0)) ** (3/5).

3. The recorded (times, u_history) time series and its CSV export -- used
   by src/tools/animate_depth.py to animate depth over time.
"""
import csv

import numpy as np
import pytest

from floods import linear_advection as la


def test_mass_conservation():
    # T=40 (not la.T_final=10) deliberately runs long enough for the
    # advancing wave to reach the right boundary and produce non-trivial
    # outflow (~9% of the balance) -- at T=10 outflow is ~0 and the test
    # would pass even if outflow tracking were broken.
    result = la.run_model(la.L, 40.0)
    x = result["x"]
    dx = x[1] - x[0]

    stored_initial = np.sum(result["u_initial"][1:]) * dx
    stored_final = np.sum(result["u_final"][1:]) * dx
    delta_mass = stored_final - stored_initial

    expected_delta = result["mass_source"] - result["mass_outflow"]

    assert delta_mass == pytest.approx(expected_delta, rel=1e-3)


def test_reaches_analytical_equilibrium(monkeypatch):
    rate = 0.0002

    def constant_rainfall(x, t):
        return np.full_like(x, rate)

    monkeypatch.setattr(la, "r", constant_rainfall)

    # Estimated equilibrium time at x=L is ~49 min (h_eq(L)/rate); this is
    # 3x that, and the steady-state residual check below confirms
    # convergence rather than assuming the margin was enough.
    result = la.run_model(la.L, 150.0)
    x = result["x"]
    dx = x[1] - x[0]
    u_final = result["u_final"]

    # Confirm steady state was actually reached: at steady state the scheme
    # enforces (q[i]-q[i-1])/dx == rate for every interior cell.
    flux_final = la.q(u_final)
    residual = (flux_final[1:] - flux_final[:-1]) / dx - rate
    assert np.max(np.abs(residual)) < 1e-6

    h_eq = (rate * x * la.n0 / np.sqrt(la.S0)) ** (3 / 5)

    # Exclude the first ~1m: h_eq -> 0 as x -> 0, so relative error blows
    # up near the boundary-condition cell even though absolute error there
    # is tiny (this is expected, not a scheme bug -- see README history).
    mask = x > 1.0
    rel_error = np.abs(u_final[mask] - h_eq[mask]) / h_eq[mask]

    assert np.max(rel_error) < 0.05
    l2_rel_error = np.sqrt(np.mean((u_final[mask] - h_eq[mask]) ** 2)) / np.sqrt(np.mean(h_eq[mask] ** 2))
    assert l2_rel_error < 0.02


def test_time_series_recorded_every_minute():
    result = la.run_model(la.L, 10.0)

    assert np.allclose(result["times"], np.arange(0, 11))
    assert result["u_history"].shape == (11, result["x"].shape[0])
    # First/last recorded rows must match u_initial/u_final exactly -- they
    # come from the same snapshots, not a re-derivation.
    assert np.array_equal(result["u_history"][0], result["u_initial"])
    assert np.array_equal(result["u_history"][-1], result["u_final"])


def test_time_series_custom_interval_and_fractional_final_time():
    # record_interval doesn't evenly divide T_final here (5.5 / 0.5 = 11,
    # so it actually does divide evenly -- pick a case that doesn't, to
    # confirm T_final itself always gets appended as the last mark).
    result = la.run_model(la.L, 5.3, record_interval=0.5)

    assert result["times"][-1] == pytest.approx(5.3)
    assert result["u_history"].shape[0] == len(result["times"])
    # marks at 0, 0.5, ..., 5.0 (11 of them) plus the trailing 5.3
    assert len(result["times"]) == 12


def test_save_time_series_csv_round_trip(tmp_path):
    result = la.run_model(la.L, 3.0)
    csv_path = tmp_path / "timeseries.csv"

    la.save_time_series_csv(result, csv_path)

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    assert header[0] == "t"
    assert len(header) - 1 == len(result["x"])
    assert len(rows) == len(result["times"])

    read_x = np.array([float(v) for v in header[1:]])
    assert np.allclose(read_x, result["x"], atol=1e-5)

    read_first_row = np.array([float(v) for v in rows[0][1:]])
    assert np.allclose(read_first_row, result["u_initial"], rtol=1e-6, atol=1e-15)
