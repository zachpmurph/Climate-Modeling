"""Tests for the 2D shallow-water (Saint-Venant) solver in
general/solvers/saint_venant_2d.py.

The scheme is a conservative finite-volume method with Rusanov (local
Lax-Friedrichs) fluxes in x and y, an inflow/open-outflow x-boundary and
reflecting solid walls in y. Covers:

1. Stability: the run completes with all-finite output (the old central-difference
   version silently overflowed to NaN while still exiting 0).
2. Mass conservation: change in stored volume == inflow - outflow + rainfall
   source + dry-floor correction.
3. Well-balanced "lake at rest": flat water, zero slope, no forcing stays exactly
   at rest.
4. y-invariance: a problem uniform in y (and with solid y-walls) stays uniform in y.
"""
import numpy as np
import pytest

from general.solvers import saint_venant_2d as sv2


def _no_rain(t):
    return 0.0


def test_runs_stably_with_all_finite_output():
    result = sv2.run_model(L=2.0, W=1.0, T_final=1.0, record_interval=0.25, left_inflow=0.02)

    for key in ("h_history", "h_initial", "h_final", "hu_final", "hv_final", "times"):
        assert np.all(np.isfinite(result[key])), f"{key} contains non-finite values"
    assert np.all(result["h_final"] >= 0.0)
    # 2D snapshots: (n_times, nx, ny)
    assert result["h_history"].ndim == 3
    assert result["h_history"].shape[0] == len(result["times"])


def test_mass_conservation(monkeypatch):
    monkeypatch.setattr(sv2, "r", lambda t: 0.0003)  # uniform rainfall rate m/min

    result = sv2.run_model(L=2.0, W=1.0, T_final=1.5, record_interval=0.5, left_inflow=0.02)

    dx = result["x"][1] - result["x"][0]
    dy = result["y"][1] - result["y"][0]
    storage_initial = float(np.sum(result["h_initial"]) * dx * dy)
    storage_final = float(np.sum(result["h_final"]) * dx * dy)
    delta = storage_final - storage_initial
    expected = (
        result["mass_inflow"]
        - result["mass_outflow"]
        + result["mass_source"]
        + result["mass_floor_correction"]
    )
    assert delta == pytest.approx(expected, rel=1e-6, abs=1e-9)
    assert result["mass_inflow"] > 0.0
    assert result["mass_source"] > 0.0


def test_lake_at_rest_stays_at_rest(monkeypatch):
    # Zero bed slope in both directions and no rainfall: a flat lake must not move.
    monkeypatch.setattr(sv2, "S0x", 0.0)
    monkeypatch.setattr(sv2, "S0y", 0.0)
    monkeypatch.setattr(sv2, "r", _no_rain)

    nx, ny = 20, 10
    h0 = np.full((nx, ny), 0.5)
    result = sv2.run_model(
        L=2.0, W=1.0, T_final=1.0, record_interval=0.5,
        h_init=h0, hu_init=np.zeros((nx, ny)), hv_init=np.zeros((nx, ny)),
        left_inflow=0.0,
    )

    assert np.allclose(result["h_final"], 0.5, atol=1e-12)
    assert np.max(np.abs(result["hu_final"])) == pytest.approx(0.0, abs=1e-12)
    assert np.max(np.abs(result["hv_final"])) == pytest.approx(0.0, abs=1e-12)


def test_problem_uniform_in_y_stays_uniform_in_y(monkeypatch):
    # Uniform in y, solid y-walls, uniform inflow -> the solution must not develop
    # any y-structure (each x-column stays constant across y).
    monkeypatch.setattr(sv2, "r", lambda t: 0.0002)

    nx, ny = 20, 10
    result = sv2.run_model(
        L=2.0, W=1.0, T_final=1.0, record_interval=0.5,
        h_init=np.full((nx, ny), 0.05),
        hu_init=np.zeros((nx, ny)), hv_init=np.zeros((nx, ny)),
        left_inflow=0.02,
    )

    h_final = result["h_final"]
    spread = np.max(np.ptp(h_final, axis=1))  # max variation across y over all x
    assert spread == pytest.approx(0.0, abs=1e-10)
    assert np.max(np.abs(result["hv_final"])) == pytest.approx(0.0, abs=1e-10)
