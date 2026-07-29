"""Tests for the 1-D river kinematic wave solver in general/solvers/linear_advection.py.

linear_advection.py is the project's kinematic wave model: it runs directly on a
RiverProfile (per-cell slope and Manning n), with optional upstream inflow and
rainfall, and is the ``kinematic_wave`` solver in the simulation harness.

Covers:

1. Loading profiles (CSV/JSON) through the documented interface.
2. Mass conservation with upstream inflow and with a timed rainfall source.
3. Convergence to the analytical kinematic-wave equilibrium for constant
   rainfall with a zero-inflow left boundary: the discrete steady state gives
   q_eq(cell i) = rate * dx * (i + 1), hence h_eq = (q_eq * n / sqrt(S))**(3/5).
4. The recorded (times, depth_history) series and its CSV/JSON export.
"""
import csv
import json

import numpy as np
import pytest

from general.solvers import linear_advection as la


def _uniform_profile(n_cells=11, length_m=1000.0, slope=0.001, manning_n=0.04):
    return la.make_profile(
        station_m=np.linspace(0.0, length_m, n_cells),
        slope=np.full(n_cells, slope),
        manning_n=np.full(n_cells, manning_n),
    )


# ── profile loading ────────────────────────────────────────────────────────
def test_load_profile_csv():
    profile = la.load_profile("real_world_rivers/tools/example_river_profile.csv")

    assert np.allclose(profile.station_m, [0, 1000, 2000, 3000, 4000])
    assert np.all(profile.dx_m > 0)
    assert np.allclose(profile.initial_depth_m, [0.04, 0.04, 0.04, 0.04, 0.04])
    assert profile.rainfall_rate_m_per_min is not None
    assert profile.labels[0] == "upstream"


def test_load_profile_json(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "segments": [
                    {"station_m": 0, "slope": 0.001, "manning_n": 0.035, "rainfall_rate_m_per_min": 0.000001},
                    {"station_m": 100, "slope": 0.0012, "manning_n": 0.04, "rainfall_rate_m_per_min": 0.000002},
                ]
            }
        ),
        encoding="utf-8",
    )

    profile = la.load_profile(profile_path)

    assert np.allclose(profile.station_m, [0, 100])
    assert profile.initial_depth_m is None
    assert np.allclose(profile.rainfall_rate_m_per_min, [0.000001, 0.000002])


# ── mass conservation ──────────────────────────────────────────────────────
def test_upstream_inflow_mass_balance():
    profile = _uniform_profile()

    result = la.run_model(
        profile,
        t_final_min=30.0,
        left_inflow_flux=0.0006,
        record_interval_min=5.0,
        base_depth_m=0.03,
        wave_amplitude_m=0.01,
        wave_center_m=200.0,
        wave_width_m=75.0,
    )

    storage_initial = np.sum(result["depth_initial"] * result["dx_m"])
    storage_final = np.sum(result["depth_final"] * result["dx_m"])
    delta_storage = storage_final - storage_initial
    expected_delta = result["mass_inflow"] + result["mass_source"] - result["mass_outflow"]

    assert delta_storage == pytest.approx(expected_delta, rel=1e-3, abs=1e-8)
    assert result["mass_source"] == pytest.approx(0.0)
    assert result["times"][-1] == pytest.approx(30.0)
    assert result["depth_history"].shape[0] == len(result["times"])


def test_rainfall_source_mass_balance():
    profile = _uniform_profile()

    result = la.run_model(
        profile,
        t_final_min=20.0,
        left_inflow_flux=0.0002,
        record_interval_min=2.0,
        base_depth_m=0.03,
        rainfall_rate_m_per_min=0.00001,
        rainfall_start_min=5.0,
        rainfall_end_min=15.0,
    )

    storage_initial = np.sum(result["depth_initial"] * result["dx_m"])
    storage_final = np.sum(result["depth_final"] * result["dx_m"])
    delta_storage = storage_final - storage_initial
    expected_delta = result["mass_inflow"] + result["mass_source"] - result["mass_outflow"]
    expected_source = 0.00001 * np.sum(profile.dx_m) * 10.0

    assert result["mass_source"] == pytest.approx(expected_source, rel=1e-10)
    assert delta_storage == pytest.approx(expected_delta, rel=1e-3, abs=1e-8)


def test_rectangular_width_uses_hydraulic_radius_and_conserves_volume():
    profile = _uniform_profile(n_cells=21, length_m=100.0)
    width = np.linspace(10.0, 20.0, len(profile.station_m))
    rainfall = 0.00001
    result = la.run_model(
        profile,
        t_final_min=1.0,
        left_inflow_flux=0.0,
        base_depth_m=0.2,
        rainfall_rate_m_per_min=rainfall,
        channel_width_m=width,
    )

    storage_delta = np.sum(
        (result["depth_final"] - result["depth_initial"])
        * width
        * profile.dx_m
    )
    expected_delta = (
        result["mass_inflow"] + result["mass_source"] - result["mass_outflow"]
    )
    assert storage_delta == pytest.approx(expected_delta, rel=1e-9, abs=1e-10)
    assert result["mass_source"] == pytest.approx(
        rainfall * np.sum(width * profile.dx_m)
    )
    assert result["uses_cross_section"] is True


def test_profile_rainfall_adds_to_uniform_rainfall():
    profile = la.make_profile(
        station_m=[0, 100, 200],
        slope=[0.001, 0.001, 0.001],
        manning_n=[0.04, 0.04, 0.04],
        rainfall_rate_m_per_min=[0.0, 0.00001, 0.00002],
    )

    result = la.run_model(profile, t_final_min=1.0, left_inflow_flux=0.0, rainfall_rate_m_per_min=0.00001)

    expected_source = np.sum((np.array([0.0, 0.00001, 0.00002]) + 0.00001) * profile.dx_m)
    assert result["mass_source"] == pytest.approx(expected_source)


def test_callable_rainfall_is_evaluated_during_time_stepping():
    profile = _uniform_profile(
        n_cells=101,
        length_m=10.0,
        slope=0.05,
        manning_n=0.05,
    )
    evaluation_times = []

    def rainfall(x, t):
        evaluation_times.append(t)
        return np.full_like(x, 0.00001 if t < 0.01 else 0.00002)

    la.run_model(
        profile,
        t_final_min=0.02,
        left_inflow_flux=0.0,
        base_depth_m=0.5,
        rainfall=rainfall,
    )

    assert len(evaluation_times) > 1
    assert min(evaluation_times) == 0.0
    assert max(evaluation_times) > 0.0


# ── analytical equilibrium ─────────────────────────────────────────────────
def test_reaches_analytical_equilibrium():
    # Small uniform reach, zero upstream inflow, constant rainfall. The discrete
    # kinematic-wave steady state is q_eq(i) = rate * dx * (i + 1), so
    # h_eq(i) = (q_eq(i) * n / sqrt(S)) ** (3/5). No cell is pinned to zero, so
    # (unlike the historical overland-flow test) every cell has a finite h_eq.
    rate = 0.0002
    slope, manning_n = 0.05, 0.05
    profile = _uniform_profile(n_cells=101, length_m=10.0, slope=slope, manning_n=manning_n)
    dx = float(profile.dx_m[0])

    result = la.run_model(
        profile,
        t_final_min=150.0,
        left_inflow_flux=0.0,
        rainfall_rate_m_per_min=rate,
    )
    depth_final = result["depth_final"]

    # Steady state actually reached: interior discrete residual ~ rate.
    flux_final = la.q(depth_final, profile.slope, profile.manning_n)
    residual = (flux_final[1:] - flux_final[:-1]) / dx - rate
    assert np.max(np.abs(residual)) < 1e-6

    q_eq = rate * dx * (np.arange(len(depth_final)) + 1)
    h_eq = (q_eq * manning_n / np.sqrt(slope)) ** (3.0 / 5.0)
    rel_error = np.abs(depth_final - h_eq) / h_eq
    assert np.max(rel_error) < 1e-3


# ── time-series recording + export ─────────────────────────────────────────
def test_time_series_recorded_on_interval_grid():
    profile = _uniform_profile()
    result = la.run_model(profile, t_final_min=10.0, left_inflow_flux=0.0002, record_interval_min=1.0)

    assert np.allclose(result["times"], np.arange(0, 11))
    assert result["depth_history"].shape == (11, len(profile.station_m))
    assert np.array_equal(result["depth_history"][0], result["depth_initial"])
    assert np.array_equal(result["depth_history"][-1], result["depth_final"])


def test_time_series_fractional_final_time_appended():
    profile = _uniform_profile()
    result = la.run_model(profile, t_final_min=5.3, left_inflow_flux=0.0002, record_interval_min=0.5)

    assert result["times"][-1] == pytest.approx(5.3)
    assert result["depth_history"].shape[0] == len(result["times"])
    assert len(result["times"]) == 12  # 0, 0.5, ..., 5.0 (11) + trailing 5.3


def test_save_outputs(tmp_path):
    profile = la.make_profile(station_m=[0, 100, 200], slope=[0.001, 0.001, 0.001], manning_n=[0.04, 0.04, 0.04])
    result = la.run_model(profile, t_final_min=2.0, left_inflow_flux=0.0001, rainfall_rate_m_per_min=0.000001)

    csv_path = tmp_path / "run.csv"
    summary_path = tmp_path / "summary.json"
    la.save_time_series_csv(result, csv_path)
    summary = la.save_summary_json(result, summary_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0][0] == "t_min"
    assert len(rows[0]) == 4
    assert summary_path.exists()
    assert summary["cells"] == 3
    assert "mass_source_m2" in summary
    assert "mass_balance_error_m2" in summary
