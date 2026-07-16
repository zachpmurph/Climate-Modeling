import csv
import json

import numpy as np
import pytest

from general.solvers import river_kinematic_wave as rkw


def test_load_profile_csv():
    profile_path = "real_world_rivers/tools/example_river_profile.csv"

    profile = rkw.load_profile(profile_path)

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

    profile = rkw.load_profile(profile_path)

    assert np.allclose(profile.station_m, [0, 100])
    assert profile.initial_depth_m is None
    assert np.allclose(profile.rainfall_rate_m_per_min, [0.000001, 0.000002])


def test_upstream_inflow_mass_balance():
    profile = rkw.make_profile(
        station_m=np.linspace(0, 1000, 11),
        slope=np.full(11, 0.001),
        manning_n=np.full(11, 0.04),
    )

    result = rkw.run_model(
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
    profile = rkw.make_profile(
        station_m=np.linspace(0, 1000, 11),
        slope=np.full(11, 0.001),
        manning_n=np.full(11, 0.04),
    )

    result = rkw.run_model(
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


def test_profile_rainfall_adds_to_uniform_rainfall():
    profile = rkw.make_profile(
        station_m=[0, 100, 200],
        slope=[0.001, 0.001, 0.001],
        manning_n=[0.04, 0.04, 0.04],
        rainfall_rate_m_per_min=[0.0, 0.00001, 0.00002],
    )

    result = rkw.run_model(
        profile,
        t_final_min=1.0,
        left_inflow_flux=0.0,
        rainfall_rate_m_per_min=0.00001,
    )

    expected_source = np.sum((np.array([0.0, 0.00001, 0.00002]) + 0.00001) * profile.dx_m)
    assert result["mass_source"] == pytest.approx(expected_source)


def test_save_outputs(tmp_path):
    profile = rkw.make_profile(
        station_m=[0, 100, 200],
        slope=[0.001, 0.001, 0.001],
        manning_n=[0.04, 0.04, 0.04],
    )
    result = rkw.run_model(profile, t_final_min=2.0, left_inflow_flux=0.0001, rainfall_rate_m_per_min=0.000001)

    csv_path = tmp_path / "run.csv"
    summary_path = tmp_path / "summary.json"
    rkw.save_time_series_csv(result, csv_path)
    summary = rkw.save_summary_json(result, summary_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0][0] == "t_min"
    assert len(rows[0]) == 4
    assert summary_path.exists()
    assert summary["cells"] == 3
    assert "mass_source_m2" in summary
    assert "mass_balance_error_m2" in summary
