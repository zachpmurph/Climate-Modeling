"""Green-Ampt soil infiltration integration tests."""

import numpy as np
import pytest

from general.solvers.contract import Scenario, UnsupportedScenario
from general.solvers.infiltration import green_ampt_step
from general.solvers.profile import (
    domain2d_from_profile,
    domain_from_profile,
    load_profile,
)
from general.solvers import saint_venant_1d, saint_venant_2d
from general.solvers.registry import dispatch


def test_green_ampt_capacity_reflects_antecedent_moisture_and_wetting():
    available = np.array([1.0, 1.0])
    initial = np.array([0.0, 0.0])
    conductivity = np.array([1e-4, 1e-4])
    suction = np.array([0.15, 0.15])
    deficit = np.array([0.05, 0.35])

    first, cumulative = green_ampt_step(
        available, initial, conductivity, suction, deficit, 10.0
    )
    second, _ = green_ampt_step(
        available, cumulative, conductivity, suction, deficit, 10.0
    )

    assert first[1] > first[0]
    assert np.all(second < first)
    assert np.all(first >= conductivity * 10.0)


def test_1d_soil_reduces_flood_depth_and_closes_mass_balance():
    cells = 6
    common = dict(
        L=6.0,
        T_final=2.0,
        record_interval=1.0,
        h_init=np.zeros(cells),
        q_init=np.zeros(cells),
        left_inflow=0.0,
        rainfall=lambda x, t: np.full_like(x, 5e-4),
        x_m=np.arange(cells, dtype=float) + 0.5,
        dx_m=np.ones(cells),
        slope=np.zeros(cells),
        manning_n=np.full(cells, 0.04 / 60.0),
        downstream_boundary="wall",
    )
    impervious = saint_venant_1d.run_model(
        **common,
        soil_ksat_m_per_min=np.zeros(cells),
        soil_suction_head_m=np.full(cells, 0.1),
        soil_moisture_deficit=np.full(cells, 0.3),
    )
    permeable = saint_venant_1d.run_model(
        **common,
        soil_ksat_m_per_min=np.full(cells, 2e-4),
        soil_suction_head_m=np.full(cells, 0.1),
        soil_moisture_deficit=np.full(cells, 0.3),
    )

    assert np.mean(permeable["h_final"]) < np.mean(impervious["h_final"])
    assert permeable["mass_infiltration"] > 0.0
    storage_change = float(
        np.sum((permeable["h_final"] - permeable["h_initial"]) * common["dx_m"])
    )
    budget = (
        permeable["mass_inflow"]
        + permeable["mass_source"]
        + permeable["mass_floor_correction"]
        - permeable["mass_outflow"]
    )
    assert storage_change == pytest.approx(budget, abs=1e-11)
    assert permeable["mass_source"] == pytest.approx(
        permeable["mass_rainfall"] - permeable["mass_infiltration"]
    )


def test_1d_infiltration_removes_proportional_momentum():
    cells = 4
    common = dict(
        L=4.0,
        T_final=0.1,
        record_interval=0.1,
        h_init=np.full(cells, 0.1),
        q_init=np.full(cells, 0.2),
        left_inflow=0.2,
        rainfall=lambda x, t: np.zeros_like(x),
        x_m=np.arange(cells, dtype=float) + 0.5,
        dx_m=np.ones(cells),
        slope=np.zeros(cells),
        manning_n=np.full(cells, 1e-12),
        downstream_boundary="outflow",
    )
    control = saint_venant_1d.run_model(**common)
    result = saint_venant_1d.run_model(
        **common,
        soil_ksat_m_per_min=np.full(cells, 0.02),
        soil_suction_head_m=np.zeros(cells),
        soil_moisture_deficit=np.zeros(cells),
    )
    interior = slice(1, -1)
    velocity_control = (
        control["q_final"][interior] / control["h_final"][interior]
    )
    velocity_final = result["q_final"][interior] / result["h_final"][interior]
    # Subsequent shallow-water steps react to the changed depth, but the
    # removal itself must not introduce a large velocity jump.
    assert velocity_final == pytest.approx(velocity_control, rel=0.02)


def test_2d_spatial_soils_change_ponding_and_close_mass_balance():
    shape = (2, 2)
    result = saint_venant_2d.run_model(
        T_final=1.0,
        record_interval=1.0,
        h_init=np.zeros(shape),
        hu_init=np.zeros(shape),
        hv_init=np.zeros(shape),
        rainfall=lambda x, y, t: np.full(shape, 1e-3),
        x_m=np.array([0.5, 1.5]),
        y_m=np.array([0.5, 1.5]),
        dx_m=np.ones(2),
        dy_m=np.ones(2),
        slope_x=np.zeros(shape),
        slope_y=np.zeros(shape),
        manning_n=np.full(shape, 0.04 / 60.0),
        soil_ksat_m_per_min=np.array([[0.0, 2e-3], [0.0, 2e-3]]),
        soil_suction_head_m=np.full(shape, 0.1),
        soil_moisture_deficit=np.full(shape, 0.3),
        boundary_x="periodic",
        boundary_y="periodic",
    )

    assert np.all(result["h_final"][:, 0] > result["h_final"][:, 1])
    storage_change = float(np.sum(result["h_final"] - result["h_initial"]))
    budget = (
        result["mass_inflow"]
        + result["mass_source"]
        + result["mass_floor_correction"]
        - result["mass_outflow"]
    )
    assert storage_change == pytest.approx(budget, abs=1e-12)
    assert result["mass_infiltration"] > 0.0


def test_profile_soil_fields_transfer_to_1d_and_2d(tmp_path):
    profile_path = tmp_path / "soil_profile.csv"
    profile_path.write_text(
        "station_m,slope,manning_n,soil_ksat_m_per_min,"
        "soil_suction_head_m,soil_moisture_deficit\n"
        "0,0.001,0.035,0.0002,0.12,0.30\n"
        "100,0.001,0.040,0.0001,0.18,0.20\n",
        encoding="utf-8",
    )
    profile = load_profile(profile_path)
    domain_1d = domain_from_profile(profile)
    domain_2d = domain2d_from_profile(profile, width_m=10.0, cross_cells=2)

    assert domain_1d.soil_ksat_m_per_min.tolist() == pytest.approx(
        [0.0002, 0.0001]
    )
    assert domain_2d.soil_suction_head_m[:, 0].tolist() == pytest.approx(
        [0.12, 0.18]
    )
    assert domain_2d.soil_moisture_deficit.shape == (2, 2)


def test_incomplete_or_invalid_soil_parameters_are_rejected(tmp_path):
    profile_path = tmp_path / "bad_soil.csv"
    profile_path.write_text(
        "station_m,slope,manning_n,soil_ksat_m_per_min\n"
        "0,0.001,0.035,0.0002\n"
        "100,0.001,0.040,0.0001\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Soil profile columns"):
        load_profile(profile_path)

    with pytest.raises(ValueError, match="between 0 and 1"):
        saint_venant_2d.run_model(
            L=1.0,
            W=1.0,
            T_final=0.0,
            soil_ksat_m_per_min=1e-4,
            soil_suction_head_m=0.1,
            soil_moisture_deficit=1.1,
        )


def test_soil_profile_is_not_silently_ignored_by_kinematic_solver(tmp_path):
    profile_path = tmp_path / "soil_profile.csv"
    profile_path.write_text(
        "station_m,slope,manning_n,soil_ksat_m_per_min,"
        "soil_suction_head_m,soil_moisture_deficit\n"
        "0,0.001,0.035,0.0002,0.12,0.30\n"
        "100,0.001,0.040,0.0001,0.18,0.20\n",
        encoding="utf-8",
    )
    domain = domain_from_profile(load_profile(profile_path))
    with pytest.raises(UnsupportedScenario, match="does not support soil"):
        dispatch(
            "kinematic_wave",
            domain,
            Scenario(t_final_min=0.0),
        )
