"""Tests for the unified run_simulation dispatch harness."""

import json

import pytest
import numpy as np

from general.solvers.contract import Domain, Domain2D, Scenario, UnsupportedScenario
from rivers.simulations import run_simulation
from rivers.simulations.registry import dispatch, SOLVERS
from rivers.simulations.ingest_to_simulate import profile_to_domain_scenario


PROFILE_PATH = "real_world_rivers/tools/example_river_profile.csv"
GEOMETRY_PATH = "real_world_rivers/tools/example_geometry.csv"


def _make_scenario(**kwargs):
    defaults = dict(t_final_min=2.0, record_interval_min=1.0, cfl=0.5)
    defaults.update(kwargs)
    return Scenario(**defaults)


def _load_domain():
    from general.solvers.profile import domain_from_profile, load_profile
    return domain_from_profile(load_profile(PROFILE_PATH))


# ── registry ──────────────────────────────────────────────────────────────
def test_registry_contains_expected_solvers():
    assert "kinematic_wave" in SOLVERS
    assert "saint_venant" in SOLVERS
    assert "saint_venant_2d" in SOLVERS


def test_unknown_solver_raises():
    with pytest.raises(KeyError, match="Unknown solver"):
        dispatch("no_such_solver", _load_domain(), _make_scenario())


# ── dispatch by name ──────────────────────────────────────────────────────
def test_dispatch_kinematic_wave_with_inflow():
    domain = _load_domain()
    scenario = _make_scenario(left_inflow=0.0006, t_final_min=2.0)
    result = dispatch("kinematic_wave", domain, scenario)
    assert result.depth_history.shape[0] >= 2
    assert result.depth_history.shape[1] == len(domain.x_m)
    assert result.mass_inflow > 0


def test_dispatch_kinematic_wave():
    domain = _load_domain()
    scenario = _make_scenario(t_final_min=2.0)
    result = dispatch("kinematic_wave", domain, scenario)
    assert result.depth_history.shape[0] >= 2
    # kinematic_wave runs on the profile's own per-cell grid (domain.x_m).
    assert result.depth_history.shape[1] == len(domain.x_m)
    assert result.mass_inflow == 0.0  # no upstream inflow in this scenario


# ── same profile, two solvers, both produce valid outputs ─────────────────
def test_two_solvers_on_same_profile():
    domain = _load_domain()

    # kinematic_wave with inflow
    r1 = dispatch("kinematic_wave", domain, _make_scenario(t_final_min=3.0, left_inflow=0.0006))
    assert np.all(r1.depth_history >= 0)

    # saint_venant on the same profile
    r2 = dispatch("saint_venant", domain, _make_scenario(t_final_min=3.0))
    assert np.all(r2.depth_history >= 0)


# ── UnsupportedScenario ───────────────────────────────────────────────────
def test_unsupported_initial_discharge_on_kinematic_wave():
    domain = _load_domain()
    scenario = _make_scenario(initial_discharge=0.001)
    with pytest.raises(UnsupportedScenario, match="initial_discharge"):
        dispatch("kinematic_wave", domain, scenario)


# ── SimulationResult shape invariants ─────────────────────────────────────
def test_simulation_result_shapes_kinematic_wave():
    domain = _load_domain()
    scenario = _make_scenario(t_final_min=3.0, left_inflow=0.0006)
    result = dispatch("kinematic_wave", domain, scenario)
    n_times = len(result.times)
    n_cells = len(domain.x_m)
    assert result.depth_history.shape == (n_times, n_cells)
    assert result.depth_initial.shape == (n_cells,)
    assert result.depth_final.shape == (n_cells,)


def test_saint_venant_extra_has_discharge():
    domain = _load_domain()
    scenario = _make_scenario(t_final_min=2.0, left_inflow=0.0006)
    result = dispatch("saint_venant", domain, scenario)
    assert "discharge_history" in result.extra
    assert "discharge_initial" in result.extra
    assert "discharge_final" in result.extra


def test_simulation_result_shapes_saint_venant():
    domain = _load_domain()
    scenario = _make_scenario(t_final_min=2.0)
    result = dispatch("saint_venant", domain, scenario)
    n_times = len(result.times)
    n_cells = len(domain.x_m)
    assert result.depth_history.shape == (n_times, n_cells)
    assert result.depth_initial.shape == (n_cells,)
    assert result.depth_final.shape == (n_cells,)
    assert result.domain is domain


def test_profile_optional_fields_transfer_to_scenario():
    from general.solvers.profile import load_profile

    profile = load_profile(PROFILE_PATH)
    domain, scenario = profile_to_domain_scenario(
        PROFILE_PATH,
        t_final_min=2.0,
        rainfall_rate_m_per_min=0.000003,
    )

    assert np.array_equal(scenario.initial_depth_m, profile.initial_depth_m)
    assert scenario.labels == profile.labels
    assert np.allclose(
        scenario.rainfall(domain.x_m, 0.0),
        profile.rainfall_rate_m_per_min + 0.000003,
    )


def test_both_solvers_use_spatial_rainfall_function():
    domain = Domain(
        x_m=np.array([0.0, 100.0, 250.0]),
        dx_m=np.array([50.0, 125.0, 100.0]),
        slope=np.array([0.001, 0.0012, 0.0008]),
        manning_n=np.array([0.0005, 0.0006, 0.0007]),
    )
    rainfall_rate = np.array([0.0, 0.00001, 0.00002])
    scenario = _make_scenario(
        t_final_min=0.1,
        initial_depth_m=np.full(3, 0.2),
        rainfall=lambda x, t: rainfall_rate,
    )
    expected_source = float(np.sum(rainfall_rate * domain.dx_m) * 0.1)

    for solver_name in ("kinematic_wave", "saint_venant"):
        result = dispatch(solver_name, domain, scenario)
        assert result.mass_source == pytest.approx(expected_source)


def test_2d_solver_uses_extruded_profile_and_shared_scenario():
    from general.solvers.profile import domain2d_from_profile, load_profile

    profile = load_profile(PROFILE_PATH)
    domain = domain2d_from_profile(profile, width_m=20.0, cross_cells=4)
    scenario = _make_scenario(
        t_final_min=0.1,
        initial_depth_m=profile.initial_depth_m,
        rainfall=lambda x, t: profile.rainfall_rate_m_per_min,
    )
    result = dispatch("saint_venant_2d", domain, scenario)

    assert isinstance(result.domain, Domain2D)
    assert result.depth_history.shape == (2, 5, 4)
    assert np.allclose(domain.slope_x[:, 0], profile.slope)
    assert np.allclose(domain.manning_n[:, -1], profile.manning_n)
    assert domain.bed_elevation_m.shape == (5, 4)
    assert np.allclose(domain.bed_elevation_m, domain.bed_elevation_m[:, :1])
    assert np.all(np.diff(domain.bed_elevation_m[:, 0]) < 0)
    expected_source = (
        np.sum(profile.rainfall_rate_m_per_min * profile.dx_m) * 20.0 * 0.1
    )
    assert result.mass_source == pytest.approx(expected_source)


def test_reviewed_geometry_builds_channel_and_floodplain_terrain():
    from general.solvers.profile import (
        domain2d_from_profile,
        load_channel_geometry,
        load_profile,
    )

    profile = load_profile(PROFILE_PATH)
    channel_width, bankfull_depth = load_channel_geometry(
        GEOMETRY_PATH, profile.station_m
    )
    domain = domain2d_from_profile(
        profile,
        width_m=100.0,
        cross_cells=20,
        channel_width_m=channel_width,
        bankfull_depth_m=bankfull_depth,
        floodplain_slope=0.02,
    )

    assert np.allclose(channel_width, [20.0, 24.0, 24.0, 24.0, 24.0])
    assert np.allclose(bankfull_depth, [2.5, 2.8, 2.8, 2.8, 2.8])
    lateral_bed = domain.bed_elevation_m - domain.bed_elevation_m[:, 9:10]
    assert np.all(lateral_bed[:, 0] > bankfull_depth)
    assert np.allclose(lateral_bed, lateral_bed[:, ::-1])
    assert np.any(domain.slope_y > 0)
    assert np.any(domain.slope_y < 0)


def test_runner_requires_reviewed_geometry_for_2d():
    with pytest.raises(SystemExit, match="hydraulic-geometry is required"):
        run_simulation.main(
            [
                PROFILE_PATH,
                "--solver",
                "saint_venant_2d",
                "--width",
                "100",
                "--t-final",
                "0",
            ]
        )


def test_runner_initializes_2d_depth_from_level_water_surface(tmp_path):
    output_dir = tmp_path / "runs"
    run_simulation.main(
        [
            PROFILE_PATH,
            "--solver",
            "saint_venant_2d",
            "--width",
            "100",
            "--cross-cells",
            "20",
            "--hydraulic-geometry",
            GEOMETRY_PATH,
            "--t-final",
            "0",
            "--output-dir",
            str(output_dir),
            "--run-name",
            "terrain",
        ]
    )

    fields = np.load(output_dir / "terrain_fields.npz")
    initial = fields["depth_initial_m"]
    bed = fields["bed_elevation_m"]
    assert np.all(initial[:, 0] == 0.0)
    assert np.all(initial[:, -1] == 0.0)
    assert np.allclose(np.max(initial, axis=1), 0.04)
    wet = initial > 0
    water_surface = bed + initial
    for row in range(len(initial)):
        assert np.ptp(water_surface[row, wet[row]]) < 1e-12

    summary = json.loads(
        (output_dir / "terrain_summary.json").read_text(encoding="utf-8")
    )
    assert summary["grid"]["hydraulic_geometry"] == GEOMETRY_PATH
    assert summary["grid"]["bankfull_depth_m"] == [2.5, 2.8, 2.8, 2.8, 2.8]


@pytest.mark.parametrize("solver", ["kinematic_wave", "saint_venant"])
def test_runner_applies_reviewed_geometry_to_1d_solvers(tmp_path, solver):
    output_dir = tmp_path / "runs"
    run_simulation.main(
        [
            PROFILE_PATH,
            "--solver",
            solver,
            "--hydraulic-geometry",
            GEOMETRY_PATH,
            "--t-final",
            "0",
            "--output-dir",
            str(output_dir),
            "--run-name",
            solver,
        ]
    )

    summary = json.loads(
        (output_dir / f"{solver}_summary.json").read_text(encoding="utf-8")
    )
    assert summary["mass_unit"] == "m3"
    assert summary["cross_section"]["shape"] == "rectangular"
    assert summary["cross_section"]["channel_width_m"] == [
        20.0,
        24.0,
        24.0,
        24.0,
        24.0,
    ]


def test_runner_records_portable_map_inputs(tmp_path):
    output_dir = tmp_path / "runs"
    markers = "real_world_rivers/tools/example_markers.csv"
    geometry = "real_world_rivers/tools/example_geometry.csv"

    run_simulation.main(
        [
            PROFILE_PATH,
            "--solver",
            "kinematic_wave",
            "--t-final",
            "0",
            "--output-dir",
            str(output_dir),
            "--run-name",
            "mapped",
            "--map-markers",
            markers,
            "--map-geometry",
            geometry,
        ]
    )

    summary = json.loads((output_dir / "mapped_summary.json").read_text(encoding="utf-8"))
    assert summary["map_inputs"] == {
        "markers": markers,
        "geometry": geometry,
    }
