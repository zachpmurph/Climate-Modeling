"""Tests for the unified run_simulation dispatch harness."""

import pytest
import numpy as np

from general.solvers.contract import Domain, Scenario, UnsupportedScenario
from rivers.simulations.registry import dispatch, SOLVERS


PROFILE_PATH = "real_world_rivers/tools/example_river_profile.csv"


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
    assert "river_kinematic_wave" in SOLVERS
    assert "saint_venant" in SOLVERS


def test_unknown_solver_raises():
    with pytest.raises(KeyError, match="Unknown solver"):
        dispatch("no_such_solver", _load_domain(), _make_scenario())


# ── dispatch by name ──────────────────────────────────────────────────────
def test_dispatch_river_kinematic_wave():
    domain = _load_domain()
    scenario = _make_scenario(left_inflow=0.0006, t_final_min=2.0)
    result = dispatch("river_kinematic_wave", domain, scenario)
    assert result.depth_history.shape[0] >= 2
    assert result.depth_history.shape[1] == len(domain.x_m)
    assert result.mass_inflow >= 0


def test_dispatch_kinematic_wave():
    domain = _load_domain()
    scenario = _make_scenario(t_final_min=2.0)
    result = dispatch("kinematic_wave", domain, scenario)
    assert result.depth_history.shape[0] >= 2
    # kinematic_wave uses its own internal grid (L*10 resolution), not domain.x_m
    assert result.depth_history.shape[1] == len(result.domain.x_m)
    assert result.mass_inflow == 0.0


# ── same profile, two solvers, both produce valid outputs ─────────────────
def test_two_solvers_on_same_profile():
    domain = _load_domain()

    # river_kinematic_wave with inflow
    r1 = dispatch("river_kinematic_wave", domain, _make_scenario(t_final_min=3.0, left_inflow=0.0006))
    assert np.all(r1.depth_history >= 0)

    # kinematic_wave without inflow (doesn't support left_inflow)
    r2 = dispatch("kinematic_wave", domain, _make_scenario(t_final_min=3.0))
    assert np.all(r2.depth_history >= 0)


# ── UnsupportedScenario ───────────────────────────────────────────────────
def test_unsupported_left_inflow_on_kinematic_wave():
    domain = _load_domain()
    scenario = _make_scenario(left_inflow=0.001)
    with pytest.raises(UnsupportedScenario, match="left_inflow"):
        dispatch("kinematic_wave", domain, scenario)


def test_unsupported_initial_discharge_on_kinematic_wave():
    domain = _load_domain()
    scenario = _make_scenario(initial_discharge=0.001)
    with pytest.raises(UnsupportedScenario, match="initial_discharge"):
        dispatch("kinematic_wave", domain, scenario)


# ── SimulationResult shape invariants ─────────────────────────────────────
def test_simulation_result_shapes_river_kinematic_wave():
    domain = _load_domain()
    scenario = _make_scenario(t_final_min=3.0, left_inflow=0.0006)
    result = dispatch("river_kinematic_wave", domain, scenario)
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
    n_cells = len(result.domain.x_m)
    assert result.depth_history.shape == (n_times, n_cells)
    assert result.depth_initial.shape == (n_cells,)
    assert result.depth_final.shape == (n_cells,)
