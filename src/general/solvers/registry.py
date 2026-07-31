"""Registry for numerical solvers, independent of any river-ingestion workflow."""

import numpy as np

from general.solvers.contract import UnsupportedScenario
import general.solvers.linear_advection as _la
import general.solvers.saint_venant_1d as _sv
import general.solvers.saint_venant_2d as _sv2


SOLVERS = {
    "kinematic_wave": _la.SOLVER,
    "saint_venant": _sv.SOLVER,
    "saint_venant_2d": _sv2.SOLVER,
}


def dispatch(name: str, domain, scenario):
    if name not in SOLVERS:
        raise KeyError(f"Unknown solver '{name}'. Available: {sorted(SOLVERS)}")
    solver = SOLVERS[name]
    _check_scenario(solver, scenario)
    if (
        getattr(domain, "soil_ksat_m_per_min", None) is not None
        and "soil_infiltration" not in solver.supports
    ):
        raise UnsupportedScenario(
            f"Solver '{solver.name}' does not support soil infiltration. "
            "Use saint_venant or saint_venant_2d for a soil-enabled domain."
        )
    return solver.run(domain, scenario)


def _check_scenario(solver, scenario):
    checks = {
        "left_inflow": lambda s: callable(s.left_inflow)
        or float(s.left_inflow) != 0.0,
        "initial_discharge": lambda s: isinstance(s.initial_discharge, np.ndarray)
        or float(s.initial_discharge) != 0.0,
        "initial_discharge_y": lambda s: isinstance(
            s.initial_discharge_y, np.ndarray
        )
        or float(s.initial_discharge_y) != 0.0,
        "rainfall": lambda s: s.rainfall is not None,
        "lateral_inflow": lambda s: s.lateral_inflow is not None,
        "rainfall_2d": lambda s: s.rainfall_2d is not None,
        "initial_depth": lambda s: isinstance(s.initial_depth_m, np.ndarray)
        or float(s.initial_depth_m) != 0.0,
        "boundary_x": lambda s: s.boundary_x != "inflow_outflow",
        "boundary_y": lambda s: s.boundary_y != "wall",
        "downstream_boundary": lambda s: s.downstream_boundary != "outflow",
        "downstream_stage": lambda s: s.downstream_stage_m is not None,
        "spatial_order": lambda s: s.spatial_order != 1,
        "initial_cumulative_infiltration": lambda s: isinstance(
            s.initial_cumulative_infiltration_m, np.ndarray
        )
        or float(s.initial_cumulative_infiltration_m) != 0.0,
    }
    for knob, is_active in checks.items():
        if knob not in solver.supports and is_active(scenario):
            raise UnsupportedScenario(
                f"Solver '{solver.name}' does not support the '{knob}' scenario "
                f"knob. Solver supports: {sorted(solver.supports)}"
            )
