from general.solvers.contract import UnsupportedScenario
import general.solvers.linear_advection as _la
import general.solvers.saint_venant_1d as _sv
import general.solvers.river_kinematic_wave as _rkw

import numpy as np

SOLVERS = {
    "kinematic_wave": _la.SOLVER,
    "saint_venant": _sv.SOLVER,
    "river_kinematic_wave": _rkw.SOLVER,
}


def dispatch(name: str, domain, scenario):
    if name not in SOLVERS:
        raise KeyError(f"Unknown solver '{name}'. Available: {sorted(SOLVERS)}")
    solver = SOLVERS[name]
    _check_scenario(solver, scenario)
    return solver.run(domain, scenario)


def _check_scenario(solver, scenario) -> None:
    """Raise UnsupportedScenario if a non-default knob isn't in solver.supports."""
    checks = {
        "left_inflow": lambda s: (callable(s.left_inflow) or float(s.left_inflow) != 0.0),
        "initial_discharge": lambda s: (
            isinstance(s.initial_discharge, np.ndarray) or float(s.initial_discharge) != 0.0
        ),
        "rainfall": lambda s: s.rainfall is not None,
        "initial_depth": lambda s: isinstance(s.initial_depth_m, np.ndarray) or float(s.initial_depth_m) != 0.0,
    }
    for knob, is_active in checks.items():
        if knob not in solver.supports and is_active(scenario):
            raise UnsupportedScenario(
                f"Solver '{solver.name}' does not support the '{knob}' scenario knob. "
                f"Solver supports: {sorted(solver.supports)}"
            )
