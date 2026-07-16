from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import numpy as np


class UnsupportedScenario(Exception):
    """Raised when a Scenario knob is not in a solver's ``supports`` set."""


@dataclass(frozen=True)
class Domain:
    """Per-cell spatial description of a 1-D river reach."""

    x_m: np.ndarray        # cell-centre positions, metres
    dx_m: np.ndarray       # cell widths, metres
    slope: np.ndarray      # bed slope S0, dimensionless
    manning_n: np.ndarray  # Manning roughness n


@dataclass
class Scenario:
    """Everything the solver needs beyond the domain geometry."""

    t_final_min: float
    record_interval_min: float = 1.0
    initial_depth_m: float | np.ndarray = 0.0
    initial_discharge: float | np.ndarray = 0.0
    left_inflow: float | Callable[[float], float] = 0.0
    rainfall: Callable[[np.ndarray, float], np.ndarray] | None = None
    cfl: float = 0.5


@dataclass
class SimulationResult:
    """Canonical output every solver must produce."""

    domain: Domain
    times: np.ndarray               # shape (n_times,)
    depth_history: np.ndarray       # shape (n_times, n_cells)
    depth_initial: np.ndarray       # shape (n_cells,)
    depth_final: np.ndarray         # shape (n_cells,)
    mass_inflow: float
    mass_source: float
    mass_outflow: float
    extra: dict = field(default_factory=dict)


@runtime_checkable
class Solver(Protocol):
    name: str
    supports: frozenset[str]

    def run(self, domain: Domain, scenario: Scenario) -> SimulationResult:
        ...
