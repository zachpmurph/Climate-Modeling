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
    bed_elevation_m: np.ndarray | None = None  # explicit z_b, shape (nx,)
    channel_width_m: np.ndarray | None = None  # reviewed/reference width, shape (nx,)
    bankfull_depth_m: np.ndarray | None = None  # reviewed reference depth, shape (nx,)
    channel_bottom_width_m: np.ndarray | None = None  # trapezoid bottom width, shape (nx,)
    side_slope_h_to_v: np.ndarray | None = None  # trapezoid side slope per bank, shape (nx,)
    cross_section_depth_m: np.ndarray | None = None  # common stage-curve depths, shape (nlevel,)
    cross_section_top_width_m: np.ndarray | None = None  # compound width curves, shape (nx, nlevel)
    cross_section_wetted_perimeter_m: np.ndarray | None = None  # surveyed curves, shape (nx, nlevel)


@dataclass(frozen=True)
class Domain2D:
    """Cell-centred description of a rectangular 2-D river domain."""

    x_m: np.ndarray          # x cell-centre positions, shape (nx,)
    y_m: np.ndarray          # y cell-centre positions, shape (ny,)
    dx_m: np.ndarray         # x cell widths, shape (nx,)
    dy_m: np.ndarray         # y cell widths, shape (ny,)
    slope_x: np.ndarray      # x bed slope, shape (nx, ny)
    slope_y: np.ndarray      # y bed slope, shape (nx, ny)
    manning_n: np.ndarray    # Manning roughness, shape (nx, ny)
    bed_elevation_m: np.ndarray | None = None  # z_b, shape (nx, ny)


@dataclass
class Scenario:
    """Everything the solver needs beyond the domain geometry."""

    t_final_min: float
    record_interval_min: float = 1.0
    initial_depth_m: float | np.ndarray = 0.0
    initial_discharge: float | np.ndarray = 0.0
    initial_discharge_y: float | np.ndarray = 0.0
    left_inflow: float | Callable[[float], float] = 0.0
    rainfall: Callable[[np.ndarray, float], np.ndarray] | None = None
    lateral_inflow: Callable[[np.ndarray, float], np.ndarray] | None = None
    rainfall_2d: Callable[[np.ndarray, np.ndarray, float], np.ndarray] | None = None
    cfl: float = 0.5
    labels: tuple[str, ...] = ()
    boundary_x: str = "inflow_outflow"
    boundary_y: str = "wall"
    downstream_boundary: str = "outflow"
    downstream_stage_m: float | Callable[[float], float] | None = None
    spatial_order: int = 1


@dataclass
class SimulationResult:
    """Canonical output every solver must produce."""

    domain: Domain | Domain2D
    times: np.ndarray               # shape (n_times,)
    depth_history: np.ndarray       # 1-D: (nt, nx); 2-D: (nt, nx, ny)
    depth_initial: np.ndarray       # 1-D: (nx,); 2-D: (nx, ny)
    depth_final: np.ndarray         # 1-D: (nx,); 2-D: (nx, ny)
    mass_inflow: float
    mass_source: float
    mass_outflow: float
    mass_correction: float = 0.0
    extra: dict = field(default_factory=dict)


@runtime_checkable
class Solver(Protocol):
    name: str
    supports: frozenset[str]

    def run(self, domain: Domain | Domain2D, scenario: Scenario) -> SimulationResult:
        ...
