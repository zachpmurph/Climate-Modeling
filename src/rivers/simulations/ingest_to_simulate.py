"""Helper that converts an ingested river profile into a (Domain, Scenario) ready for dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from general.solvers.contract import Domain, Scenario
from general.solvers.profile import domain_from_profile, load_profile


def profile_to_domain_scenario(
    profile_path: str | Path,
    t_final_min: float,
    left_inflow: float = 0.0,
    rainfall_rate_m_per_min: float = 0.0,
    record_interval_min: float = 1.0,
    cfl: float = 0.5,
) -> tuple[Domain, Scenario]:
    """Load *profile_path* and build a (Domain, Scenario) pair.

    Args:
        profile_path: CSV or JSON river profile produced by rivers.ingest.export_profile.
        t_final_min: Simulation duration, minutes.
        left_inflow: Constant upstream inflow flux, m^2/min.
        rainfall_rate_m_per_min: Uniform rainfall rate, m/min (0 = off).
        record_interval_min: Snapshot interval, minutes.
        cfl: CFL target.

    Returns:
        (domain, scenario) ready for ``registry.dispatch(solver_name, domain, scenario)``.
    """
    profile = load_profile(profile_path)
    domain = domain_from_profile(profile)

    rainfall_fn = None
    if rainfall_rate_m_per_min > 0:
        rate = rainfall_rate_m_per_min
        rainfall_fn = lambda x, t: np.full_like(x, rate)

    scenario = Scenario(
        t_final_min=t_final_min,
        record_interval_min=record_interval_min,
        left_inflow=left_inflow,
        rainfall=rainfall_fn,
        cfl=cfl,
    )
    return domain, scenario
