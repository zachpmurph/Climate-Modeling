"""Helper that converts an ingested river profile into a (Domain, Scenario) ready for dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from general.solvers.contract import Domain, Scenario
from general.solvers.profile import domain_from_profile, load_profile


def rainfall_from_profile(profile, additional_rate_m_per_min=0.0):
    """Build a spatial rainfall function from profile data plus a uniform rate."""
    if additional_rate_m_per_min < 0:
        raise ValueError("rainfall_rate_m_per_min must be non-negative")
    profile_rate = profile.rainfall_rate_m_per_min
    if profile_rate is None and additional_rate_m_per_min == 0:
        return None

    stations = np.asarray(profile.station_m, dtype=float).copy()
    base_rate = (
        np.zeros_like(stations)
        if profile_rate is None
        else np.asarray(profile_rate, dtype=float).copy()
    )
    additional_rate = float(additional_rate_m_per_min)

    def rainfall(x, t):
        del t
        x = np.asarray(x, dtype=float)
        if x.shape == stations.shape and np.array_equal(x, stations):
            spatial_rate = base_rate
        else:
            spatial_rate = np.interp(x, stations, base_rate)
        return spatial_rate + additional_rate

    return rainfall


def scenario_from_profile(
    profile,
    *,
    t_final_min,
    left_inflow=0.0,
    rainfall_rate_m_per_min=0.0,
    record_interval_min=1.0,
    cfl=0.5,
):
    """Transfer every optional RiverProfile field into a Scenario."""
    initial_depth = (
        np.asarray(profile.initial_depth_m, dtype=float).copy()
        if profile.initial_depth_m is not None
        else 0.0
    )
    return Scenario(
        t_final_min=t_final_min,
        record_interval_min=record_interval_min,
        initial_depth_m=initial_depth,
        left_inflow=left_inflow,
        rainfall=rainfall_from_profile(profile, rainfall_rate_m_per_min),
        cfl=cfl,
        labels=tuple(profile.labels),
    )


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

    scenario = scenario_from_profile(
        profile,
        t_final_min=t_final_min,
        record_interval_min=record_interval_min,
        left_inflow=left_inflow,
        rainfall_rate_m_per_min=rainfall_rate_m_per_min,
        cfl=cfl,
    )
    return domain, scenario
