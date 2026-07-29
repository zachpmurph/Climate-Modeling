"""Shared event-scale Green-Ampt infiltration utilities.

Units are metres and minutes.  The implementation follows the ponded
Green-Ampt cumulative-infiltration relation and limits infiltration to the
surface water actually available in a cell.
"""

from __future__ import annotations

import numpy as np


def prepare_green_ampt(
    saturated_hydraulic_conductivity_m_per_min,
    suction_head_m,
    moisture_deficit,
    shape,
):
    """Validate and broadcast a complete set of Green-Ampt soil properties."""
    values = (
        saturated_hydraulic_conductivity_m_per_min,
        suction_head_m,
        moisture_deficit,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "Green-Ampt infiltration requires soil_ksat_m_per_min, "
            "soil_suction_head_m, and soil_moisture_deficit together"
        )

    def field(value, name):
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(shape, float(array), dtype=float)
        else:
            try:
                array = np.broadcast_to(array, shape).copy()
            except ValueError as exc:
                raise ValueError(f"{name} must be scalar or have shape {shape}") from exc
        if np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    conductivity = field(values[0], "soil_ksat_m_per_min")
    suction = field(values[1], "soil_suction_head_m")
    deficit = field(values[2], "soil_moisture_deficit")
    if np.any(conductivity < 0.0):
        raise ValueError("soil_ksat_m_per_min cannot be negative")
    if np.any(suction < 0.0):
        raise ValueError("soil_suction_head_m cannot be negative")
    if np.any((deficit < 0.0) | (deficit > 1.0)):
        raise ValueError("soil_moisture_deficit must be between 0 and 1")
    return conductivity, suction, deficit


def initial_cumulative_infiltration(value, shape):
    """Validate and broadcast initial cumulative infiltration depth."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(shape, float(array), dtype=float)
    else:
        try:
            array = np.broadcast_to(array, shape).copy()
        except ValueError as exc:
            raise ValueError(
                f"initial_cumulative_infiltration_m must be scalar or have shape {shape}"
            ) from exc
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(
            "initial_cumulative_infiltration_m must contain finite, "
            "non-negative values"
        )
    return array


def green_ampt_step(
    available_depth_m,
    cumulative_infiltration_m,
    saturated_hydraulic_conductivity_m_per_min,
    suction_head_m,
    moisture_deficit,
    dt_min,
):
    """Return actual infiltration depth and updated cumulative infiltration.

    The potential ponded increment is the positive root of

        Ks dt = dF - A log((F + dF + A) / (F + A)),

    where ``A = suction_head * moisture_deficit``.  Bisection is monotone and
    avoids the singular instantaneous-capacity expression at ``F = 0``.
    Actual infiltration is capped by surface water available during the step.
    """
    available = np.maximum(np.asarray(available_depth_m, dtype=float), 0.0)
    cumulative = np.asarray(cumulative_infiltration_m, dtype=float)
    conductivity = np.asarray(
        saturated_hydraulic_conductivity_m_per_min, dtype=float
    )
    suction_product = (
        np.asarray(suction_head_m, dtype=float)
        * np.asarray(moisture_deficit, dtype=float)
    )
    target = conductivity * float(dt_min)

    # A == 0 is the saturated/zero-suction limit, dF = Ks dt.
    lower = np.zeros_like(available)
    upper = target + np.sqrt(np.maximum(2.0 * suction_product * target, 0.0))
    upper += target + suction_product

    def residual(increment):
        logarithm = np.zeros_like(increment)
        positive_a = suction_product > 0.0
        logarithm[positive_a] = np.log(
            (
                cumulative[positive_a]
                + increment[positive_a]
                + suction_product[positive_a]
            )
            / (cumulative[positive_a] + suction_product[positive_a])
        )
        return increment - suction_product * logarithm - target

    # Expand the bracket in the unlikely event that the analytic upper guess
    # is too small, then solve all cells together.
    active = target > 0.0
    for _ in range(32):
        too_small = active & (residual(upper) < 0.0)
        if not np.any(too_small):
            break
        upper[too_small] = 2.0 * upper[too_small] + target[too_small]
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        below = active & (residual(midpoint) < 0.0)
        lower = np.where(below, midpoint, lower)
        upper = np.where(below, upper, midpoint)

    potential = np.where(active, upper, 0.0)
    infiltrated = np.minimum(available, potential)
    return infiltrated, cumulative + infiltrated
