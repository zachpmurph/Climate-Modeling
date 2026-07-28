"""Align a predicted time series onto observation times and score the agreement.

Simulated hydrographs and gauge observations rarely share a time grid, so the
predicted series is linearly interpolated onto the observation timestamps before
the skill metrics are computed. Times are plain floats in the model's minutes.
"""

import numpy as np

from .skill import skill_scores


def interpolate_to(times, values, target_times):
    """Linearly interpolate ``values`` (sampled at ``times``) onto ``target_times``.

    ``times`` must be strictly increasing. Target times outside the sampled range
    are clamped to the endpoint values (np.interp default) — callers that require
    full temporal overlap should check the windows first.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    target = np.asarray(target_times, dtype=float)
    if times.shape != values.shape or times.ndim != 1:
        raise ValueError("times and values must be 1-D arrays of equal length")
    if times.size < 2:
        raise ValueError("need at least two samples to interpolate")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    return np.interp(target, times, values)


def evaluate_series(observed_times, observed, predicted_times, predicted):
    """Score a predicted hydrograph against an observed one.

    The predicted series is interpolated onto the observation timestamps, then the
    standard skill metrics are computed on the aligned pair. Returns the
    ``skill_scores`` dict augmented with the aligned ``predicted_on_obs`` array.
    """
    observed = np.asarray(observed, dtype=float)
    observed_times = np.asarray(observed_times, dtype=float)
    if observed.shape != observed_times.shape or observed.ndim != 1:
        raise ValueError("observed_times and observed must be 1-D arrays of equal length")
    if observed.size == 0:
        raise ValueError("no observations to evaluate against")

    predicted_on_obs = interpolate_to(predicted_times, predicted, observed_times)
    scores = skill_scores(observed, predicted_on_obs)
    scores["predicted_on_obs"] = predicted_on_obs
    return scores
