"""Hydrologic skill metrics for comparing a predicted series to an observed one.

All functions take two equal-length 1-D arrays ``observed`` and ``predicted`` and
return a scalar. They are pure and unit-agnostic (whatever units both series share).
"""

import numpy as np


def _paired(observed, predicted):
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(f"observed and predicted must have the same shape, got {obs.shape} vs {pred.shape}")
    if obs.ndim != 1 or obs.size == 0:
        raise ValueError("observed and predicted must be non-empty 1-D arrays")
    return obs, pred


def rmse(observed, predicted):
    """Root-mean-square error (same units as the series; 0 is perfect)."""
    obs, pred = _paired(observed, predicted)
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def mean_bias(observed, predicted):
    """Mean signed error, mean(predicted - observed). Positive = over-prediction."""
    obs, pred = _paired(observed, predicted)
    return float(np.mean(pred - obs))


def percent_bias(observed, predicted):
    """Percent bias: 100 * sum(predicted - observed) / sum(observed)."""
    obs, pred = _paired(observed, predicted)
    denom = float(np.sum(obs))
    if denom == 0.0:
        return float("nan")
    return 100.0 * float(np.sum(pred - obs)) / denom


def nash_sutcliffe(observed, predicted):
    """Nash-Sutcliffe efficiency in (-inf, 1]. 1 is perfect; 0 means the model is no
    better than predicting the observed mean; < 0 means worse than the mean."""
    obs, pred = _paired(observed, predicted)
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    if denom == 0.0:
        return float("nan")  # observations are constant; NSE undefined
    return 1.0 - float(np.sum((pred - obs) ** 2)) / denom


def pearson_r(observed, predicted):
    """Pearson correlation coefficient in [-1, 1]."""
    obs, pred = _paired(observed, predicted)
    if np.std(obs) == 0.0 or np.std(pred) == 0.0:
        return float("nan")
    return float(np.corrcoef(obs, pred)[0, 1])


def skill_scores(observed, predicted):
    """Return every metric as a dict, plus the sample count ``n``."""
    obs, _ = _paired(observed, predicted)
    return {
        "nse": nash_sutcliffe(observed, predicted),
        "rmse": rmse(observed, predicted),
        "bias": mean_bias(observed, predicted),
        "percent_bias": percent_bias(observed, predicted),
        "pearson_r": pearson_r(observed, predicted),
        "n": int(obs.size),
    }
