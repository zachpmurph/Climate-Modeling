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


def normalized_rmse(observed, predicted):
    """RMSE divided by the observed range; 0 is perfect."""
    obs, pred = _paired(observed, predicted)
    observed_range = float(np.max(obs) - np.min(obs))
    if observed_range == 0.0:
        return float("nan")
    return rmse(obs, pred) / observed_range


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


def kling_gupta(observed, predicted):
    """Original KGE using correlation, variability, and mean-flow ratios."""
    obs, pred = _paired(observed, predicted)
    observed_mean = float(np.mean(obs))
    observed_std = float(np.std(obs))
    predicted_mean = float(np.mean(pred))
    predicted_std = float(np.std(pred))
    correlation = pearson_r(obs, pred)
    if (
        observed_mean == 0.0
        or observed_std == 0.0
        or not np.isfinite(correlation)
    ):
        return float("nan")
    variability_ratio = predicted_std / observed_std
    mean_ratio = predicted_mean / observed_mean
    return 1.0 - float(
        np.sqrt(
            (correlation - 1.0) ** 2
            + (variability_ratio - 1.0) ** 2
            + (mean_ratio - 1.0) ** 2
        )
    )


def volumetric_efficiency(observed, predicted):
    """One minus absolute flow error divided by total observed flow volume."""
    obs, pred = _paired(observed, predicted)
    denominator = float(np.sum(np.abs(obs)))
    if denominator == 0.0:
        return float("nan")
    return 1.0 - float(np.sum(np.abs(pred - obs))) / denominator


def benchmark_skill(observed, predicted, benchmark):
    """Squared-error skill over a supplied benchmark; positive adds value."""
    obs, pred = _paired(observed, predicted)
    _, baseline = _paired(obs, benchmark)
    denominator = float(np.sum((baseline - obs) ** 2))
    if denominator == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((pred - obs) ** 2)) / denominator


def skill_scores(observed, predicted):
    """Return every metric as a dict, plus the sample count ``n``."""
    obs, _ = _paired(observed, predicted)
    return {
        "nse": nash_sutcliffe(observed, predicted),
        "rmse": rmse(observed, predicted),
        "normalized_rmse": normalized_rmse(observed, predicted),
        "bias": mean_bias(observed, predicted),
        "percent_bias": percent_bias(observed, predicted),
        "pearson_r": pearson_r(observed, predicted),
        "kge": kling_gupta(observed, predicted),
        "volumetric_efficiency": volumetric_efficiency(observed, predicted),
        "n": int(obs.size),
    }
