"""Hydrograph error decomposition without parameter fitting."""

from __future__ import annotations

import numpy as np

from rivers.validation.skill import skill_scores


def _peak_time(times, values):
    return float(times[int(np.argmax(values))])


def _routing_lag_minutes(times, upstream, downstream):
    """Return the lag giving the strongest anomaly correlation.

    Positive lag means the downstream series follows the upstream series.
    This is descriptive timing evidence, not a time-shift correction.
    """
    times = np.asarray(times, dtype=float)
    upstream = np.asarray(upstream, dtype=float)
    downstream = np.asarray(downstream, dtype=float)
    if len(times) < 5:
        return None
    interval = float(np.median(np.diff(times)))
    maximum_shift = max(1, min(len(times) // 3, int(12 * 60 / interval)))
    best = None
    for shift in range(-maximum_shift, maximum_shift + 1):
        if shift >= 0:
            first = upstream[: len(upstream) - shift or None]
            second = downstream[shift:]
        else:
            first = upstream[-shift:]
            second = downstream[: len(downstream) + shift]
        if len(first) < 4 or np.std(first) == 0.0 or np.std(second) == 0.0:
            continue
        correlation = float(np.corrcoef(first, second)[0, 1])
        candidate = (correlation, -abs(shift), shift)
        if best is None or candidate > best:
            best = candidate
    return None if best is None else float(best[2] * interval)


def diagnose_reach_routing(
    times,
    upstream,
    observed_downstream,
    predicted_downstream,
):
    """Compare reach-scale water volume and routing timing."""
    times = np.asarray(times, dtype=float)
    upstream = np.asarray(upstream, dtype=float)
    observed = np.asarray(observed_downstream, dtype=float)
    predicted = np.asarray(predicted_downstream, dtype=float)
    if not (
        times.shape == upstream.shape == observed.shape == predicted.shape
    ):
        raise ValueError("routing diagnostics require aligned 1-D series")

    upstream_volume = float(np.trapezoid(upstream, times))
    observed_volume = float(np.trapezoid(observed, times))
    predicted_volume = float(np.trapezoid(predicted, times))
    observed_exchange = observed_volume - upstream_volume
    modeled_exchange = predicted_volume - upstream_volume
    observed_lag = _routing_lag_minutes(times, upstream, observed)
    modeled_lag = _routing_lag_minutes(times, upstream, predicted)
    return {
        "upstream_volume_m3": upstream_volume,
        "observed_downstream_volume_m3": observed_volume,
        "predicted_downstream_volume_m3": predicted_volume,
        "observed_net_change_from_upstream_m3": observed_exchange,
        "observed_net_change_fraction": (
            None if abs(upstream_volume) < 1e-12 else observed_exchange / upstream_volume
        ),
        "modeled_net_change_from_upstream_m3": modeled_exchange,
        "modeled_net_change_fraction": (
            None if abs(upstream_volume) < 1e-12 else modeled_exchange / upstream_volume
        ),
        "unexplained_downstream_volume_m3": observed_volume - predicted_volume,
        "observed_routing_lag_min": observed_lag,
        "modeled_routing_lag_min": modeled_lag,
        "routing_lag_error_min": (
            None
            if observed_lag is None or modeled_lag is None
            else modeled_lag - observed_lag
        ),
        "guard": (
            "Volume change combines intervening flows, withdrawals, measurement "
            "error, and net reach-storage change; it is not a calibrated lateral flow."
        ),
    }


def diagnose_hydrograph(times, observed, predicted):
    """Separate volume, amplitude, timing, and shape symptoms.

    These are diagnostic indicators, not a parameter-selection objective.
    """
    times = np.asarray(times, dtype=float)
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if not (times.shape == observed.shape == predicted.shape) or times.ndim != 1:
        raise ValueError("times, observed, and predicted must be equal 1-D arrays")
    if times.size < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("diagnostics need at least two strictly ordered samples")

    observed_volume = float(np.trapezoid(observed, times))
    predicted_volume = float(np.trapezoid(predicted, times))
    observed_range = float(np.ptp(observed))
    predicted_range = float(np.ptp(predicted))
    volume_ratio = (
        None if abs(observed_volume) < 1e-12 else predicted_volume / observed_volume
    )
    volume_scale = (
        None if abs(predicted_volume) < 1e-12 else observed_volume / predicted_volume
    )
    amplitude_ratio = (
        None if observed_range < 1e-12 else predicted_range / observed_range
    )
    peak_lag = _peak_time(times, predicted) - _peak_time(times, observed)
    correlation = (
        None
        if float(np.std(observed)) == 0.0 or float(np.std(predicted)) == 0.0
        else float(np.corrcoef(observed, predicted)[0, 1])
    )

    signals = []
    if volume_ratio is not None and abs(volume_ratio - 1.0) > 0.10:
        signals.append(
            {
                "component": "volume",
                "likely_sources": [
                    "missing or excess lateral flow",
                    "boundary forcing bias",
                    "storage/geometry bias",
                ],
            }
        )
    if amplitude_ratio is not None and amplitude_ratio < 0.75:
        signals.append(
            {
                "component": "attenuation",
                "likely_sources": [
                    "excess channel/floodplain storage",
                    "roughness or geometry bias",
                    "numerical diffusion",
                ],
            }
        )
    elif amplitude_ratio is not None and amplitude_ratio > 1.25:
        signals.append(
            {
                "component": "amplification",
                "likely_sources": [
                    "insufficient storage",
                    "geometry or roughness bias",
                    "unrepresented downstream control",
                ],
            }
        )
    sample_interval = float(np.median(np.diff(times)))
    if abs(peak_lag) > sample_interval:
        signals.append(
            {
                "component": "timing",
                "likely_sources": [
                    "wave-speed/roughness error",
                    "bed or cross-section geometry error",
                    "boundary-control timing",
                ],
            }
        )
    if correlation is not None and np.isfinite(correlation) and correlation < 0.7:
        signals.append(
            {
                "component": "shape",
                "likely_sources": [
                    "missing reach processes or controls",
                    "forcing timing error",
                    "structural model inadequacy",
                ],
            }
        )
    if not signals:
        signals.append(
            {
                "component": "residual",
                "likely_sources": [
                    "measurement uncertainty",
                    "smaller combined forcing, geometry, and numerical errors",
                ],
            }
        )

    return {
        "observed_volume_m3": observed_volume,
        "predicted_volume_m3": predicted_volume,
        "volume_ratio": volume_ratio,
        "volume_only_counterfactual": (
            None
            if volume_scale is None
            else {
                "scale_applied_to_prediction": volume_scale,
                "scores": {
                    key: (
                        int(value)
                        if key == "n"
                        else None
                        if not np.isfinite(float(value))
                        else float(value)
                    )
                    for key, value in skill_scores(
                        observed, predicted * volume_scale
                    ).items()
                },
                "guard": (
                    "Uses held-out downstream volume as an oracle diagnostic. "
                    "It is not a permissible forcing correction or calibration."
                ),
            }
        ),
        "amplitude_ratio": amplitude_ratio,
        "observed_peak_time_min": _peak_time(times, observed),
        "predicted_peak_time_min": _peak_time(times, predicted),
        "peak_lag_min": peak_lag,
        "pearson_r": correlation,
        "signals": signals,
    }
