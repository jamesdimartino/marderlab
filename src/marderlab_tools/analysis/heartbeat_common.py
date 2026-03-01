from __future__ import annotations

from typing import Any

import numpy as np

from marderlab_tools.config.schema import PipelineSettings
from marderlab_tools.preprocess.baseline import compute_baseline
from marderlab_tools.preprocess.quality import assess_signal, zero_metrics
from marderlab_tools.preprocess.units import calibration_from_season, volts_to_centinewtons


def analyze_heartbeat_trace(
    time_s: np.ndarray,
    force_v: np.ndarray,
    sample_rate_hz: float,
    metadata: dict[str, Any],
    settings: PipelineSettings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(force_v, dtype=float)
    fs = float(sample_rate_hz or settings.sample_rate_hz)
    if y.size < 3 or t.size != y.size:
        return {
            "baseline_v": 0.0,
            "peak_count": 0,
            "mean_period_s": 0.0,
            "heart_rate_bpm": 0.0,
            "mean_peak_to_peak_cn": 0.0,
        }, [{"code": "invalid_trace", "message": "Trace invalid for heartbeat analysis.", "severity": "error"}]

    baseline_v = compute_baseline(y, fs, None, baseline_seconds=float(getattr(settings, "baseline_seconds", 2.0)))
    centered = y - baseline_v
    try:
        from scipy.signal import find_peaks  # type: ignore

        min_dist = max(1, int(round(0.2 * fs)))
        prominence = max(1e-4, float(np.nanstd(centered)) * 0.25)
        peaks, _ = find_peaks(centered, distance=min_dist, prominence=prominence)
        troughs, _ = find_peaks(-centered, distance=min_dist, prominence=prominence)
    except Exception:
        # Fallback simple local maxima/minima.
        d = np.diff(centered)
        peaks = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
        troughs = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1

    periods = np.diff(t[peaks]) if peaks.size > 1 else np.array([], dtype=float)
    mean_period = float(np.nanmean(periods)) if periods.size else 0.0
    heart_rate = float(60.0 / mean_period) if mean_period > 0 else 0.0
    ptp_v = float(np.nanmean(y[peaks]) - np.nanmean(y[troughs])) if peaks.size and troughs.size else 0.0

    explicit_cal = None
    try:
        raw_cal = metadata.get("calibration")
        explicit_cal = float(raw_cal) if raw_cal is not None else None
    except (TypeError, ValueError):
        explicit_cal = None
    calibration = calibration_from_season(metadata.get("season"), explicit_cal)
    ptp_cn = float(volts_to_centinewtons(np.array([abs(ptp_v)]), calibration)[0])

    quality = assess_signal(centered, settings.quality_std_floor, settings.quality_clip_abs)
    flags = quality.to_dicts()
    metrics = {
        "baseline_v": float(baseline_v),
        "peak_count": int(peaks.size),
        "mean_period_s": mean_period,
        "heart_rate_bpm": heart_rate,
        "mean_peak_to_peak_cn": ptp_cn,
        "temperature": metadata.get("temperature"),
        "calibration": float(calibration),
    }
    if quality.poor_signal:
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        zeroed, flag = zero_metrics(numeric, "Poor heartbeat signal quality; metrics set to 0.")
        metrics.update(zeroed)
        flags.append({"code": flag.code, "message": flag.message, "severity": flag.severity})
    return metrics, flags
