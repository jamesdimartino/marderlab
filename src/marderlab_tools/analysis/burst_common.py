from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from marderlab_tools.config.schema import PipelineSettings
from marderlab_tools.preprocess.baseline import compute_baseline, find_trigger_start
from marderlab_tools.preprocess.quality import assess_signal, zero_metrics
from marderlab_tools.preprocess.units import calibration_from_season, volts_to_centinewtons


def as_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num):
        return None
    return num


def _find_rising_edges(trigger_v: np.ndarray, threshold: float) -> np.ndarray:
    trigger = np.asarray(trigger_v, dtype=float)
    if trigger.size < 2:
        return np.array([], dtype=int)
    above = trigger >= float(threshold)
    return np.where(~above[:-1] & above[1:])[0] + 1


def _window_bounds(
    trigger_v: np.ndarray,
    sample_rate_hz: float,
    threshold: float,
    fallback_window_s: float,
) -> list[tuple[int, int]]:
    n = int(np.asarray(trigger_v).size)
    if n == 0:
        return []
    edges = _find_rising_edges(np.asarray(trigger_v, dtype=float), threshold)
    if edges.size == 0:
        stim_start = find_trigger_start(np.asarray(trigger_v, dtype=float), threshold_ratio=0.5)
        if stim_start is None:
            return [(0, n)]
        width = max(1, int(round(float(sample_rate_hz) * float(fallback_window_s))))
        return [(stim_start, min(n, stim_start + width))]

    bounds: list[tuple[int, int]] = []
    for i, start in enumerate(edges):
        stop = int(edges[i + 1]) if i + 1 < edges.size else min(
            n, int(start + max(1, round(float(sample_rate_hz) * float(fallback_window_s))))
        )
        if stop > start:
            bounds.append((int(start), int(stop)))
    return bounds or [(0, n)]


def compute_burst_metrics(
    time_s: np.ndarray,
    force_v: np.ndarray,
    trigger_v: np.ndarray,
    sample_rate_hz: float,
    metadata: dict[str, Any],
    settings: PipelineSettings,
    window_seconds: float = 10.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(force_v, dtype=float)
    trig = np.asarray(trigger_v, dtype=float)
    fs = float(sample_rate_hz or settings.sample_rate_hz)
    if y.size == 0 or t.size != y.size:
        return [], [{"code": "invalid_trace", "message": "Empty or mismatched trace.", "severity": "error"}]

    threshold = float(getattr(settings, "trigger_threshold", 0.5))
    windows = _window_bounds(trig, fs, threshold=threshold, fallback_window_s=window_seconds)

    explicit_cal = as_float(metadata.get("calibration"))
    calibration = calibration_from_season(metadata.get("season"), explicit_cal)
    trace_flags: list[dict[str, Any]] = []
    per_burst: list[dict[str, Any]] = []

    baseline_sec = float(getattr(settings, "baseline_seconds", 2.0))
    for burst_idx, (start, stop) in enumerate(windows, start=1):
        if stop <= start:
            continue
        baseline_v = compute_baseline(y, fs, start, baseline_seconds=baseline_sec)
        seg_y = y[start:stop]
        seg_t = t[start:stop]
        if seg_y.size == 0:
            continue

        delta = seg_y - baseline_v
        peak_i = int(np.argmax(delta))
        trough_i = int(np.argmin(delta))
        if abs(float(delta[trough_i])) > abs(float(delta[peak_i])):
            peak_val_v = float(seg_y[trough_i])
            amp_v_signed = float(delta[trough_i])
            direction = "down"
            peak_time = float(seg_t[trough_i])
        else:
            peak_val_v = float(seg_y[peak_i])
            amp_v_signed = float(delta[peak_i])
            direction = "up"
            peak_time = float(seg_t[peak_i])
        amp_v = float(abs(amp_v_signed))
        amp_cn = float(volts_to_centinewtons(np.array([amp_v]), calibration)[0])

        slope_v_per_s = 0.0
        if seg_t.size > 1:
            deriv = np.gradient(delta, seg_t)
            slope_v_per_s = float(np.nanmax(np.abs(deriv))) if deriv.size else 0.0
        slope_cn_per_s = float(volts_to_centinewtons(np.array([slope_v_per_s]), calibration)[0])
        auc_cn_s = float(
            np.trapezoid(
                volts_to_centinewtons(np.clip(np.abs(delta), 0, None), calibration),
                seg_t,
            )
        )

        quality = assess_signal(delta, settings.quality_std_floor, settings.quality_clip_abs)
        flags = quality.to_dicts()
        metrics: dict[str, Any] = {
            "burst_index": burst_idx,
            "start_time_s": float(seg_t[0]),
            "end_time_s": float(seg_t[-1]),
            "baseline_v": float(baseline_v),
            "peak_value_v": peak_val_v,
            "amplitude_v": amp_v,
            "amplitude_cn": amp_cn,
            "latency_s": float(max(0.0, peak_time - seg_t[0])),
            "slope_cn_per_s": slope_cn_per_s,
            "auc_cn_s": auc_cn_s,
            "direction": direction,
            "calibration": float(calibration),
            "stim_index": as_float(metadata.get("stim_index")),
            "temperature": as_float(metadata.get("temperature")),
        }
        if quality.poor_signal:
            numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            zeroed, zero_flag = zero_metrics(numeric, "Poor signal quality; metrics set to 0.")
            metrics.update(zeroed)
            metrics["direction"] = direction
            flags.append(asdict(zero_flag))
        per_burst.append({"metrics": metrics, "flags": flags})
        trace_flags.extend(flags)

    return per_burst, trace_flags
