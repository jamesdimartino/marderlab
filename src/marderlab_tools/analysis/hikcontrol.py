from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marderlab_tools.config.schema import PipelineSettings
from marderlab_tools.preprocess.quality import assess_signal, zero_metrics
from marderlab_tools.preprocess.units import calibration_from_season, volts_to_centinewtons
from marderlab_tools.stats.markers import compute_stat_markers


@dataclass
class TraceRecord:
    file_path: Path
    time_s: np.ndarray
    force_v: np.ndarray
    trigger_v: np.ndarray
    sample_rate_hz: float
    metadata: dict[str, Any]


def _as_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num):
        return None
    return num


def _safe_gradient(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    if y.size < 2 or t.size < 2:
        return np.array([], dtype=float)
    return np.gradient(y, t)


def _tau_rise_loglin(t: np.ndarray, y: np.ndarray, b: float, amp: float, t10: float, t90: float) -> float:
    if not (np.isfinite(t10) and np.isfinite(t90) and t90 > t10 and amp > 0):
        return float("nan")
    m = (t >= t10) & (t <= t90) & (y > b) & ((y - b) < amp)
    if m.sum() < 8:
        return float("nan")
    z = 1.0 - (y[m] - b) / amp
    z = np.clip(z, 1e-8, 1 - 1e-8)
    A = np.vstack([t[m], np.ones_like(t[m])]).T
    slope, _ = np.linalg.lstsq(A, np.log(z), rcond=None)[0]
    if slope >= 0:
        return float("nan")
    return float(-1.0 / slope)


def _tau_decay_loglin(t: np.ndarray, y: np.ndarray, b: float, amp: float, t90d: float, t30d: float) -> float:
    if not (np.isfinite(t90d) and np.isfinite(t30d) and t30d > t90d and amp > 0):
        return float("nan")
    m = (t >= t90d) & (t <= t30d) & ((y - b) > 0)
    if m.sum() < 8:
        return float("nan")
    z = (y[m] - b) / amp
    z = np.clip(z, 1e-8, 1 - 1e-8)
    A = np.vstack([t[m], np.ones_like(t[m])]).T
    slope, _ = np.linalg.lstsq(A, np.log(z), rcond=None)[0]
    if slope >= 0:
        return float("nan")
    return float(-1.0 / slope)


def _compute_file_metrics(record: TraceRecord, settings: PipelineSettings) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t = np.asarray(record.time_s, dtype=float)
    y = np.asarray(record.force_v, dtype=float)
    fs = float(record.sample_rate_hz or settings.sample_rate_hz)
    if t.size != y.size or y.size < 3:
        metrics = {
            "baseline_v": 0.0,
            "peak_value_v": 0.0,
            "amplitude_v": 0.0,
            "amplitude_cn": 0.0,
            "t_peak_s": 0.0,
            "tau_r_s": 0.0,
            "tau_d_s": 0.0,
            "upslope_cn_per_s": 0.0,
            "auc_cn_s": 0.0,
            "amp_at_highkoff_cn": 0.0,
            "direction": "flat",
            "calibration": 0.0,
        }
        return metrics, [{"code": "invalid_trace", "message": "Trace length invalid.", "severity": "error"}]

    baseline_seconds = float(getattr(settings, "baseline_seconds", 2.0))
    highk_s = 180.0
    decay_fit_max_s = 240.0
    baseline_samples = max(1, int(round(fs * baseline_seconds)))
    baseline_v = float(np.median(y[:baseline_samples]))

    stim_mask = (t >= 0.0) & (t <= highk_s)
    if stim_mask.sum() < max(10, int(0.5 * fs)):
        metrics = {
            "baseline_v": baseline_v,
            "peak_value_v": baseline_v,
            "amplitude_v": 0.0,
            "amplitude_cn": 0.0,
            "t_peak_s": 0.0,
            "tau_r_s": 0.0,
            "tau_d_s": 0.0,
            "upslope_cn_per_s": 0.0,
            "auc_cn_s": 0.0,
            "amp_at_highkoff_cn": 0.0,
            "direction": "flat",
            "calibration": 0.0,
        }
        return metrics, [
            {
                "code": "short_stim_window",
                "message": "Insufficient samples in high-K stimulus window.",
                "severity": "warning",
            }
        ]

    # Handle both upward and downward deflections and keep the larger magnitude.
    peak_indices = np.where(stim_mask)[0]
    i_max = int(peak_indices[np.argmax(y[stim_mask])])
    i_min = int(peak_indices[np.argmin(y[stim_mask])])
    amp_up = float(y[i_max] - baseline_v)
    amp_down = float(baseline_v - y[i_min])
    if abs(amp_down) > abs(amp_up):
        i_peak = i_min
        peak_val_v = float(y[i_min])
        amp_v = amp_down
        direction = "down"
    else:
        i_peak = i_max
        peak_val_v = float(y[i_max])
        amp_v = amp_up
        direction = "up"
    t_peak = float(t[i_peak])

    # Rise tau with 10%-90% crossings pre-peak.
    y10 = baseline_v + (0.10 * amp_v if direction == "up" else -0.10 * amp_v)
    y90 = baseline_v + (0.90 * amp_v if direction == "up" else -0.90 * amp_v)
    pre = (t >= 0) & (t <= t_peak)
    if direction == "up":
        i10 = np.where(pre & (y >= y10))[0]
        i90 = np.where(pre & (y >= y90))[0]
    else:
        i10 = np.where(pre & (y <= y10))[0]
        i90 = np.where(pre & (y <= y90))[0]
    t10 = float(t[i10[0]]) if i10.size else float("nan")
    t90 = float(t[i90[0]]) if i90.size else float("nan")
    if direction == "up":
        tau_r_s = _tau_rise_loglin(t, y, baseline_v, amp_v, t10, t90)
    else:
        # Reflect downward deflection around baseline for equivalent rise fit.
        y_reflect = baseline_v + (baseline_v - y)
        tau_r_s = _tau_rise_loglin(t, y_reflect, baseline_v, amp_v, t10, t90)

    # Decay tau after high-K off.
    post = t >= highk_s
    t_decay_end = min(float(t[-1]), highk_s + decay_fit_max_s)
    post2 = (t >= highk_s) & (t <= t_decay_end)
    if direction == "up":
        y90d = baseline_v + 0.90 * amp_v
        y30d = baseline_v + 0.30 * amp_v
        i90d = np.where(post2 & (y <= y90d))[0]
        i30d = np.where(post2 & (y <= y30d))[0]
        t90d = float(t[i90d[0]]) if i90d.size else float("nan")
        t30d = float(t[i30d[0]]) if i30d.size else float("nan")
        tau_d_s = _tau_decay_loglin(t, y, baseline_v, amp_v, t90d, t30d)
    else:
        # Reflect downward for decay calculation.
        y_reflect = baseline_v + (baseline_v - y)
        y90d = baseline_v + 0.90 * amp_v
        y30d = baseline_v + 0.30 * amp_v
        i90d = np.where(post2 & (y_reflect <= y90d))[0]
        i30d = np.where(post2 & (y_reflect <= y30d))[0]
        t90d = float(t[i90d[0]]) if i90d.size else float("nan")
        t30d = float(t[i30d[0]]) if i30d.size else float("nan")
        tau_d_s = _tau_decay_loglin(t, y_reflect, baseline_v, amp_v, t90d, t30d)

    q = int(np.argmin(np.abs(t - highk_s)))
    amp_at_off_v = float(y[q] - baseline_v) if direction == "up" else float(baseline_v - y[q])

    explicit_cal = _as_float(record.metadata.get("calibration"))
    calibration = calibration_from_season(record.metadata.get("season"), explicit_cal)
    amp_cn = float(volts_to_centinewtons(np.array([amp_v]), calibration)[0])
    amp_at_off_cn = float(volts_to_centinewtons(np.array([amp_at_off_v]), calibration)[0])

    signal_for_slope = (y - baseline_v) if direction == "up" else (baseline_v - y)
    slopes = _safe_gradient(signal_for_slope, t)
    upslope_cn_per_s = (
        float(volts_to_centinewtons(np.array([np.nanmax(slopes)]), calibration)[0])
        if slopes.size else 0.0
    )
    auc_cn_s = float(np.trapezoid(
        volts_to_centinewtons(np.clip(signal_for_slope, 0, None), calibration),
        t
    ))

    metrics = {
        "baseline_v": baseline_v,
        "peak_value_v": peak_val_v,
        "amplitude_v": float(amp_v),
        "amplitude_cn": amp_cn,
        "t_peak_s": t_peak,
        "tau_r_s": float(tau_r_s) if np.isfinite(tau_r_s) else np.nan,
        "tau_d_s": float(tau_d_s) if np.isfinite(tau_d_s) else np.nan,
        "upslope_cn_per_s": upslope_cn_per_s,
        "auc_cn_s": auc_cn_s,
        "amp_at_highkoff_cn": amp_at_off_cn,
        "direction": direction,
        "temperature": _as_float(record.metadata.get("temperature")),
        "file_index": _as_float(record.metadata.get("file_index")),
        "calibration": float(calibration),
    }

    quality = assess_signal(signal_for_slope, settings.quality_std_floor, settings.quality_clip_abs)
    flags = quality.to_dicts()
    if quality.poor_signal:
        numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        zeroed, zero_flag = zero_metrics(numeric_metrics, "Poor signal quality; metrics set to 0.")
        metrics.update(zeroed)
        metrics["direction"] = direction
        flags.append(asdict(zero_flag))
    return metrics, flags


def analyze_experiment(records: list[TraceRecord], settings: PipelineSettings) -> dict[str, Any]:
    output: dict[str, Any] = {
        "pipeline": "hikcontrol",
        "files": [],
        "summary": {},
        "flags": [],
    }
    if not records:
        output["summary"] = {"n_files": 0, "mean_amplitude_cn": 0.0}
        return output

    amps: list[float] = []
    slopes: list[float] = []
    grouped_by_temp: dict[str, list[float]] = {}
    for record in sorted(records, key=lambda r: int(r.metadata.get("file_index", 0))):
        metrics, flags = _compute_file_metrics(record, settings)
        output["files"].append(
            {
                "file_path": str(record.file_path),
                "file_index": int(record.metadata.get("file_index", 0)),
                "metrics": metrics,
                "flags": flags,
            }
        )
        amp_cn = float(metrics.get("amplitude_cn", 0.0))
        slope_cn = float(metrics.get("upslope_cn_per_s", 0.0))
        amps.append(amp_cn)
        slopes.append(slope_cn)

        temp = metrics.get("temperature")
        if temp is not None and np.isfinite(float(temp)):
            key = str(int(round(float(temp))))
            grouped_by_temp.setdefault(key, []).append(amp_cn)

        for flag in flags:
            output["flags"].append({"file_path": str(record.file_path), **flag})

    output["summary"] = {
        "n_files": len(output["files"]),
        "mean_amplitude_cn": float(np.mean(amps)) if amps else 0.0,
        "mean_upslope_cn_per_s": float(np.mean(slopes)) if slopes else 0.0,
        "stats_by_temperature": compute_stat_markers(grouped_by_temp) if grouped_by_temp else {
            "rule": "by_group_count",
            "test": "insufficient_groups",
            "p_value": None,
            "stars": "ns",
            "pairwise": [],
        },
    }
    return output
