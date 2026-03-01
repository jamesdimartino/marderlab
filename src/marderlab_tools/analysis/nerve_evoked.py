from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marderlab_tools.config.schema import PipelineSettings
from marderlab_tools.preprocess.quality import assess_signal, zero_metrics
from marderlab_tools.preprocess.units import calibration_from_season, volts_to_centinewtons


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


def _smooth_trace(y: np.ndarray, fs: float) -> np.ndarray:
    try:
        from scipy.signal import savgol_filter  # type: ignore
    except Exception:
        return y

    if y.size < 7:
        return y
    win_len = int(0.1 * fs) if fs > 0 else 11
    if win_len % 2 == 0:
        win_len += 1
    win_len = max(win_len, 5)
    if win_len >= y.size:
        win_len = y.size - 1 if y.size % 2 == 0 else y.size
    if win_len < 5 or win_len >= y.size:
        return y
    return savgol_filter(y, win_len, 3)


def _find_peak_index(y_smooth: np.ndarray, baseline_v: float, fs: float) -> int:
    try:
        from scipy.signal import find_peaks  # type: ignore
    except Exception:
        return int(np.argmax(y_smooth))

    peaks, props = find_peaks(
        y_smooth,
        height=baseline_v + 0.05,
        width=max(1, int(0.2 * fs)),
        prominence=0.02,
    )
    if len(peaks) == 0:
        return int(np.argmax(y_smooth))
    return int(peaks[np.argmax(props.get("peak_heights", np.zeros_like(peaks, dtype=float)))])


def _compute_file_metrics(record: TraceRecord, settings: PipelineSettings) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t = np.asarray(record.time_s, dtype=float)
    y = np.asarray(record.force_v, dtype=float)
    fs = float(record.sample_rate_hz or settings.sample_rate_hz)

    if y.size == 0 or t.size != y.size:
        metrics = {
            "baseline_v": 0.0,
            "peak_value_v": 0.0,
            "peak_time_s": 0.0,
            "amplitude_v": 0.0,
            "amplitude_cn": 0.0,
            "calibration": 0.0,
        }
        return metrics, [
            {
                "code": "invalid_trace",
                "message": "Trace is empty or mismatched.",
                "severity": "error",
            }
        ]

    baseline_points = max(1, int(fs))
    baseline_v = float(np.mean(y[:baseline_points]))

    mask = (t >= 10.0) & (t <= 50.0)
    t_win = t[mask]
    y_win = y[mask]
    if y_win.size == 0:
        metrics = {
            "baseline_v": baseline_v,
            "peak_value_v": 0.0,
            "peak_time_s": 0.0,
            "amplitude_v": 0.0,
            "amplitude_cn": 0.0,
            "calibration": 0.0,
        }
        return metrics, [
            {
                "code": "empty_window",
                "message": "No data in the 10-50s analysis window.",
                "severity": "warning",
            }
        ]

    y_smooth = _smooth_trace(y_win, fs)
    peak_idx = _find_peak_index(y_smooth, baseline_v, fs)
    peak_val_v = float(y_smooth[peak_idx])
    peak_time_s = float(t_win[peak_idx])
    amp_v = float(peak_val_v - baseline_v)

    explicit_cal = _as_float(record.metadata.get("calibration"))
    calibration = calibration_from_season(record.metadata.get("season"), explicit_cal)
    amp_cn = float(volts_to_centinewtons(np.array([amp_v]), calibration)[0])

    metrics: dict[str, Any] = {
        "baseline_v": baseline_v,
        "peak_value_v": peak_val_v,
        "peak_time_s": peak_time_s,
        "amplitude_v": amp_v,
        "amplitude_cn": amp_cn,
        "stim_index": _as_float(record.metadata.get("stim_index")),
        "temperature": _as_float(record.metadata.get("temperature")),
        "calibration": float(calibration),
    }

    quality = assess_signal(y_win - baseline_v, settings.quality_std_floor, settings.quality_clip_abs)
    flags = quality.to_dicts()
    if quality.poor_signal:
        numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        zeroed, zero_flag = zero_metrics(numeric_metrics, "Poor signal quality; metrics set to 0.")
        metrics.update(zeroed)
        flags.append(asdict(zero_flag))
    return metrics, flags


def analyze_experiment(records: list[TraceRecord], settings: PipelineSettings) -> dict[str, Any]:
    output: dict[str, Any] = {
        "pipeline": "nerve_evoked",
        "files": [],
        "summary": {},
        "flags": [],
    }

    amplitudes: list[float] = []
    for record in sorted(records, key=lambda r: int(r.metadata.get("file_index", 0))):
        metrics, flags = _compute_file_metrics(record, settings)
        file_index = int(record.metadata.get("file_index", 0))
        output["files"].append(
            {
                "file_path": str(record.file_path),
                "file_index": file_index,
                "metrics": metrics,
                "flags": flags,
            }
        )
        amplitudes.append(float(metrics.get("amplitude_cn", 0.0)))
        for flag in flags:
            output["flags"].append({"file_path": str(record.file_path), **flag})

    output["summary"] = {
        "n_files": len(output["files"]),
        "mean_amplitude_cn": float(np.mean(amplitudes)) if amplitudes else 0.0,
    }
    return output
