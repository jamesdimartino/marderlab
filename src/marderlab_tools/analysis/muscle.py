from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marderlab_tools.analysis.burst_common import compute_burst_metrics
from marderlab_tools.config.schema import PipelineSettings
from marderlab_tools.stats.markers import compute_stat_markers


@dataclass
class TraceRecord:
    file_path: Path
    time_s: np.ndarray
    force_v: np.ndarray
    trigger_v: np.ndarray
    sample_rate_hz: float
    metadata: dict[str, Any]


def _maybe_lowpass(y: np.ndarray, fs: float) -> np.ndarray:
    try:
        from scipy.signal import butter, filtfilt  # type: ignore
    except Exception:
        return y
    if y.size < 12 or fs <= 0:
        return y
    cutoff = min(5.0, 0.45 * fs)
    wn = cutoff / (0.5 * fs)
    if wn <= 0 or wn >= 1:
        return y
    b, a = butter(2, wn, btype="low")
    return filtfilt(b, a, y)


def analyze_experiment(records: list[TraceRecord], settings: PipelineSettings) -> dict[str, Any]:
    output: dict[str, Any] = {"pipeline": "muscle", "files": [], "summary": {}, "flags": []}
    grouped: dict[str, list[float]] = {}
    amplitudes: list[float] = []

    for record in sorted(records, key=lambda r: int(r.metadata.get("file_index", 0))):
        filtered = _maybe_lowpass(np.asarray(record.force_v, dtype=float), float(record.sample_rate_hz))
        rec = TraceRecord(
            file_path=record.file_path,
            time_s=record.time_s,
            force_v=filtered,
            trigger_v=record.trigger_v,
            sample_rate_hz=record.sample_rate_hz,
            metadata=record.metadata,
        )
        bursts, flags = compute_burst_metrics(
            time_s=rec.time_s,
            force_v=rec.force_v,
            trigger_v=rec.trigger_v,
            sample_rate_hz=rec.sample_rate_hz,
            metadata=rec.metadata,
            settings=settings,
            window_seconds=6.0,
        )
        amp_values = [float(b["metrics"].get("amplitude_cn", 0.0)) for b in bursts]
        mean_amp = float(np.mean(amp_values)) if amp_values else 0.0
        metrics = {
            "n_bursts": len(bursts),
            "mean_amplitude_cn": mean_amp,
            "temperature": record.metadata.get("temperature"),
            "condition": record.metadata.get("condition"),
            "stim_index": record.metadata.get("stim_index"),
        }
        output["files"].append(
            {
                "file_path": str(record.file_path),
                "file_index": int(record.metadata.get("file_index", 0)),
                "metrics": metrics,
                "bursts": bursts,
                "flags": flags,
            }
        )
        cond = str(record.metadata.get("condition", "unknown"))
        grouped.setdefault(cond, []).append(mean_amp)
        amplitudes.append(mean_amp)
        for flag in flags:
            output["flags"].append({"file_path": str(record.file_path), **flag})

    output["summary"] = {
        "n_files": len(output["files"]),
        "mean_amplitude_cn": float(np.mean(amplitudes)) if amplitudes else 0.0,
        "stats_by_condition": compute_stat_markers(grouped),
    }
    return output
