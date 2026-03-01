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


def _group_key(meta: dict[str, Any]) -> str:
    cond = str(meta.get("condition", "unknown"))
    temp = meta.get("temperature")
    return f"{cond}@{temp}"


def analyze_experiment(records: list[TraceRecord], settings: PipelineSettings) -> dict[str, Any]:
    output: dict[str, Any] = {"pipeline": "gm56acclim", "files": [], "summary": {}, "flags": []}
    grouped: dict[str, list[float]] = {}
    all_amps: list[float] = []

    for record in sorted(records, key=lambda r: int(r.metadata.get("file_index", 0))):
        bursts, trace_flags = compute_burst_metrics(
            time_s=record.time_s,
            force_v=record.force_v,
            trigger_v=record.trigger_v,
            sample_rate_hz=record.sample_rate_hz,
            metadata=record.metadata,
            settings=settings,
            window_seconds=12.0,
        )
        burst_metrics = [b["metrics"] for b in bursts]
        amp_values = [float(m.get("amplitude_cn", 0.0)) for m in burst_metrics]
        file_metrics = {
            "n_bursts": len(burst_metrics),
            "mean_amplitude_cn": float(np.mean(amp_values)) if amp_values else 0.0,
            "max_amplitude_cn": float(np.max(amp_values)) if amp_values else 0.0,
            "mean_latency_s": float(np.mean([float(m.get("latency_s", 0.0)) for m in burst_metrics]))
            if burst_metrics
            else 0.0,
            "mean_auc_cn_s": float(np.mean([float(m.get("auc_cn_s", 0.0)) for m in burst_metrics]))
            if burst_metrics
            else 0.0,
            "temperature": record.metadata.get("temperature"),
            "condition": record.metadata.get("condition"),
            "stim_index": record.metadata.get("stim_index"),
        }

        all_amps.append(float(file_metrics["mean_amplitude_cn"]))
        grouped.setdefault(_group_key(record.metadata), []).append(float(file_metrics["mean_amplitude_cn"]))
        output["files"].append(
            {
                "file_path": str(record.file_path),
                "file_index": int(record.metadata.get("file_index", 0)),
                "metrics": file_metrics,
                "bursts": bursts,
                "flags": trace_flags,
            }
        )
        for flag in trace_flags:
            output["flags"].append({"file_path": str(record.file_path), **flag})

    output["summary"] = {
        "n_files": len(output["files"]),
        "mean_amplitude_cn": float(np.mean(all_amps)) if all_amps else 0.0,
        "stats_by_condition_temp": compute_stat_markers(grouped),
    }
    return output
