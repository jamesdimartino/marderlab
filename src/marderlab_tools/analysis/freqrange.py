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


def analyze_experiment(records: list[TraceRecord], settings: PipelineSettings) -> dict[str, Any]:
    output: dict[str, Any] = {"pipeline": "freqrange", "files": [], "summary": {}, "flags": []}
    group_by_stim: dict[str, list[float]] = {}
    all_amps: list[float] = []

    for record in sorted(records, key=lambda r: int(r.metadata.get("file_index", 0))):
        bursts, flags = compute_burst_metrics(
            time_s=record.time_s,
            force_v=record.force_v,
            trigger_v=record.trigger_v,
            sample_rate_hz=record.sample_rate_hz,
            metadata=record.metadata,
            settings=settings,
            window_seconds=8.0,
        )
        best = max((b["metrics"] for b in bursts), key=lambda m: float(m.get("amplitude_cn", 0.0)), default=None)
        metrics = best or {
            "amplitude_cn": 0.0,
            "latency_s": 0.0,
            "slope_cn_per_s": 0.0,
            "auc_cn_s": 0.0,
            "stim_index": record.metadata.get("stim_index"),
        }
        stim = str(metrics.get("stim_index", "unknown"))
        amp = float(metrics.get("amplitude_cn", 0.0))
        group_by_stim.setdefault(stim, []).append(amp)
        all_amps.append(amp)

        output["files"].append(
            {
                "file_path": str(record.file_path),
                "file_index": int(record.metadata.get("file_index", 0)),
                "metrics": metrics,
                "flags": flags,
                "bursts": bursts,
            }
        )
        for flag in flags:
            output["flags"].append({"file_path": str(record.file_path), **flag})

    output["summary"] = {
        "n_files": len(output["files"]),
        "mean_amplitude_cn": float(np.mean(all_amps)) if all_amps else 0.0,
        "stats_by_stim_index": compute_stat_markers(group_by_stim),
    }
    return output
