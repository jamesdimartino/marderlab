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


def _find_burst_metric(bursts: list[dict[str, Any]], burst_index: int, metric_name: str) -> float:
    for burst in bursts:
        metrics = burst.get("metrics", {})
        if int(metrics.get("burst_index", -1)) == int(burst_index):
            return float(metrics.get(metric_name, 0.0))
    return 0.0


def analyze_experiment(records: list[TraceRecord], settings: PipelineSettings) -> dict[str, Any]:
    output: dict[str, Any] = {"pipeline": "gm56weaklink", "files": [], "summary": {}, "flags": []}
    weak_ratio_groups: dict[str, list[float]] = {}
    all_ratios: list[float] = []

    for record in sorted(records, key=lambda r: int(r.metadata.get("file_index", 0))):
        bursts, flags = compute_burst_metrics(
            time_s=record.time_s,
            force_v=record.force_v,
            trigger_v=record.trigger_v,
            sample_rate_hz=record.sample_rate_hz,
            metadata=record.metadata,
            settings=settings,
            window_seconds=10.0,
        )
        amp8 = _find_burst_metric(bursts, 8, "amplitude_cn")
        amp9 = _find_burst_metric(bursts, 9, "amplitude_cn")
        auc8 = _find_burst_metric(bursts, 8, "auc_cn_s")
        lat8 = _find_burst_metric(bursts, 8, "latency_s")
        slope8 = _find_burst_metric(bursts, 8, "slope_cn_per_s")
        ratio = float(amp9 / amp8) if amp8 > 0 else 0.0

        metrics = {
            "amp_burst8_cn": amp8,
            "amp_burst9_cn": amp9,
            "weaklink_ratio_b9_b8": ratio,
            "auc_burst8_cn_s": auc8,
            "latency_burst8_s": lat8,
            "slope_burst8_cn_per_s": slope8,
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
        weak_ratio_groups.setdefault(cond, []).append(ratio)
        all_ratios.append(ratio)
        for flag in flags:
            output["flags"].append({"file_path": str(record.file_path), **flag})

    output["summary"] = {
        "n_files": len(output["files"]),
        "mean_weaklink_ratio_b9_b8": float(np.mean(all_ratios)) if all_ratios else 0.0,
        "stats_weaklink_ratio_by_condition": compute_stat_markers(weak_ratio_groups),
    }
    return output
