from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marderlab_tools.analysis.heartbeat_common import analyze_heartbeat_trace
from marderlab_tools.config.schema import PipelineSettings


@dataclass
class TraceRecord:
    file_path: Path
    time_s: np.ndarray
    force_v: np.ndarray
    trigger_v: np.ndarray
    sample_rate_hz: float
    metadata: dict[str, Any]


def analyze_experiment(records: list[TraceRecord], settings: PipelineSettings) -> dict[str, Any]:
    output: dict[str, Any] = {"pipeline": "rawheart", "files": [], "summary": {}, "flags": []}
    rates: list[float] = []
    for record in sorted(records, key=lambda r: int(r.metadata.get("file_index", 0))):
        metrics, flags = analyze_heartbeat_trace(
            time_s=record.time_s,
            force_v=record.force_v,
            sample_rate_hz=record.sample_rate_hz,
            metadata=record.metadata,
            settings=settings,
        )
        output["files"].append(
            {
                "file_path": str(record.file_path),
                "file_index": int(record.metadata.get("file_index", 0)),
                "metrics": metrics,
                "flags": flags,
            }
        )
        rates.append(float(metrics.get("heart_rate_bpm", 0.0)))
        for flag in flags:
            output["flags"].append({"file_path": str(record.file_path), **flag})
    output["summary"] = {
        "n_files": len(output["files"]),
        "mean_heart_rate_bpm": float(np.mean(rates)) if rates else 0.0,
    }
    return output
