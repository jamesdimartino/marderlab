from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marderlab_tools.analysis.hikcontrol import TraceRecord as HikTraceRecord
from marderlab_tools.analysis.hikcontrol import analyze_experiment as analyze_hikcontrol_experiment
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
    typed = [HikTraceRecord(**record.__dict__) for record in records]
    output = analyze_hikcontrol_experiment(typed, settings)
    output["pipeline"] = "dualhik"

    by_condition: dict[str, list[float]] = {}
    by_temperature: dict[str, list[float]] = {}
    for item in output.get("files", []):
        metrics = item.get("metrics", {})
        amp = float(metrics.get("amplitude_cn", 0.0))
        if not np.isfinite(amp):
            continue
        record = next((r for r in records if str(r.file_path) == item.get("file_path")), None)
        if record is None:
            continue
        condition = str(record.metadata.get("condition", "unknown"))
        by_condition.setdefault(condition, []).append(amp)
        temp = record.metadata.get("temperature")
        if temp is not None:
            by_temperature.setdefault(str(temp), []).append(amp)

    output.setdefault("summary", {})
    output["summary"]["stats_by_condition"] = compute_stat_markers(by_condition) if by_condition else {
        "rule": "by_group_count",
        "test": "insufficient_groups",
        "p_value": None,
        "stars": "ns",
        "pairwise": [],
    }
    output["summary"]["stats_by_temperature"] = compute_stat_markers(by_temperature) if by_temperature else {
        "rule": "by_group_count",
        "test": "insufficient_groups",
        "p_value": None,
        "stars": "ns",
        "pairwise": [],
    }
    return output
