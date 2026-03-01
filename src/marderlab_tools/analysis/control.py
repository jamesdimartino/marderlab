from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marderlab_tools.analysis.hikcontrol import analyze_experiment as analyze_hikcontrol_experiment
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
    output = analyze_hikcontrol_experiment(records, settings)
    output["pipeline"] = "control"
    return output
