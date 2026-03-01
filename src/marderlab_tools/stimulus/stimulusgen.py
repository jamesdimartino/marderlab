from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class StimulusSpec:
    duration_s: float = 60.0
    sample_rate_hz: float = 10000.0
    burst_count: int = 10
    burst_width_s: float = 0.08
    burst_amplitude_v: float = 5.0
    start_delay_s: float = 2.0
    inter_burst_s: float = 4.0


def generate_burst_train(spec: StimulusSpec) -> dict[str, Any]:
    n = max(1, int(round(float(spec.duration_s) * float(spec.sample_rate_hz))))
    t = np.arange(n, dtype=float) / float(spec.sample_rate_hz)
    y = np.zeros(n, dtype=float)

    starts: list[float] = []
    for i in range(max(0, int(spec.burst_count))):
        start_s = float(spec.start_delay_s) + (i * float(spec.inter_burst_s))
        end_s = start_s + float(spec.burst_width_s)
        if start_s >= float(spec.duration_s):
            break
        starts.append(start_s)
        mask = (t >= start_s) & (t < end_s)
        y[mask] = float(spec.burst_amplitude_v)

    return {"time_s": t, "stimulus_v": y, "burst_starts_s": starts, "spec": asdict(spec)}


def write_stimulus_file(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    time_s = np.asarray(payload.get("time_s", []), dtype=float)
    stimulus_v = np.asarray(payload.get("stimulus_v", []), dtype=float)
    data = np.column_stack([time_s, stimulus_v]) if time_s.size and stimulus_v.size else np.empty((0, 2))
    header = "time_s,stimulus_v"
    np.savetxt(out, data, delimiter=",", header=header, comments="")
    return out
