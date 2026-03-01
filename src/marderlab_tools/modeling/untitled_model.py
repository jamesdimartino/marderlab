from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class UntitledParams:
    duration_s: float = 10.0
    dt_s: float = 0.0005
    temperature_c: float = 12.0
    input_current: float = 0.8


def run_untitled_model(params: UntitledParams) -> dict[str, Any]:
    t = np.arange(0.0, float(params.duration_s), float(params.dt_s), dtype=float)
    if t.size == 0:
        return {"time_s": np.array([]), "voltage_v": np.array([]), "summary": {}}
    q10 = 2.0 ** ((float(params.temperature_c) - 12.0) / 10.0)
    tau = 0.02 / max(q10, 1e-8)
    v = np.zeros_like(t)
    v[0] = -0.06
    i_inj = np.full_like(t, float(params.input_current))

    for i in range(1, t.size):
        dv = (-(v[i - 1] + 0.06) + 0.08 * np.tanh(10.0 * i_inj[i - 1])) / tau
        v[i] = v[i - 1] + dv * float(params.dt_s)

    summary = {
        "temperature_c": float(params.temperature_c),
        "q10": float(q10),
        "mean_voltage_v": float(np.nanmean(v)),
        "peak_voltage_v": float(np.nanmax(v)),
        "min_voltage_v": float(np.nanmin(v)),
    }
    return {"time_s": t, "voltage_v": v, "input_current": i_inj, "summary": summary}
