from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class HiKSimParams:
    duration_s: float = 300.0
    dt_s: float = 0.01
    temperature_c: float = 12.0
    stimulus_start_s: float = 30.0
    stimulus_end_s: float = 180.0
    stimulus_amp: float = 1.0


def run_hiksim(params: HiKSimParams) -> dict[str, Any]:
    t = np.arange(0.0, float(params.duration_s), float(params.dt_s), dtype=float)
    if t.size == 0:
        return {"time_s": np.array([]), "vm": np.array([]), "w": np.array([]), "ca": np.array([]), "summary": {}}

    temp_scale = 1.0 + 0.03 * (float(params.temperature_c) - 12.0)
    stim = np.zeros_like(t)
    m = (t >= float(params.stimulus_start_s)) & (t <= float(params.stimulus_end_s))
    stim[m] = float(params.stimulus_amp)

    vm = np.zeros_like(t)
    w = np.zeros_like(t)
    ca = np.zeros_like(t)
    vm[0] = -0.45
    w[0] = 0.15
    ca[0] = 0.02

    for i in range(1, t.size):
        dv = (-0.15 * vm[i - 1] - 0.6 * w[i - 1] + 1.2 * stim[i - 1]) * temp_scale
        dw = (0.25 * (vm[i - 1] + 0.5) - w[i - 1]) * (0.2 * temp_scale)
        dca = (0.5 * max(vm[i - 1], 0.0) - 0.1 * ca[i - 1]) * (0.12 * temp_scale)
        vm[i] = vm[i - 1] + dv * float(params.dt_s)
        w[i] = w[i - 1] + dw * float(params.dt_s)
        ca[i] = ca[i - 1] + dca * float(params.dt_s)

    summary = {
        "temperature_c": float(params.temperature_c),
        "peak_vm": float(np.nanmax(vm)),
        "min_vm": float(np.nanmin(vm)),
        "peak_ca": float(np.nanmax(ca)),
        "mean_vm": float(np.nanmean(vm)),
    }
    return {"time_s": t, "vm": vm, "w": w, "ca": ca, "stimulus": stim, "summary": summary}
