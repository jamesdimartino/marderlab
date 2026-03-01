from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FiberParams:
    duration_s: float = 20.0
    dt_s: float = 0.001
    g_peak: float = 0.8
    stim_hz: float = 2.0
    temperature_c: float = 12.0


def run_modelfiber(params: FiberParams) -> dict[str, Any]:
    t = np.arange(0.0, float(params.duration_s), float(params.dt_s), dtype=float)
    if t.size == 0:
        return {"time_s": np.array([]), "ejp": np.array([]), "spikes": np.array([]), "summary": {}}

    temp_scale = 1.0 + 0.02 * (float(params.temperature_c) - 12.0)
    spikes = np.zeros_like(t)
    if params.stim_hz > 0:
        period = max(1, int(round((1.0 / float(params.stim_hz)) / float(params.dt_s))))
        spikes[::period] = 1.0

    kernel_t = np.arange(0.0, 1.0, float(params.dt_s), dtype=float)
    tau = 0.08 / max(1e-6, temp_scale)
    kernel = np.exp(-kernel_t / tau)
    kernel /= np.sum(kernel) if np.sum(kernel) > 0 else 1.0

    ejp = np.convolve(spikes, kernel, mode="full")[: t.size] * float(params.g_peak) * temp_scale
    summary = {
        "peak_ejp": float(np.nanmax(ejp)) if ejp.size else 0.0,
        "mean_ejp": float(np.nanmean(ejp)) if ejp.size else 0.0,
        "n_spikes": int(np.sum(spikes > 0)),
        "temperature_c": float(params.temperature_c),
    }
    return {"time_s": t, "ejp": ejp, "spikes": spikes, "summary": summary}
