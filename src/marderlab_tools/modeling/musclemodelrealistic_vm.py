from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MuscleVMParams:
    duration_s: float = 240.0
    dt_s: float = 0.005
    temperatures_c: tuple[float, ...] = (10.0, 12.0, 16.0, 20.0)
    stim_amp: float = 1.0
    stim_width_s: float = 0.15
    stim_period_s: float = 2.0


def _make_stimulus(t: np.ndarray, amp: float, width_s: float, period_s: float) -> np.ndarray:
    if period_s <= 0 or width_s <= 0:
        return np.zeros_like(t)
    phase = np.mod(t, period_s)
    return np.where(phase <= width_s, amp, 0.0)


def _simulate_one_temp(t: np.ndarray, temp_c: float, stim: np.ndarray, dt_s: float) -> dict[str, np.ndarray]:
    scale = 1.0 + 0.025 * (temp_c - 12.0)
    vm = np.zeros_like(t)
    ca = np.zeros_like(t)
    m = np.zeros_like(t)
    vm[0] = -0.5
    for i in range(1, t.size):
        m_inf = 1.0 / (1.0 + np.exp(-(vm[i - 1] + 0.25) * 12.0))
        tau_m = 0.08 / max(scale, 1e-6)
        m[i] = m[i - 1] + (m_inf - m[i - 1]) * (dt_s / tau_m)
        dv = (-0.18 * vm[i - 1] + 0.7 * m[i] + 1.3 * stim[i - 1] - 0.4 * ca[i - 1]) * scale
        dca = (0.9 * max(vm[i - 1], 0.0) - 0.15 * ca[i - 1]) * (0.18 * scale)
        vm[i] = vm[i - 1] + dv * dt_s
        ca[i] = ca[i - 1] + dca * dt_s
    return {"vm": vm, "ca": ca, "m": m}


def run_musclemodelrealistic_vm(params: MuscleVMParams) -> dict[str, Any]:
    t = np.arange(0.0, float(params.duration_s), float(params.dt_s), dtype=float)
    stim = _make_stimulus(t, float(params.stim_amp), float(params.stim_width_s), float(params.stim_period_s))
    traces: dict[str, dict[str, np.ndarray]] = {}
    summary: dict[str, Any] = {"temperatures": []}
    for temp in params.temperatures_c:
        sim = _simulate_one_temp(t, float(temp), stim, float(params.dt_s))
        key = f"{float(temp):.1f}C"
        traces[key] = sim
        summary["temperatures"].append(
            {
                "temperature_c": float(temp),
                "peak_vm": float(np.nanmax(sim["vm"])) if sim["vm"].size else 0.0,
                "peak_ca": float(np.nanmax(sim["ca"])) if sim["ca"].size else 0.0,
            }
        )
    return {"time_s": t, "stimulus": stim, "traces": traces, "summary": summary}
