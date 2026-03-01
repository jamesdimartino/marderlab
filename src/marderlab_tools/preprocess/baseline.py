from __future__ import annotations

import numpy as np


def find_trigger_start(trigger: np.ndarray, threshold_ratio: float = 0.5) -> int | None:
    if trigger.size == 0:
        return None
    trigger_max = float(np.nanmax(trigger))
    if trigger_max <= 0:
        return None
    threshold = trigger_max * threshold_ratio
    idx = np.where(trigger >= threshold)[0]
    if idx.size == 0:
        return None
    return int(idx[0])


def compute_baseline(
    trace: np.ndarray,
    sample_rate_hz: float,
    stim_start_idx: int | None,
    baseline_seconds: float = 2.0,
) -> float:
    if trace.size == 0:
        return 0.0
    if sample_rate_hz <= 0:
        sample_rate_hz = float(trace.size)
    window = max(1, int(round(sample_rate_hz * baseline_seconds)))

    if stim_start_idx is None:
        end = min(trace.size, window)
    else:
        end = max(1, min(trace.size, stim_start_idx))
    start = max(0, end - window)
    baseline_window = trace[start:end]
    if baseline_window.size == 0:
        return 0.0
    return float(np.nanmean(baseline_window))


def subtract_baseline(trace: np.ndarray, baseline: float) -> np.ndarray:
    return np.asarray(trace, dtype=float) - float(baseline)
