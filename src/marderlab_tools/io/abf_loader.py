from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marderlab_tools.config.schema import ChannelMap


@dataclass
class LoadedTrace:
    file_path: Path
    sample_rate_hz: float
    time_s: np.ndarray
    force_v: np.ndarray
    trigger_v: np.ndarray
    channel_names: list[str]


def _read_abf(file_path: Path) -> Any:
    try:
        import pyabf  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency-dependent
        raise RuntimeError(
            "pyabf is required to read .abf files. Install with: pip install pyabf"
        ) from exc
    return pyabf.ABF(str(file_path))


def _resolve_channel_index(adc_names: list[str], target: str, fallback: int) -> int:
    for idx, name in enumerate(adc_names):
        if str(name).strip().lower() == target.strip().lower():
            return idx
    return fallback


def load_force_trigger(file_path: Path, channel_map: ChannelMap) -> LoadedTrace:
    abf = _read_abf(file_path)
    adc_names = [str(name) for name in getattr(abf, "adcNames", [])]
    sample_rate = float(getattr(abf, "dataRate", 0.0) or 0.0)

    force_idx = _resolve_channel_index(adc_names, channel_map.force, 0)
    trig_idx = _resolve_channel_index(adc_names, channel_map.trigger, min(1, max(0, len(adc_names) - 1)))

    abf.setSweep(sweepNumber=0, channel=force_idx)
    force = np.asarray(abf.sweepY, dtype=float)
    time_s = np.asarray(abf.sweepX, dtype=float)

    abf.setSweep(sweepNumber=0, channel=trig_idx)
    trigger = np.asarray(abf.sweepY, dtype=float)

    if sample_rate <= 0 and len(time_s) > 1:
        dt = float(time_s[1] - time_s[0])
        sample_rate = 1.0 / dt if dt > 0 else 0.0

    return LoadedTrace(
        file_path=file_path,
        sample_rate_hz=sample_rate,
        time_s=time_s,
        force_v=force,
        trigger_v=trigger,
        channel_names=adc_names,
    )
