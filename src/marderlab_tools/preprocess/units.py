from __future__ import annotations

import numpy as np


def calibration_from_season(season: int | float | str | None, explicit: float | None = None) -> float:
    if explicit is not None:
        return float(explicit)

    try:
        season_value = float(season) if season is not None else float("nan")
    except (TypeError, ValueError):
        season_value = float("nan")

    if np.isfinite(season_value) and 10 <= season_value <= 25:
        return 0.3
    return 0.35


def volts_to_centinewtons(force_volts: np.ndarray, calibration: float) -> np.ndarray:
    cal = float(calibration)
    if cal == 0:
        return np.zeros_like(np.asarray(force_volts, dtype=float))
    # Calibration is interpreted as volts per 10 cN.
    return (np.asarray(force_volts, dtype=float) / cal) * 10.0


def seconds_to_minutes(time_seconds: np.ndarray) -> np.ndarray:
    return np.asarray(time_seconds, dtype=float) / 60.0
