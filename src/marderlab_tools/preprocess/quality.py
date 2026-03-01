from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from marderlab_tools.config.schema import QualityFlag


@dataclass
class QualityAssessment:
    poor_signal: bool
    flags: list[QualityFlag]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [asdict(flag) for flag in self.flags]


def assess_signal(
    trace: np.ndarray,
    std_floor: float,
    clip_abs: float,
) -> QualityAssessment:
    flags: list[QualityFlag] = []

    if trace.size == 0:
        flags.append(QualityFlag(code="empty_trace", message="Trace is empty.", severity="error"))
        return QualityAssessment(poor_signal=True, flags=flags)

    if not np.isfinite(trace).all():
        flags.append(
            QualityFlag(
                code="non_finite",
                message="Trace contains non-finite values.",
                severity="error",
            )
        )

    if float(np.nanstd(trace)) < float(std_floor):
        flags.append(
            QualityFlag(
                code="low_variance",
                message=f"Trace standard deviation below threshold ({std_floor}).",
                severity="warning",
            )
        )

    if float(np.nanmax(np.abs(trace))) >= float(clip_abs):
        flags.append(
            QualityFlag(
                code="clipping",
                message=f"Trace exceeds clipping threshold ({clip_abs}).",
                severity="warning",
            )
        )

    return QualityAssessment(poor_signal=len(flags) > 0, flags=flags)


def zero_metrics(metrics: dict[str, float], reason: str) -> tuple[dict[str, float], QualityFlag]:
    zeroed = {key: 0.0 for key in metrics}
    return zeroed, QualityFlag(code="poor_signal_zeroed", message=reason, severity="warning")
