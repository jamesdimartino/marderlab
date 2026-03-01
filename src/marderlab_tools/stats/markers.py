from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def stars_for_pvalue(p_value: float | None) -> str:
    if p_value is None or not np.isfinite(p_value):
        return "ns"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _safe_float_list(values: Iterable[float]) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            num = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(num):
            out.append(num)
    return out


def _welch_t(a: list[float], b: list[float]) -> tuple[float | None, str]:
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return None, "welch_t_unavailable"
    if len(a) < 2 or len(b) < 2:
        return None, "welch_t"
    _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(p), "welch_t"


def _anova_and_posthoc(groups: dict[str, list[float]]) -> tuple[float | None, list[dict[str, Any]]]:
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return None, []

    values = [v for v in groups.values() if len(v) >= 2]
    if len(values) < 2:
        return None, []

    _, p_omnibus = stats.f_oneway(*values)
    names = list(groups.keys())
    pairwise: list[dict[str, Any]] = []
    denom = max(1, len(names) * (len(names) - 1) // 2)

    for idx_a in range(len(names)):
        for idx_b in range(idx_a + 1, len(names)):
            a = groups[names[idx_a]]
            b = groups[names[idx_b]]
            if len(a) < 2 or len(b) < 2:
                p_raw = None
            else:
                _, p_raw = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            p_adj = min(1.0, float(p_raw) * denom) if p_raw is not None else None
            pairwise.append(
                {
                    "group_a": names[idx_a],
                    "group_b": names[idx_b],
                    "test": "welch_t_bonferroni",
                    "p_value": p_adj,
                    "stars": stars_for_pvalue(p_adj),
                }
            )
    return float(p_omnibus), pairwise


def compute_stat_markers(group_values: dict[str, Iterable[float]]) -> dict[str, Any]:
    cleaned = {name: _safe_float_list(vals) for name, vals in group_values.items()}
    cleaned = {k: v for k, v in cleaned.items() if v}

    if len(cleaned) < 2:
        return {
            "rule": "by_group_count",
            "test": "insufficient_groups",
            "p_value": None,
            "stars": "ns",
            "pairwise": [],
        }

    if len(cleaned) == 2:
        names = list(cleaned.keys())
        p_value, test_name = _welch_t(cleaned[names[0]], cleaned[names[1]])
        return {
            "rule": "by_group_count",
            "test": test_name,
            "groups": names,
            "p_value": p_value,
            "stars": stars_for_pvalue(p_value),
            "pairwise": [],
        }

    p_omnibus, pairwise = _anova_and_posthoc(cleaned)
    return {
        "rule": "by_group_count",
        "test": "anova_oneway",
        "p_value": p_omnibus,
        "stars": stars_for_pvalue(p_omnibus),
        "pairwise": pairwise,
    }
