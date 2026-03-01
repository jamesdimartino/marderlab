import numpy as np

from marderlab_tools.preprocess.baseline import compute_baseline, find_trigger_start


def test_compute_baseline_pre_stim_window() -> None:
    trace = np.arange(1000, dtype=float)
    baseline = compute_baseline(trace, sample_rate_hz=100.0, stim_start_idx=500, baseline_seconds=2.0)
    assert abs(baseline - 399.5) < 1e-9


def test_find_trigger_start() -> None:
    trigger = np.array([0, 0, 0.1, 0.2, 1.0, 1.2], dtype=float)
    assert find_trigger_start(trigger, threshold_ratio=0.5) == 4
