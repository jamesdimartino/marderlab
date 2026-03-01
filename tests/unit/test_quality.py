import numpy as np

from marderlab_tools.preprocess.quality import assess_signal, zero_metrics


def test_assess_signal_flags_low_variance() -> None:
    signal = np.ones(100)
    assessment = assess_signal(signal, std_floor=1e-6, clip_abs=10.0)
    assert assessment.poor_signal
    assert any(flag.code == "low_variance" for flag in assessment.flags)


def test_zero_metrics_sets_all_to_zero() -> None:
    metrics = {"peak": 1.2, "auc": 3.4}
    zeroed, flag = zero_metrics(metrics, "poor")
    assert zeroed == {"peak": 0.0, "auc": 0.0}
    assert flag.code == "poor_signal_zeroed"
