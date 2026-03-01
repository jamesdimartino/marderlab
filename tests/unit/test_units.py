import numpy as np

from marderlab_tools.preprocess.units import calibration_from_season, seconds_to_minutes, volts_to_centinewtons


def test_calibration_fallback_rules() -> None:
    assert calibration_from_season(10, None) == 0.3
    assert calibration_from_season(25, None) == 0.3
    assert calibration_from_season(26, None) == 0.35
    assert calibration_from_season(None, None) == 0.35
    assert calibration_from_season(15, 0.42) == 0.42


def test_unit_conversions() -> None:
    signal = np.array([0.0, 1.0, 2.0])
    assert np.allclose(volts_to_centinewtons(signal, 0.5), np.array([0.0, 20.0, 40.0]))
    assert np.allclose(seconds_to_minutes(np.array([0.0, 60.0])), np.array([0.0, 1.0]))
