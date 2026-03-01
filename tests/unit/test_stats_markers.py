from marderlab_tools.stats.markers import compute_stat_markers


def test_two_group_stats_shape() -> None:
    out = compute_stat_markers({"a": [1, 2, 3], "b": [2, 3, 4]})
    assert out["rule"] == "by_group_count"
    assert "test" in out
    assert "p_value" in out
    assert "stars" in out


def test_multi_group_stats_shape() -> None:
    out = compute_stat_markers({"a": [1, 2, 3], "b": [2, 3, 4], "c": [4, 5, 6]})
    assert out["rule"] == "by_group_count"
    assert out["test"] == "anova_oneway"
    assert "pairwise" in out
