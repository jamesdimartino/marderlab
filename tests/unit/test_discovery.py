from pathlib import Path

from marderlab_tools.io.experiment_discovery import discover_experiments


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_discover_experiments_groups_and_sorts(tmp_path: Path) -> None:
    exp_dir = tmp_path / "997_001"
    _touch(exp_dir / "997_001_0003.abf")
    _touch(exp_dir / "997_001_0001.abf")
    _touch(exp_dir / "997_001_0002.abf")
    _touch(exp_dir / "bad_name.abf")
    _touch(tmp_path / "998_000" / "998_000_0001.abf")

    experiments = discover_experiments(tmp_path)
    assert sorted(experiments.keys()) == ["997_001", "998_000"]
    assert [p.name for p in experiments["997_001"]] == [
        "997_001_0001.abf",
        "997_001_0002.abf",
        "997_001_0003.abf",
    ]
