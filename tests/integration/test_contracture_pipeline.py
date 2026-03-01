from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from marderlab_tools.config.schema import load_config
from marderlab_tools.run import orchestrator


@dataclass
class FakeTrace:
    file_path: Path
    sample_rate_hz: float
    time_s: np.ndarray
    force_v: np.ndarray
    trigger_v: np.ndarray
    channel_names: list[str]


def _write_config(path: Path, raw_root: Path, processed_root: Path, cache_root: Path) -> None:
    payload = {
        "paths": {
            "raw_data_root": str(raw_root),
            "processed_root": str(processed_root),
            "cache_root": str(cache_root),
        },
        "metadata": {
            "google_sheet_url": "https://example.com/sheet/pubhtml",
            "tabs": ["FTBath"],
            "cache_csv": str(cache_root / "metadata.csv"),
            "required_fields": [
                "notebook_page",
                "file_index",
                "stim_index",
                "temperature",
                "condition",
                "experiment_type",
                "season",
            ],
            "column_map": {},
        },
        "pipelines": {
            "contracture": {
                "experiment_type_values": ["contracture"],
                "force_channel": "force",
                "trig_channel": "trig",
            }
        },
        "checks": {"fail_mode": "experiment_only"},
        "stats": {"default_rule": "by_group_count"},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_contracture_pipeline_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    exp = raw / "997_001"
    exp.mkdir(parents=True)
    (exp / "997_001_0001.abf").write_bytes(b"")
    (exp / "997_001_0002.abf").write_bytes(b"")

    cfg_path = tmp_path / "config.yml"
    _write_config(cfg_path, raw, processed, cache)
    config = load_config(cfg_path)

    metadata = pd.DataFrame(
        {
            "notebook_page": ["997_001", "997_001"],
            "file_index": [1, 2],
            "stim_index": [1, 1],
            "temperature": [10.0, 10.0],
            "condition": ["control", "control"],
            "experiment_type": ["contracture", "contracture"],
            "season": [20, 20],
            "calibration": [0.3, 0.3],
        }
    )

    def fake_load_metadata(_cfg):
        return metadata, True, "metadata_source=cache_csv"

    def fake_abf_loader(file_path, _channel_map):
        n = 500
        t = np.linspace(0, 5, n)
        trigger = np.zeros(n)
        trigger[200:210] = 1.0
        force = np.sin(t) + 0.1
        return FakeTrace(
            file_path=file_path,
            sample_rate_hz=100.0,
            time_s=t,
            force_v=force,
            trigger_v=trigger,
            channel_names=["force", "trig"],
        )

    monkeypatch.setattr(orchestrator, "load_metadata_with_fallback", fake_load_metadata)
    monkeypatch.setattr(orchestrator, "load_force_trigger", fake_abf_loader)

    report = orchestrator.run_pipeline(config, "contracture", generate_plots=False)
    assert report["summary"]["total_experiments"] == 1
    assert report["summary"]["success_count"] == 1
    npy_file = processed / "997_001" / "npy" / "contracture_metrics.npy"
    tidy_csv = processed / "997_001" / "npy" / "contracture_metrics_tidy.csv"
    assert npy_file.exists()
    assert tidy_csv.exists()
    tidy_df = pd.read_csv(tidy_csv)
    required_cols = {
        "notebook_page",
        "pipeline",
        "file_index",
        "temperature",
        "condition",
        "stim_index",
        "metric_name",
        "metric_value",
        "metric_unit",
    }
    assert required_cols.issubset(set(tidy_df.columns))
    assert (tidy_df["pipeline"] == "contracture").all()
