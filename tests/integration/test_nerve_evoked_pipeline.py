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
            "nerve_evoked": {
                "experiment_type_values": ["nerve_evoked"],
                "force_channel": "force",
                "trig_channel": "trig",
            }
        },
        "checks": {"fail_mode": "experiment_only"},
        "stats": {"default_rule": "by_group_count"},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_nerve_evoked_pipeline_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    exp = raw / "997_002"
    exp.mkdir(parents=True)
    (exp / "997_002_0001.abf").write_bytes(b"")

    cfg_path = tmp_path / "config.yml"
    _write_config(cfg_path, raw, processed, cache)
    config = load_config(cfg_path)

    metadata = pd.DataFrame(
        {
            "notebook_page": ["997_002"],
            "file_index": [1],
            "stim_index": [1],
            "temperature": [12.0],
            "condition": ["control"],
            "experiment_type": ["nerve_evoked"],
            "season": [26],
            "calibration": [0.35],
        }
    )

    def fake_load_metadata(_cfg):
        return metadata, True, "metadata_source=cache_csv"

    def fake_abf_loader(file_path, _channel_map):
        n = 600
        t = np.linspace(0, 6, n)
        trigger = np.zeros(n)
        trigger[250:255] = 1.0
        force = np.sin(t * 2) + np.linspace(0, 0.5, n)
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

    report = orchestrator.run_pipeline(config, "nerve-evoked", generate_plots=False)
    assert report["summary"]["total_experiments"] == 1
    assert report["summary"]["success_count"] == 1
    npy_file = processed / "997_002" / "npy" / "nerve_evoked_metrics.npy"
    assert npy_file.exists()
