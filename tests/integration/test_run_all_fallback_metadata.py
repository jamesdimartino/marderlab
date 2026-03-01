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
            },
            "nerve_evoked": {
                "experiment_type_values": ["nerve_evoked"],
                "force_channel": "force",
                "trig_channel": "trig",
            },
        },
        "checks": {"fail_mode": "experiment_only"},
        "stats": {"default_rule": "by_group_count"},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_run_all_uses_cache_fallback_when_sync_fails(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    (raw / "997_010").mkdir(parents=True)
    (raw / "997_010" / "997_010_0001.abf").write_bytes(b"")
    (raw / "997_011").mkdir(parents=True)
    (raw / "997_011" / "997_011_0001.abf").write_bytes(b"")

    cfg_path = tmp_path / "config.yml"
    _write_config(cfg_path, raw, processed, cache)
    config = load_config(cfg_path)

    cache.mkdir(parents=True, exist_ok=True)
    cached = pd.DataFrame(
        {
            "notebook_page": ["997_010", "997_011"],
            "file_index": [1, 1],
            "stim_index": [1, 1],
            "temperature": [10.0, 11.0],
            "condition": ["control", "control"],
            "experiment_type": ["contracture", "nerve_evoked"],
            "season": [20, 26],
            "calibration": [0.3, 0.35],
        }
    )
    cached.to_csv(config.metadata.cache_csv, index=False)

    def fake_sync_metadata(_cfg):
        raise RuntimeError("network down")

    def fake_abf_loader(file_path, _channel_map):
        n = 400
        t = np.linspace(0, 4, n)
        trigger = np.zeros(n)
        trigger[100:110] = 1.0
        force = np.cos(t) + 0.2
        return FakeTrace(
            file_path=file_path,
            sample_rate_hz=100.0,
            time_s=t,
            force_v=force,
            trigger_v=trigger,
            channel_names=["force", "trig"],
        )

    monkeypatch.setattr(orchestrator, "sync_metadata", fake_sync_metadata)
    monkeypatch.setattr(orchestrator, "load_force_trigger", fake_abf_loader)

    report = orchestrator.run_all(config, generate_plots=False)
    assert report["summary"]["total_experiments"] == 2
    assert "metadata_source=cache_csv" in report["metadata_note"]
