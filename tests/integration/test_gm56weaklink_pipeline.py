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
            "tabs": ["FTMuscle"],
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
            "gm56weaklink": {
                "metadata_tabs": ["FTMuscle"],
                "experiment_type_values": ["weaklink"],
                "force_channel": "force",
                "trig_channel": "trig",
            }
        },
        "checks": {"fail_mode": "experiment_only"},
        "stats": {"default_rule": "by_group_count"},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _make_trace(file_path: Path) -> FakeTrace:
    fs = 1000.0
    t = np.arange(0, 12, 1 / fs)
    trigger = np.zeros_like(t)
    force = np.zeros_like(t)
    for i in range(1, 11):
        start = int(i * fs)
        trigger[start : start + 20] = 1.0
        amp = 0.02 * i
        force[start : start + 200] += amp * np.exp(-np.linspace(0, 2.5, 200))
    return FakeTrace(
        file_path=file_path,
        sample_rate_hz=fs,
        time_s=t,
        force_v=force,
        trigger_v=trigger,
        channel_names=["force", "trig"],
    )


def test_gm56weaklink_pipeline_outputs_ratio(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    exp = raw / "997_444"
    exp.mkdir(parents=True)
    (exp / "997_444_0001.abf").write_bytes(b"")

    cfg_path = tmp_path / "config.yml"
    _write_config(cfg_path, raw, processed, cache)
    config = load_config(cfg_path)

    metadata = pd.DataFrame(
        {
            "notebook_page": ["997_444"],
            "file_index": [1],
            "stim_index": [8],
            "temperature": [12.0],
            "condition": ["gm56"],
            "experiment_type": ["weaklink"],
            "season": [20],
            "source_tab": ["FTMuscle"],
        }
    )

    def fake_load_metadata(_cfg):
        return metadata, True, "metadata_source=cache_csv"

    monkeypatch.setattr(orchestrator, "load_metadata_with_fallback", fake_load_metadata)
    monkeypatch.setattr(orchestrator, "load_force_trigger", lambda fp, _cm: _make_trace(fp))

    report = orchestrator.run_pipeline(config, "gm56weaklink", generate_plots=False)
    assert report["summary"]["total_experiments"] == 1
    assert report["summary"]["success_count"] == 1
    npy_file = processed / "997_444" / "npy" / "gm56weaklink_metrics.npy"
    assert npy_file.exists()
