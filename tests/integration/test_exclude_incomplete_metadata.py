from pathlib import Path

import pandas as pd
import yaml

from marderlab_tools.config.schema import load_config
from marderlab_tools.run import orchestrator


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


def test_incomplete_metadata_excludes_experiment(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    exp = raw / "997_123"
    exp.mkdir(parents=True)
    (exp / "997_123_0001.abf").write_bytes(b"")

    cfg_path = tmp_path / "config.yml"
    _write_config(cfg_path, raw, processed, cache)
    config = load_config(cfg_path)

    metadata = pd.DataFrame(
        {
            "notebook_page": ["997_123"],
            "file_index": [1],
            "stim_index": [1],
            "temperature": [None],  # incomplete required field
            "condition": ["control"],
            "experiment_type": ["contracture"],
            "season": [20],
            "calibration": [0.3],
        }
    )

    def fake_load_metadata(_cfg):
        return metadata, True, "metadata_source=cache_csv"

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("ABF loader should not be called when metadata is incomplete.")

    monkeypatch.setattr(orchestrator, "load_metadata_with_fallback", fake_load_metadata)
    monkeypatch.setattr(orchestrator, "load_force_trigger", fail_if_called)

    report = orchestrator.run_pipeline(config, "contracture", generate_plots=False)
    assert report["summary"]["total_experiments"] == 1
    assert report["summary"]["success_count"] == 0
    result = report["results"][0]
    assert "Excluded from analysis: incomplete metadata" in result["message"]
    npy_file = processed / "997_123" / "npy" / "contracture_metrics.npy"
    assert not npy_file.exists()
