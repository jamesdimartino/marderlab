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


def test_doctor_passes_with_cached_metadata(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    raw.mkdir(parents=True)

    cfg_path = tmp_path / "config.yml"
    _write_config(cfg_path, raw, processed, cache)
    config = load_config(cfg_path)

    cache.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "notebook_page": ["997_099"],
            "file_index": [1],
            "stim_index": [1],
            "temperature": [11.0],
            "condition": ["control"],
            "experiment_type": ["contracture"],
            "season": [20],
        }
    ).to_csv(config.metadata.cache_csv, index=False)

    def fake_sync_metadata(_cfg):
        raise RuntimeError("network down")

    monkeypatch.setattr(orchestrator, "sync_metadata", fake_sync_metadata)
    result = orchestrator.doctor(config)
    assert result["ok"]
