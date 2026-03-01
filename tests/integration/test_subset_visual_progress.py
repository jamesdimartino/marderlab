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
                "metadata_tabs": ["FTBath"],
                "experiment_type_values": ["contracture"],
                "force_channel": "force",
                "trig_channel": "trig",
            }
        },
        "checks": {"fail_mode": "experiment_only"},
        "stats": {"default_rule": "by_group_count"},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_subset_pages_and_live_progress(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"

    exp_a = raw / "997_101"
    exp_a.mkdir(parents=True)
    (exp_a / "997_101_0001.abf").write_bytes(b"")

    exp_b = raw / "997_102"
    exp_b.mkdir(parents=True)
    (exp_b / "997_102_0001.abf").write_bytes(b"")

    cfg_path = tmp_path / "config.yml"
    _write_config(cfg_path, raw, processed, cache)
    config = load_config(cfg_path)

    metadata = pd.DataFrame(
        {
            "notebook_page": ["997_101", "997_102"],
            "file_index": [1, 1],
            "stim_index": [1, 1],
            "temperature": [10.0, 10.0],
            "condition": ["control", "control"],
            "experiment_type": ["contracture", "contracture"],
            "season": [20, 20],
            "source_tab": ["FTBath", "FTBath"],
        }
    )

    def fake_load_metadata(_cfg):
        return metadata, True, "metadata_source=cache_csv"

    def fake_abf_loader(file_path, _channel_map):
        n = 300
        t = np.linspace(0, 3, n)
        trigger = np.zeros(n)
        trigger[100:110] = 1.0
        force = np.sin(t)
        return FakeTrace(
            file_path=file_path,
            sample_rate_hz=100.0,
            time_s=t,
            force_v=force,
            trigger_v=trigger,
            channel_names=["force", "trig"],
        )

    def fake_plot(output_svg, _title, _y_values):
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        output_svg.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return None

    logs: list[str] = []

    monkeypatch.setattr(orchestrator, "load_metadata_with_fallback", fake_load_metadata)
    monkeypatch.setattr(orchestrator, "load_force_trigger", fake_abf_loader)
    monkeypatch.setattr(orchestrator, "_maybe_plot", fake_plot)

    report = orchestrator.run_pipeline(
        config,
        "contracture",
        generate_plots=True,
        include_pages=["997_102"],
        max_experiments=1,
        progress=logs.append,
    )
    assert report["summary"]["total_experiments"] == 1
    assert report["summary"]["success_count"] == 1
    assert len(report["results"]) == 1
    assert report["results"][0]["notebook_page"] == "997_102"
    assert any("start contracture 997_102" in line for line in logs)
    assert any("ok contracture 997_102" in line for line in logs)
    gallery = report["artifacts"].get("vscode_sanity_html")
    assert gallery is not None
    assert Path(gallery).exists()
