from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathSettings:
    raw_data_root: Path
    processed_root: Path
    cache_root: Path


@dataclass
class MetadataSettings:
    google_sheet_url: str
    tabs: list[str]
    cache_csv: Path
    required_fields: list[str]
    column_map: dict[str, list[str]]


@dataclass
class PipelineSettings:
    experiment_type_values: list[str]
    force_channel: str
    trig_channel: str
    metadata_tabs: list[str] = field(default_factory=list)
    baseline_seconds: float = 2.0
    sample_rate_hz: float = 10000.0
    quality_std_floor: float = 1e-6
    quality_clip_abs: float = 10.0
    trigger_threshold: float = 0.5


@dataclass
class CheckSettings:
    fail_mode: str = "experiment_only"


@dataclass
class StatsSettings:
    default_rule: str = "by_group_count"


@dataclass
class RunConfig:
    paths: PathSettings
    metadata: MetadataSettings
    pipelines: dict[str, PipelineSettings]
    checks: CheckSettings
    stats: StatsSettings
    config_path: Path | None = None


@dataclass
class ChannelMap:
    force: str
    trigger: str


@dataclass
class QualityFlag:
    code: str
    message: str
    severity: str = "warning"


@dataclass
class ExperimentRecord:
    notebook_page: str
    files: list[Path]
    metadata_rows: list[dict[str, Any]]
    pipeline_name: str
    channel_map: ChannelMap


@dataclass
class PipelineResult:
    notebook_page: str
    pipeline: str
    success: bool
    message: str
    output_paths: dict[str, str] = field(default_factory=dict)
    flags: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunManifest:
    run_id: str
    started_at: str
    finished_at: str
    config_path: str
    raw_data_root: str
    processed_root: str
    cache_root: str
    git_hash: str
    machine: str
    user: str
    input_files: list[str]
    parameters: dict[str, Any]


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _parse_pipeline_settings(raw: dict[str, Any]) -> PipelineSettings:
    return PipelineSettings(
        experiment_type_values=list(raw.get("experiment_type_values", [])),
        force_channel=str(raw.get("force_channel", "force")),
        trig_channel=str(raw.get("trig_channel", "trig")),
        metadata_tabs=list(raw.get("metadata_tabs", [])),
        baseline_seconds=float(raw.get("baseline_seconds", 2.0)),
        sample_rate_hz=float(raw.get("sample_rate_hz", 10000.0)),
        quality_std_floor=float(raw.get("quality_std_floor", 1e-6)),
        quality_clip_abs=float(raw.get("quality_clip_abs", 10.0)),
        trigger_threshold=float(raw.get("trigger_threshold", 0.5)),
    )


def load_config(path: str | Path) -> RunConfig:
    config_path = _path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    paths_raw = raw.get("paths", {})
    metadata_raw = raw.get("metadata", {})
    pipelines_raw = raw.get("pipelines", {})

    paths = PathSettings(
        raw_data_root=_path(paths_raw.get("raw_data_root", ".")),
        processed_root=_path(paths_raw.get("processed_root", "./processed_data")),
        cache_root=_path(paths_raw.get("cache_root", "./.cache/marderlab")),
    )

    metadata = MetadataSettings(
        google_sheet_url=str(metadata_raw.get("google_sheet_url", "")).strip(),
        tabs=list(metadata_raw.get("tabs", [])),
        cache_csv=_path(metadata_raw.get("cache_csv", paths.cache_root / "metadata_latest.csv")),
        required_fields=list(metadata_raw.get("required_fields", [])),
        column_map=dict(metadata_raw.get("column_map", {})),
    )

    pipelines = {
        name: _parse_pipeline_settings(values)
        for name, values in pipelines_raw.items()
    }
    if "nerve-evoked" not in pipelines and "nerve_evoked" in pipelines:
        pipelines["nerve-evoked"] = pipelines["nerve_evoked"]
    if "nerve_evoked" not in pipelines and "nerve-evoked" in pipelines:
        pipelines["nerve_evoked"] = pipelines["nerve-evoked"]

    checks = CheckSettings(fail_mode=str(raw.get("checks", {}).get("fail_mode", "experiment_only")))
    stats = StatsSettings(default_rule=str(raw.get("stats", {}).get("default_rule", "by_group_count")))
    return RunConfig(
        paths=paths,
        metadata=metadata,
        pipelines=pipelines,
        checks=checks,
        stats=stats,
        config_path=config_path,
    )


def config_to_dict(config: RunConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["paths"]["raw_data_root"] = str(config.paths.raw_data_root)
    payload["paths"]["processed_root"] = str(config.paths.processed_root)
    payload["paths"]["cache_root"] = str(config.paths.cache_root)
    payload["metadata"]["cache_csv"] = str(config.metadata.cache_csv)
    payload["config_path"] = str(config.config_path) if config.config_path else None
    return payload
