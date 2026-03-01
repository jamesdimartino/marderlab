from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from marderlab_tools.config.schema import ChannelMap, RunConfig


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_paths(config: RunConfig) -> list[CheckResult]:
    processed_parent_ok = config.paths.processed_root.parent.exists() or config.paths.processed_root.exists()
    cache_parent_ok = (
        config.paths.cache_root.parent.exists()
        or config.paths.cache_root.exists()
        or config.paths.cache_root.parent.parent.exists()
    )
    return [
        CheckResult(
            name="raw_data_root_exists",
            passed=config.paths.raw_data_root.exists(),
            message=f"Raw data root: {config.paths.raw_data_root}",
        ),
        CheckResult(
            name="processed_root_parent_exists",
            passed=processed_parent_ok,
            message=f"Processed root parent: {config.paths.processed_root.parent}",
        ),
        CheckResult(
            name="cache_root_parent_exists",
            passed=cache_parent_ok,
            message=f"Cache root parent: {config.paths.cache_root.parent}",
        ),
    ]


def check_channel_map(channel_map: ChannelMap) -> CheckResult:
    ok = bool(channel_map.force and channel_map.trigger)
    return CheckResult(
        name="channel_map_present",
        passed=ok,
        message=f"force={channel_map.force}, trigger={channel_map.trigger}",
    )


def check_required_metadata_fields(frame: pd.DataFrame, required_fields: list[str]) -> CheckResult:
    missing = [field for field in required_fields if field not in frame.columns]
    if missing:
        return CheckResult(
            name="metadata_required_fields",
            passed=False,
            message=f"Missing required metadata fields: {missing}",
        )
    return CheckResult(
        name="metadata_required_fields",
        passed=True,
        message="All required metadata fields are present.",
        severity="info",
    )


def check_experiment_has_metadata(frame: pd.DataFrame, notebook_page: str) -> CheckResult:
    has = bool((frame.get("notebook_page") == notebook_page).any())
    return CheckResult(
        name="experiment_metadata_presence",
        passed=has,
        message=f"Metadata rows found for notebook_page={notebook_page}: {has}",
    )


def all_passed(results: list[CheckResult]) -> bool:
    return all(result.passed for result in results)


def serialize_checks(results: list[CheckResult]) -> list[dict[str, Any]]:
    return [r.to_dict() for r in results]


def check_writable_directory(path: Path) -> CheckResult:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".write_test.tmp"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return CheckResult(
            name="directory_writable",
            passed=True,
            message=f"Writable directory: {path}",
            severity="info",
        )
    except Exception as exc:
        return CheckResult(
            name="directory_writable",
            passed=False,
            message=f"Directory not writable ({path}): {exc}",
        )
