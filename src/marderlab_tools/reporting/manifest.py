from __future__ import annotations

import getpass
import platform
import socket
import subprocess
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from marderlab_tools.config.schema import RunConfig, RunManifest


def _get_git_hash(cwd: Path | None = None) -> str:
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def make_manifest(
    config: RunConfig,
    input_files: list[Path],
    parameters: dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
) -> RunManifest:
    run_id = uuid.uuid4().hex[:12]
    machine = f"{platform.system()}-{platform.release()}@{socket.gethostname()}"
    return RunManifest(
        run_id=run_id,
        started_at=started_at.astimezone(UTC).isoformat(),
        finished_at=finished_at.astimezone(UTC).isoformat(),
        config_path=str(config.config_path or ""),
        raw_data_root=str(config.paths.raw_data_root),
        processed_root=str(config.paths.processed_root),
        cache_root=str(config.paths.cache_root),
        git_hash=_get_git_hash(config.config_path.parent if config.config_path else None),
        machine=machine,
        user=getpass.getuser(),
        input_files=[str(p) for p in input_files],
        parameters=parameters,
    )


def manifest_to_dict(manifest: RunManifest) -> dict[str, Any]:
    return asdict(manifest)
