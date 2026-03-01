from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_PATTERN = re.compile(r"^(?P<page>\d{3}_\d{3})$")
ABF_FILE_PATTERN = re.compile(r"^(?P<page>\d{3}_\d{3})_(?P<index>\d{4})\.abf$", re.IGNORECASE)


@dataclass
class ExperimentFiles:
    notebook_page: str
    files: list[Path]


def parse_file_index(file_path: Path) -> int:
    match = ABF_FILE_PATTERN.match(file_path.name)
    if not match:
        raise ValueError(f"File does not match expected ABF naming pattern: {file_path.name}")
    return int(match.group("index"))


def discover_experiments(raw_data_root: Path) -> dict[str, list[Path]]:
    experiments: dict[str, list[Path]] = {}
    for child in raw_data_root.iterdir():
        if not child.is_dir():
            continue
        match = EXPERIMENT_PATTERN.match(child.name)
        if not match:
            continue
        notebook_page = match.group("page")
        files = [
            p for p in child.glob("*.abf")
            if ABF_FILE_PATTERN.match(p.name) and p.name.startswith(f"{notebook_page}_")
        ]
        if not files:
            continue
        files.sort(key=parse_file_index)
        experiments[notebook_page] = files
    return experiments


def iter_all_input_files(experiments: dict[str, list[Path]]) -> list[Path]:
    return [p for files in experiments.values() for p in files]
