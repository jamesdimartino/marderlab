from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_dataframe_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write_text(path, frame.to_csv(index=False))


def load_dataframe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata cache not found: {path}")
    return pd.read_csv(path)


def save_tab_caches(cache_root: Path, tab_frames: dict[str, pd.DataFrame]) -> dict[str, Path]:
    tab_dir = cache_root / "tabs"
    tab_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for tab, frame in tab_frames.items():
        path = tab_dir / f"{tab}.csv"
        save_dataframe_csv(path, frame)
        out[tab] = path
    return out


def load_tab_cache(cache_root: Path, tab_name: str) -> pd.DataFrame:
    return load_dataframe_csv(cache_root / "tabs" / f"{tab_name}.csv")
