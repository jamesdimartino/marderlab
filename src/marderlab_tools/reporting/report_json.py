from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
        return path
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
