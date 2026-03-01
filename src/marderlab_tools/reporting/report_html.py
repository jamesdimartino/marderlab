from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path
from typing import Any


def _status_label(success: bool) -> str:
    return "PASS" if success else "FAIL"


def build_html_report(report: dict[str, Any]) -> str:
    title = html.escape(str(report.get("title", "Marder Run Report")))
    manifest = report.get("manifest", {})
    results = report.get("results", [])

    rows = []
    for entry in results:
        notebook_page = html.escape(str(entry.get("notebook_page", "")))
        pipeline = html.escape(str(entry.get("pipeline", "")))
        status = _status_label(bool(entry.get("success")))
        message = html.escape(str(entry.get("message", "")))
        rows.append(
            f"<tr><td>{notebook_page}</td><td>{pipeline}</td><td>{status}</td><td>{message}</td></tr>"
        )

    meta_lines = []
    for key in ("run_id", "started_at", "finished_at", "git_hash", "user", "machine"):
        meta_lines.append(f"<li><b>{html.escape(key)}</b>: {html.escape(str(manifest.get(key, '')))}</li>")

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ccc;padding:6px 8px;text-align:left;}th{background:#f4f4f4;}"
        ".PASS{color:#0b7a0b;font-weight:700;}.FAIL{color:#a11010;font-weight:700;}</style>"
        "</head><body>"
        f"<h1>{title}</h1>"
        "<h2>Manifest</h2><ul>"
        + "".join(meta_lines)
        + "</ul><h2>Results</h2><table><thead><tr><th>Experiment</th><th>Pipeline</th><th>Status</th><th>Message</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )


def write_html_report(path: Path, report: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(build_html_report(report))
        os.replace(tmp_path, path)
        return path
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
