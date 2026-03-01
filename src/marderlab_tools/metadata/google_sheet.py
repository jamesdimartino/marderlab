from __future__ import annotations

from io import StringIO
from pathlib import Path
import re
from urllib.parse import quote

import pandas as pd
import requests


PUBLISHED_ITEM_RE = re.compile(
    r'items\.push\(\{name: "([^"]+)", pageUrl: "[^"]+", gid: "(\d+)"'
)


def _is_published_sheet_url(sheet_url: str) -> bool:
    return "/d/e/" in sheet_url and ("/pubhtml" in sheet_url or "/pub" in sheet_url)


def _published_base(sheet_url: str) -> str:
    cleaned = sheet_url.strip()
    if "/pubhtml" in cleaned:
        return cleaned.split("/pubhtml", 1)[0]
    if "/pub?" in cleaned:
        return cleaned.split("/pub?", 1)[0]
    if cleaned.endswith("/pub"):
        return cleaned[:-4]
    if "/pub/" in cleaned:
        return cleaned.split("/pub/", 1)[0]
    return cleaned.rstrip("/")


def _published_html_url(sheet_url: str) -> str:
    base = _published_base(sheet_url)
    return f"{base}/pubhtml"


def fetch_published_tab_gid_map(sheet_url: str, timeout_s: int = 20) -> dict[str, str]:
    html_url = _published_html_url(sheet_url)
    response = requests.get(html_url, timeout=timeout_s)
    response.raise_for_status()
    html = response.text
    matches = PUBLISHED_ITEM_RE.findall(html)
    mapping = {name.strip(): gid for name, gid in matches}
    if not mapping:
        raise ValueError("Could not parse published Google Sheet tab gid mapping.")
    return mapping


def build_tab_csv_url(sheet_url: str, tab_name: str, tab_gid_map: dict[str, str] | None = None) -> str:
    tab_q = quote(tab_name)
    cleaned = sheet_url.strip()

    if _is_published_sheet_url(cleaned):
        gid_map = tab_gid_map or fetch_published_tab_gid_map(cleaned)
        gid = gid_map.get(tab_name)
        if not gid:
            available = ", ".join(sorted(gid_map))
            raise ValueError(f"Tab '{tab_name}' not found in published sheet. Available tabs: {available}")
        base = _published_base(cleaned)
        return f"{base}/pub?gid={gid}&single=true&output=csv"

    if "/edit" in cleaned and "/d/" in cleaned:
        base = cleaned.split("/edit", 1)[0]
        return f"{base}/gviz/tq?tqx=out:csv&sheet={tab_q}"

    if cleaned.endswith(".csv"):
        return cleaned

    return f"{cleaned.rstrip('/')}/gviz/tq?tqx=out:csv&sheet={tab_q}"


def fetch_tab(
    sheet_url: str,
    tab_name: str,
    timeout_s: int = 20,
    tab_gid_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    url = build_tab_csv_url(sheet_url, tab_name, tab_gid_map=tab_gid_map)
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    payload = response.text
    if not payload.strip():
        raise ValueError(f"Google Sheet tab returned empty payload: {tab_name}")
    frame = pd.read_csv(StringIO(payload))
    frame["source_tab"] = tab_name
    return frame


def fetch_tabs(sheet_url: str, tabs: list[str]) -> dict[str, pd.DataFrame]:
    if not tabs:
        raise ValueError("No metadata tabs configured.")
    gid_map = fetch_published_tab_gid_map(sheet_url) if _is_published_sheet_url(sheet_url) else None
    output: dict[str, pd.DataFrame] = {}
    for tab in tabs:
        output[tab] = fetch_tab(sheet_url, tab, tab_gid_map=gid_map)
    return output


def sync_tabs_to_folder(sheet_url: str, tabs: list[str], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = fetch_tabs(sheet_url, tabs)
    paths: dict[str, Path] = {}
    for tab_name, frame in frames.items():
        path = out_dir / f"{tab_name}.csv"
        frame.to_csv(path, index=False)
        paths[tab_name] = path
    return paths
