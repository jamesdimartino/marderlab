from __future__ import annotations

import re
from typing import Any

import pandas as pd


FILE_ID_PATTERN = re.compile(r"(?P<page>\d{3}_\d{3})_(?P<index>\d{4})", re.IGNORECASE)
RANGE_PATTERN = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


def normalize_columns(frame: pd.DataFrame, column_map: dict[str, list[str]]) -> pd.DataFrame:
    df = frame.copy()
    lower_to_real = {c.lower(): c for c in df.columns}
    rename_map: dict[str, str] = {}

    for canonical, aliases in column_map.items():
        candidates = [canonical, *aliases]
        for alias in candidates:
            real = lower_to_real.get(alias.lower())
            if real is not None:
                rename_map[real] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _extract_keys_from_filename(value: Any) -> tuple[str | None, int | None]:
    if value is None:
        return None, None
    text = str(value)
    match = FILE_ID_PATTERN.search(text)
    if not match:
        return None, None
    return match.group("page"), int(match.group("index"))


def _format_notebook_page(notebook: Any, page: Any) -> str | None:
    try:
        n = int(float(notebook))
        p = int(float(page))
    except (TypeError, ValueError):
        return None
    if n < 0 or p < 0:
        return None
    return f"{n:03d}_{p:03d}"


def _parse_file_indices(value: Any) -> list[int]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "files", "none"}:
        return []
    text = text.replace(",", " ").replace(";", " ")
    parts = [p for p in text.split() if p]
    out: list[int] = []
    for part in parts:
        range_match = RANGE_PATTERN.match(part)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if end >= start:
                out.extend(range(start, end + 1))
            continue
        try:
            out.append(int(part))
        except ValueError:
            digits = re.sub(r"[^\d]", "", part)
            if digits:
                out.append(int(digits))
    return out


def _expand_files_column(df: pd.DataFrame) -> pd.DataFrame:
    if "files" not in df.columns:
        return df
    expanded_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        row_data = row.to_dict()
        indices = _parse_file_indices(row_data.get("files"))
        if indices:
            for file_idx in indices:
                new_row = dict(row_data)
                new_row["file_index"] = file_idx
                expanded_rows.append(new_row)
        else:
            expanded_rows.append(row_data)
    if not expanded_rows:
        return df
    return pd.DataFrame(expanded_rows)


def attach_file_keys(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if "notebook_page" in df.columns:
        df["notebook_page"] = df["notebook_page"].astype(str).str.extract(r"(\d{3}_\d{3})", expand=False)
    elif "notebook" in df.columns and "page" in df.columns:
        df["notebook_page"] = [
            _format_notebook_page(notebook, page)
            for notebook, page in zip(df["notebook"], df["page"])
        ]

    if "file_index" in df.columns:
        df["file_index"] = pd.to_numeric(df["file_index"], errors="coerce")

    if ("file_index" not in df.columns or not df["file_index"].notna().any()) and "files" in df.columns:
        df = _expand_files_column(df)
        if "file_index" in df.columns:
            df["file_index"] = pd.to_numeric(df["file_index"], errors="coerce")

    has_page = "notebook_page" in df.columns and df["notebook_page"].notna().any()
    has_file = "file_index" in df.columns and df["file_index"].notna().any()
    if has_page and has_file:
        df["file_index"] = df["file_index"].astype("Int64")
        df = df.loc[df["notebook_page"].notna()].copy()
        return df

    source_col = None
    for candidate in ("filename", "file_name", "abf_file", "abf_filename"):
        if candidate in df.columns:
            source_col = candidate
            break

    if source_col is None:
        return df

    keys = df[source_col].apply(_extract_keys_from_filename)
    df["notebook_page"] = [k[0] for k in keys]
    df["file_index"] = pd.Series([k[1] for k in keys], dtype="Int64")
    df = df.loc[df["notebook_page"].notna()].copy()
    return df


def require_fields(frame: pd.DataFrame, required_fields: list[str]) -> None:
    missing = [field for field in required_fields if field not in frame.columns]
    if missing:
        raise ValueError(f"Missing required metadata fields: {missing}")
    empty_required = [f for f in required_fields if frame[f].isna().all()]
    if empty_required:
        raise ValueError(f"Required metadata fields are present but empty: {empty_required}")


def merge_metadata_tabs(
    tab_frames: dict[str, pd.DataFrame],
    column_map: dict[str, list[str]],
    required_fields: list[str],
) -> pd.DataFrame:
    if not tab_frames:
        raise ValueError("No metadata tab data available for merge.")

    normalized = [attach_file_keys(normalize_columns(frame, column_map)) for frame in tab_frames.values()]
    merged = pd.concat(normalized, ignore_index=True, sort=False)
    require_fields(merged, required_fields)

    if "file_index" in merged.columns:
        merged["file_index"] = pd.to_numeric(merged["file_index"], errors="coerce").astype("Int64")
    if "notebook_page" in merged.columns:
        merged["notebook_page"] = merged["notebook_page"].astype(str)
    return merged


def metadata_for_experiment(frame: pd.DataFrame, notebook_page: str) -> pd.DataFrame:
    if "notebook_page" not in frame.columns:
        raise ValueError("Metadata frame missing 'notebook_page' column.")
    subset = frame.loc[frame["notebook_page"] == notebook_page].copy()
    if "file_index" in subset.columns:
        subset = subset.sort_values(by="file_index")
    return subset.reset_index(drop=True)
