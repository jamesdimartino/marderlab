from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from marderlab_tools.analysis import (
    contracture,
    control,
    dualhik,
    freqrange,
    gm56acclim,
    gm56weaklink,
    heartbeat,
    hikcontrol,
    muscle,
    nerve_evoked,
    rawheart,
)
from marderlab_tools.checks.validators import (
    CheckResult,
    all_passed,
    check_channel_map,
    check_experiment_has_metadata,
    check_paths,
    check_required_metadata_fields,
    check_writable_directory,
    serialize_checks,
)
from marderlab_tools.config.schema import ChannelMap, PipelineResult, RunConfig
from marderlab_tools.io.abf_loader import load_force_trigger
from marderlab_tools.io.experiment_discovery import discover_experiments, iter_all_input_files, parse_file_index
from marderlab_tools.metadata.cache import load_dataframe_csv, save_dataframe_csv, save_tab_caches
from marderlab_tools.metadata.google_sheet import fetch_tabs
from marderlab_tools.metadata.merge import merge_metadata_tabs, metadata_for_experiment
from marderlab_tools.reporting.manifest import make_manifest, manifest_to_dict
from marderlab_tools.reporting.report_html import write_html_report
from marderlab_tools.reporting.report_json import write_json_report
from marderlab_tools.stats.markers import compute_stat_markers


PIPELINE_CANONICAL = {
    "contracture": "contracture",
    "nerve-evoked": "nerve_evoked",
    "nerve_evoked": "nerve_evoked",
    "hikcontrol": "hikcontrol",
    "hik-control": "hikcontrol",
    "control": "control",
    "dualhik": "dualhik",
    "freqrange": "freqrange",
    "gm56acclim": "gm56acclim",
    "gm56weaklink": "gm56weaklink",
    "heartbeat": "heartbeat",
    "rawheart": "rawheart",
    "muscle": "muscle",
}

ProgressFn = Callable[[str], None]

PIPELINE_ANALYZERS: dict[str, tuple[Any, Callable[[list[Any], Any], dict[str, Any]]]] = {
    "contracture": (contracture.TraceRecord, contracture.analyze_experiment),
    "nerve_evoked": (nerve_evoked.TraceRecord, nerve_evoked.analyze_experiment),
    "hikcontrol": (hikcontrol.TraceRecord, hikcontrol.analyze_experiment),
    "control": (control.TraceRecord, control.analyze_experiment),
    "dualhik": (dualhik.TraceRecord, dualhik.analyze_experiment),
    "freqrange": (freqrange.TraceRecord, freqrange.analyze_experiment),
    "gm56acclim": (gm56acclim.TraceRecord, gm56acclim.analyze_experiment),
    "gm56weaklink": (gm56weaklink.TraceRecord, gm56weaklink.analyze_experiment),
    "heartbeat": (heartbeat.TraceRecord, heartbeat.analyze_experiment),
    "rawheart": (rawheart.TraceRecord, rawheart.analyze_experiment),
    "muscle": (muscle.TraceRecord, muscle.analyze_experiment),
}


def _normalize_pipeline_name(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    if key in PIPELINE_CANONICAL:
        return PIPELINE_CANONICAL[key]
    if key == "nerve-evoked":
        return "nerve_evoked"
    return name


def _pipeline_setting(config: RunConfig, pipeline_name: str):
    normalized = _normalize_pipeline_name(pipeline_name)
    if normalized in config.pipelines:
        return normalized, config.pipelines[normalized]
    alt = normalized.replace("-", "_")
    if alt in config.pipelines:
        return alt, config.pipelines[alt]
    raise KeyError(f"Pipeline not configured: {pipeline_name}")


def sync_metadata(config: RunConfig) -> pd.DataFrame:
    tab_frames = fetch_tabs(config.metadata.google_sheet_url, config.metadata.tabs)
    save_tab_caches(config.paths.cache_root, tab_frames)
    merged = merge_metadata_tabs(tab_frames, config.metadata.column_map, config.metadata.required_fields)
    save_dataframe_csv(config.metadata.cache_csv, merged)
    return merged


def load_metadata_with_fallback(config: RunConfig) -> tuple[pd.DataFrame, bool, str]:
    try:
        frame = sync_metadata(config)
        return frame, False, "metadata_source=google_sheet"
    except Exception as exc:
        if not config.metadata.cache_csv.exists():
            raise RuntimeError(
                f"Metadata sync failed and cache is unavailable: {exc}"
            ) from exc
        frame = load_dataframe_csv(config.metadata.cache_csv)
        frame = merge_metadata_tabs(
            {"cached": frame},
            config.metadata.column_map,
            config.metadata.required_fields,
        )
        return frame, True, f"metadata_source=cache_csv reason={exc}"


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _atomic_save_npy(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, payload, allow_pickle=True)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".csv", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number))


def _metric_unit(metric_name: str) -> str:
    name = metric_name.lower()
    if name.endswith("_cn"):
        return "cN"
    if name.endswith("_cn_s"):
        return "cN*s"
    if name.endswith("_cn_per_s"):
        return "cN/s"
    if name.endswith("_v"):
        return "V"
    if name.endswith("_s"):
        return "s"
    if name.endswith("_hz"):
        return "Hz"
    if "temperature" in name:
        return "C"
    return "unitless"


def _build_tidy_metrics_frame(
    notebook_page: str,
    pipeline_name: str,
    payload: dict[str, Any],
    file_metadata_by_index: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in payload.get("files", []):
        file_index = int(entry.get("file_index", 0))
        metadata = file_metadata_by_index.get(file_index, {})
        metrics = entry.get("metrics", {})
        source_file = str(entry.get("file_path", ""))
        for metric_name, metric_value in metrics.items():
            if not _is_finite_number(metric_value):
                continue
            rows.append(
                {
                    "notebook_page": notebook_page,
                    "pipeline": pipeline_name,
                    "file_index": file_index,
                    "temperature": metadata.get("temperature"),
                    "condition": metadata.get("condition"),
                    "stim_index": metadata.get("stim_index"),
                    "experiment_type": metadata.get("experiment_type"),
                    "season": metadata.get("season"),
                    "source_tab": metadata.get("source_tab"),
                    "metric_name": str(metric_name),
                    "metric_value": float(metric_value),
                    "metric_unit": _metric_unit(str(metric_name)),
                    "source_file": source_file,
                }
            )
    return pd.DataFrame(rows)


def _maybe_plot(output_svg: Path, title: str, y_values: list[float]) -> str | None:
    if not y_values:
        return None
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return "matplotlib unavailable; skipped plotting."

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, len(y_values) + 1), y_values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("File Number (ordered)")
    ax.set_ylabel("Peak (cN)")
    fig.tight_layout()
    fig.savefig(output_svg, format="svg")
    plt.close(fig)
    return None


def _filter_metadata_for_pipeline(frame: pd.DataFrame, settings: Any) -> pd.DataFrame:
    working = frame.copy()
    metadata_tabs = [str(v).strip() for v in getattr(settings, "metadata_tabs", []) if str(v).strip()]
    if metadata_tabs and "source_tab" in working.columns:
        allowed_tabs = {tab.lower() for tab in metadata_tabs}
        mask = working["source_tab"].astype(str).str.strip().str.lower().isin(allowed_tabs)
        working = working.loc[mask].copy()

    experiment_type_values = list(getattr(settings, "experiment_type_values", []))
    if "experiment_type" in working.columns and experiment_type_values:
        allowed = {str(v).strip().lower() for v in experiment_type_values}
        mask = working["experiment_type"].astype(str).str.strip().str.lower().isin(allowed)
        working = working.loc[mask].copy()
    return working


def _build_trace_records(
    notebook_page: str,
    files: list[Path],
    metadata_df: pd.DataFrame,
    channel_map: ChannelMap,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    available_files = {parse_file_index(file_path): file_path for file_path in files}
    rows = metadata_df.copy()
    if "file_index" not in rows.columns:
        raise ValueError(f"Missing crucial metadata for {notebook_page}: file_index column is absent.")
    rows = rows.loc[rows["file_index"].notna()].copy()
    if rows.empty:
        raise ValueError(f"Missing crucial metadata for {notebook_page}: no file_index rows available.")
    rows["file_index"] = pd.to_numeric(rows["file_index"], errors="coerce").astype("Int64")
    rows = rows.loc[rows["file_index"].notna()].sort_values(by="file_index")

    for _, row in rows.iterrows():
        file_idx = int(row["file_index"])
        file_path = available_files.get(file_idx)
        if file_path is None:
            raise ValueError(
                f"Metadata references missing ABF for {notebook_page} file_index={file_idx}."
            )
        trace = load_force_trigger(file_path, channel_map)
        records.append(
            {
                "file_path": file_path,
                "time_s": trace.time_s,
                "force_v": trace.force_v,
                "trigger_v": trace.trigger_v,
                "sample_rate_hz": trace.sample_rate_hz,
                "metadata": row.to_dict(),
            }
        )
    return records


def _metadata_incomplete_issues(
    metadata_df: pd.DataFrame,
    files: list[Path],
    required_fields: list[str],
) -> list[str]:
    issues: list[str] = []
    if metadata_df.empty:
        return ["no metadata rows found for experiment"]

    if "file_index" not in metadata_df.columns:
        return ["missing file_index column"]

    rows = metadata_df.copy()
    rows["file_index"] = pd.to_numeric(rows["file_index"], errors="coerce")
    rows = rows.loc[rows["file_index"].notna()].copy()
    if rows.empty:
        return ["no file_index rows available"]
    rows["file_index"] = rows["file_index"].astype(int)

    available_indices = {parse_file_index(file_path) for file_path in files}
    missing_abf = sorted(set(rows["file_index"]) - available_indices)
    if missing_abf:
        preview = ", ".join(str(v) for v in missing_abf[:5])
        if len(missing_abf) > 5:
            preview += ", ..."
        issues.append(f"metadata references missing ABF file indices [{preview}]")

    for field in required_fields:
        if field not in rows.columns:
            issues.append(f"missing required field '{field}'")
            continue

        col = rows[field]
        missing_count = int(col.isna().sum())
        if missing_count > 0:
            issues.append(f"required field '{field}' has {missing_count} missing values")
            continue

        if col.dtype == object:
            blank_count = int(col.astype(str).str.strip().eq("").sum())
            if blank_count > 0:
                issues.append(f"required field '{field}' has {blank_count} blank values")
    return issues


def _excluded_experiment_result(notebook_page: str, pipeline_name: str, issues: list[str]) -> PipelineResult:
    detail = "; ".join(issues)
    return PipelineResult(
        notebook_page=notebook_page,
        pipeline=pipeline_name,
        success=False,
        message=f"Excluded from analysis: incomplete metadata ({detail})",
        flags=[
            {
                "code": "metadata_incomplete_excluded",
                "message": detail,
                "severity": "error",
            }
        ],
        checks=serialize_checks(
            [
                CheckResult(
                    name="metadata_complete_for_experiment",
                    passed=False,
                    message=detail,
                )
            ]
        ),
    )


def _run_single_experiment(
    notebook_page: str,
    files: list[Path],
    metadata_df: pd.DataFrame,
    pipeline_name: str,
    settings: Any,
    output_root: Path,
    generate_plots: bool,
) -> PipelineResult:
    checks: list[CheckResult] = []
    checks.append(check_experiment_has_metadata(metadata_df, notebook_page))
    channel_map = ChannelMap(force=settings.force_channel, trigger=settings.trig_channel)
    checks.append(check_channel_map(channel_map))

    if not all_passed(checks):
        return PipelineResult(
            notebook_page=notebook_page,
            pipeline=pipeline_name,
            success=False,
            message="Pre-run checks failed.",
            checks=serialize_checks(checks),
        )

    trace_records = _build_trace_records(notebook_page, files, metadata_df, channel_map)
    file_metadata_by_index: dict[int, dict[str, Any]] = {}
    for record in trace_records:
        file_idx = int(record["metadata"].get("file_index", 0))
        file_metadata_by_index[file_idx] = record["metadata"]

    analyzer_entry = PIPELINE_ANALYZERS.get(pipeline_name)
    if analyzer_entry is None and pipeline_name == "nerve-evoked":
        analyzer_entry = PIPELINE_ANALYZERS.get("nerve_evoked")
    if analyzer_entry is None and pipeline_name == "hik-control":
        analyzer_entry = PIPELINE_ANALYZERS.get("hikcontrol")
    if analyzer_entry is None:
        raise ValueError(f"Unsupported pipeline: {pipeline_name}")
    trace_type, analyze_fn = analyzer_entry
    typed_records = [trace_type(**r) for r in trace_records]
    payload = analyze_fn(typed_records, settings)

    npy_dir = output_root / notebook_page / "npy"
    plots_dir = output_root / notebook_page / "plots"
    npy_path = npy_dir / f"{pipeline_name}_metrics.npy"
    tidy_csv_path = npy_dir / f"{pipeline_name}_metrics_tidy.csv"
    _atomic_save_npy(npy_path, payload)
    tidy_frame = _build_tidy_metrics_frame(
        notebook_page=notebook_page,
        pipeline_name=pipeline_name,
        payload=payload,
        file_metadata_by_index=file_metadata_by_index,
    )
    _atomic_save_csv(tidy_csv_path, tidy_frame)

    output_paths = {"npy": str(npy_path), "tidy_csv": str(tidy_csv_path)}
    if generate_plots:
        peaks = [
            float(item["metrics"].get("peak_cn", item["metrics"].get("amplitude_cn", 0.0)))
            for item in payload.get("files", [])
        ]
        plot_path = plots_dir / f"{pipeline_name}_peaks.svg"
        plot_warning = _maybe_plot(plot_path, f"{notebook_page} {pipeline_name} peaks", peaks)
        output_paths["plot"] = str(plot_path)
        if plot_warning:
            payload.setdefault("flags", []).append(
                {"code": "plot_skipped", "message": plot_warning, "severity": "warning"}
            )

    checks.append(CheckResult(name="output_npy_exists", passed=npy_path.exists(), message=str(npy_path)))
    checks.append(CheckResult(name="output_tidy_csv_exists", passed=tidy_csv_path.exists(), message=str(tidy_csv_path)))
    return PipelineResult(
        notebook_page=notebook_page,
        pipeline=pipeline_name,
        success=all_passed(checks),
        message="ok" if all_passed(checks) else "Output validation failed.",
        output_paths=output_paths,
        flags=list(payload.get("flags", [])),
        checks=serialize_checks(checks),
    )


def _pipeline_pages(
    experiments: dict[str, list[Path]],
    metadata_df: pd.DataFrame,
) -> list[str]:
    pages_from_meta = sorted(set(metadata_df.get("notebook_page", pd.Series(dtype=str)).dropna().astype(str)))
    return [page for page in pages_from_meta if page in experiments]


def _apply_page_subset(
    pages: list[str],
    include_pages: list[str] | None = None,
    max_experiments: int | None = None,
) -> list[str]:
    filtered = pages
    if include_pages:
        include = {p.strip() for p in include_pages if p.strip()}
        filtered = [p for p in filtered if p in include]
    if max_experiments is not None:
        filtered = filtered[: max(0, int(max_experiments))]
    return filtered


def _compute_run_stats(results: list[PipelineResult]) -> dict[str, Any]:
    success = sum(1 for r in results if r.success)
    total = len(results)
    group_values: dict[str, list[float]] = {}
    for result in results:
        group = result.pipeline
        group_values.setdefault(group, []).append(1.0 if result.success else 0.0)
    return {
        "success_count": success,
        "failure_count": total - success,
        "total_experiments": total,
        "stat_markers": compute_stat_markers(group_values),
    }


def _write_run_reports(
    config: RunConfig,
    title: str,
    results: list[PipelineResult],
    input_files: list[Path],
    parameters: dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
    metadata_note: str,
) -> dict[str, Any]:
    manifest = make_manifest(config, input_files, parameters, started_at, finished_at)
    run_dir = config.paths.processed_root / "_runs" / manifest.run_id
    report = {
        "title": title,
        "manifest": manifest_to_dict(manifest),
        "metadata_note": metadata_note,
        "summary": _compute_run_stats(results),
        "results": [asdict(result) for result in results],
    }

    run_json = write_json_report(run_dir / "run_report.json", report)
    run_html = write_html_report(run_dir / "run_report.html", report)
    manifest_path = write_json_report(
        config.paths.cache_root / "manifests" / f"{manifest.run_id}.json",
        manifest_to_dict(manifest),
    )
    report["artifacts"] = {
        "run_report_json": str(run_json),
        "run_report_html": str(run_html),
        "manifest_json": str(manifest_path),
    }

    vscode_gallery = _write_vscode_sanity_gallery(
        config=config,
        run_id=manifest.run_id,
        results=results,
    )
    if vscode_gallery is not None:
        report["artifacts"]["vscode_sanity_html"] = str(vscode_gallery)
    return report


def _write_vscode_sanity_gallery(
    config: RunConfig,
    run_id: str,
    results: list[PipelineResult],
) -> Path | None:
    gallery_root = config.paths.cache_root / "vscode_sanity" / run_id
    plots_dir = gallery_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, str]] = []
    for result in results:
        source_plot = result.output_paths.get("plot")
        if not source_plot:
            continue
        source_path = Path(source_plot)
        if not source_path.exists():
            continue
        safe_name = (
            f"{result.pipeline}_{result.notebook_page}_{source_path.name}"
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        dest = plots_dir / safe_name
        shutil.copy2(source_path, dest)
        copied.append(
            {
                "pipeline": result.pipeline,
                "notebook_page": result.notebook_page,
                "status": "PASS" if result.success else "FAIL",
                "plot_rel": f"plots/{safe_name}",
                "source_plot": str(source_path),
            }
        )

    if not copied:
        return None

    rows = []
    for item in copied:
        rows.append(
            "<div class='card'>"
            f"<h3>{item['pipeline']} - {item['notebook_page']} ({item['status']})</h3>"
            f"<p><a href='{item['plot_rel']}' target='_blank'>{item['plot_rel']}</a></p>"
            f"<p>Processed-data file: {item['source_plot']}</p>"
            f"<img src='{item['plot_rel']}' loading='lazy' />"
            "</div>"
        )

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>VSCode Sanity Gallery</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:16px;background:#f8fafc;}"
        "h1{margin-bottom:6px;} .meta{margin-bottom:14px;color:#444;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px;}"
        ".card{background:#fff;padding:10px;border:1px solid #ddd;border-radius:8px;}"
        ".card img{width:100%;height:auto;border:1px solid #ddd;background:#fff;}"
        "a{color:#0b57d0;}"
        "</style></head><body>"
        f"<h1>VSCode Sanity Gallery</h1><div class='meta'>Run ID: {run_id}</div>"
        "<div class='grid'>"
        + "".join(rows)
        + "</div></body></html>"
    )

    gallery_html = gallery_root / "index.html"
    gallery_html.write_text(html_doc, encoding="utf-8")
    return gallery_html


def run_pipeline(
    config: RunConfig,
    pipeline_name: str,
    generate_plots: bool = False,
    include_pages: list[str] | None = None,
    max_experiments: int | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(tz=UTC)
    pipeline_key, settings = _pipeline_setting(config, pipeline_name)
    metadata_df, _used_fallback, note = load_metadata_with_fallback(config)

    required_check = check_required_metadata_fields(metadata_df, config.metadata.required_fields)
    if not required_check.passed:
        raise ValueError(required_check.message)

    filtered_meta = _filter_metadata_for_pipeline(metadata_df, settings)
    experiments = discover_experiments(config.paths.raw_data_root)
    selected_pages = _apply_page_subset(
        _pipeline_pages(experiments, filtered_meta),
        include_pages=include_pages,
        max_experiments=max_experiments,
    )
    if progress:
        progress(
            f"pipeline={pipeline_key} selected_experiments={len(selected_pages)} "
            f"plots={generate_plots} metadata_note={note}"
        )

    results: list[PipelineResult] = []
    analyzed_input_files: list[Path] = []
    total = len(selected_pages)
    for idx, page in enumerate(selected_pages, start=1):
        if progress:
            progress(f"[{idx}/{total}] start {pipeline_key} {page}")
        files = experiments[page]
        meta = metadata_for_experiment(filtered_meta, page)
        issues = _metadata_incomplete_issues(meta, files, config.metadata.required_fields)
        if issues:
            result = _excluded_experiment_result(page, pipeline_key, issues)
            results.append(result)
            if progress:
                progress(f"[{idx}/{total}] excluded {pipeline_key} {page}: {result.message}")
            continue

        analyzed_input_files.extend(files)
        try:
            result = _run_single_experiment(
                notebook_page=page,
                files=files,
                metadata_df=meta,
                pipeline_name=pipeline_key,
                settings=settings,
                output_root=config.paths.processed_root,
                generate_plots=generate_plots,
            )
        except Exception as exc:
            result = PipelineResult(
                notebook_page=page,
                pipeline=pipeline_key,
                success=False,
                message=str(exc),
            )
        results.append(result)
        if progress:
            artifact = result.output_paths.get("plot") or result.output_paths.get("npy", "")
            status = "ok" if result.success else "failed"
            progress(f"[{idx}/{total}] {status} {pipeline_key} {page} {artifact}".strip())

    finished_at = datetime.now(tz=UTC)
    input_files = analyzed_input_files or iter_all_input_files({p: experiments[p] for p in selected_pages})
    return _write_run_reports(
        config=config,
        title=f"Marder run report: {pipeline_key}",
        results=results,
        input_files=input_files,
        parameters={
            "pipeline": pipeline_key,
            "plots": generate_plots,
            "include_pages": include_pages or [],
            "max_experiments": max_experiments,
        },
        started_at=started_at,
        finished_at=finished_at,
        metadata_note=note,
    )


def run_all(
    config: RunConfig,
    generate_plots: bool = False,
    include_pages: list[str] | None = None,
    max_experiments: int | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(tz=UTC)
    metadata_df, _used_fallback, note = load_metadata_with_fallback(config)
    required_check = check_required_metadata_fields(metadata_df, config.metadata.required_fields)
    if not required_check.passed:
        raise ValueError(required_check.message)

    experiments = discover_experiments(config.paths.raw_data_root)
    results: list[PipelineResult] = []
    selected_input_files: list[Path] = []

    seen: set[str] = set()
    configured: list[str] = []
    for name in config.pipelines.keys():
        canonical = _normalize_pipeline_name(name)
        if canonical not in PIPELINE_ANALYZERS:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        configured.append(canonical)

    for pipeline_name in configured:
        canonical = _normalize_pipeline_name(pipeline_name)
        settings = config.pipelines.get(canonical)
        if settings is None:
            alt = canonical.replace("-", "_")
            if alt in config.pipelines:
                settings = config.pipelines[alt]
            else:
                continue
        pipeline_run_name = canonical
        filtered_meta = _filter_metadata_for_pipeline(metadata_df, settings)
        pages = _apply_page_subset(
            _pipeline_pages(experiments, filtered_meta),
            include_pages=include_pages,
            max_experiments=max_experiments,
        )
        if progress:
            progress(
                f"pipeline={pipeline_run_name} selected_experiments={len(pages)} "
                f"plots={generate_plots} metadata_note={note}"
            )

        total = len(pages)
        for idx, page in enumerate(pages, start=1):
            if progress:
                progress(f"[{idx}/{total}] start {pipeline_run_name} {page}")
            files = experiments[page]
            meta = metadata_for_experiment(filtered_meta, page)
            issues = _metadata_incomplete_issues(meta, files, config.metadata.required_fields)
            if issues:
                result = _excluded_experiment_result(page, pipeline_run_name, issues)
                results.append(result)
                if progress:
                    progress(f"[{idx}/{total}] excluded {pipeline_run_name} {page}: {result.message}")
                continue

            selected_input_files.extend(files)
            try:
                result = _run_single_experiment(
                    notebook_page=page,
                    files=files,
                    metadata_df=meta,
                    pipeline_name=pipeline_run_name,
                    settings=settings,
                    output_root=config.paths.processed_root,
                    generate_plots=generate_plots,
                )
            except Exception as exc:
                result = PipelineResult(
                    notebook_page=page,
                    pipeline=pipeline_run_name,
                    success=False,
                    message=str(exc),
                )
            results.append(result)
            if progress:
                artifact = result.output_paths.get("plot") or result.output_paths.get("npy", "")
                status = "ok" if result.success else "failed"
                progress(f"[{idx}/{total}] {status} {pipeline_run_name} {page} {artifact}".strip())

    finished_at = datetime.now(tz=UTC)
    return _write_run_reports(
        config=config,
        title="Marder run report: run-all",
        results=results,
        input_files=selected_input_files,
        parameters={
            "pipeline": "run-all",
            "plots": generate_plots,
            "include_pages": include_pages or [],
            "max_experiments": max_experiments,
        },
        started_at=started_at,
        finished_at=finished_at,
        metadata_note=note,
    )


def doctor(config: RunConfig) -> dict[str, Any]:
    checks: list[CheckResult] = []
    checks.extend(check_paths(config))
    checks.append(check_writable_directory(config.paths.cache_root))
    checks.append(check_writable_directory(config.paths.processed_root))

    try:
        metadata_df, used_fallback, note = load_metadata_with_fallback(config)
        checks.append(check_required_metadata_fields(metadata_df, config.metadata.required_fields))
    except Exception as exc:
        note = f"metadata_check_failed={exc}"
        checks.append(CheckResult(name="metadata_access", passed=False, message=str(exc)))
        used_fallback = False

    for name, settings in config.pipelines.items():
        checks.append(check_channel_map(ChannelMap(force=settings.force_channel, trigger=settings.trig_channel)))
        checks.append(
            CheckResult(
                name=f"pipeline_config_{_safe_slug(name)}",
                passed=bool(settings.experiment_type_values),
                message=f"experiment_type_values={settings.experiment_type_values}",
            )
        )

    return {
        "ok": all_passed(checks),
        "used_fallback": used_fallback,
        "metadata_note": note,
        "checks": serialize_checks(checks),
    }
