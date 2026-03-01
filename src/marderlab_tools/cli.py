from __future__ import annotations

import argparse
import json
import sys
from typing import Callable

from marderlab_tools.config.schema import load_config
from marderlab_tools.run.orchestrator import doctor, run_all, run_pipeline, sync_metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marder", description="Marder Lab analysis pipelines")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor_parser = sub.add_parser("doctor", help="Validate install, paths, metadata, and channel mapping.")
    doctor_parser.add_argument("--config", required=True, help="Path to YAML config file.")

    sync_parser = sub.add_parser("sync-metadata", help="Sync metadata from Google Sheet into local cache.")
    sync_parser.add_argument("--config", required=True, help="Path to YAML config file.")

    run_parser = sub.add_parser("run", help="Run one pipeline.")
    run_parser.add_argument("--pipeline", required=True, choices=["contracture", "nerve-evoked", "nerve_evoked"])
    run_parser.add_argument("--config", required=True, help="Path to YAML config file.")
    run_parser.add_argument("--plots", action="store_true", help="Generate SVG plots.")
    run_parser.add_argument(
        "--pages",
        default="",
        help="Comma-separated notebook_page subset (example: 997_006,997_008).",
    )
    run_parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum number of experiments to run after filtering.",
    )
    run_parser.add_argument(
        "--live",
        action="store_true",
        help="Print per-experiment progress as analysis runs.",
    )

    run_all_parser = sub.add_parser("run-all", help="Run all configured pipelines.")
    run_all_parser.add_argument("--config", required=True, help="Path to YAML config file.")
    run_all_parser.add_argument("--plots", action="store_true", help="Generate SVG plots.")
    run_all_parser.add_argument(
        "--pages",
        default="",
        help="Comma-separated notebook_page subset (applies to each pipeline).",
    )
    run_all_parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum experiments per pipeline after filtering.",
    )
    run_all_parser.add_argument(
        "--live",
        action="store_true",
        help="Print per-experiment progress as analysis runs.",
    )
    return parser


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2))


def _parse_pages(raw: str) -> list[str]:
    if not raw:
        return []
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def _progress(enabled: bool) -> Callable[[str], None] | None:
    if not enabled:
        return None
    return lambda msg: print(msg, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    try:
        if args.command == "doctor":
            result = doctor(config)
            _print_json(result)
            return 0 if result.get("ok") else 1

        if args.command == "sync-metadata":
            frame = sync_metadata(config)
            print(f"metadata rows: {len(frame)}")
            print(f"cache csv: {config.metadata.cache_csv}")
            return 0

        if args.command == "run":
            report = run_pipeline(
                config,
                args.pipeline,
                generate_plots=bool(args.plots),
                include_pages=_parse_pages(args.pages),
                max_experiments=args.max_experiments,
                progress=_progress(bool(args.live)),
            )
            _print_json(
                {
                    "summary": report.get("summary"),
                    "artifacts": report.get("artifacts"),
                    "metadata_note": report.get("metadata_note"),
                }
            )
            return 0

        if args.command == "run-all":
            report = run_all(
                config,
                generate_plots=bool(args.plots),
                include_pages=_parse_pages(args.pages),
                max_experiments=args.max_experiments,
                progress=_progress(bool(args.live)),
            )
            _print_json(
                {
                    "summary": report.get("summary"),
                    "artifacts": report.get("artifacts"),
                    "metadata_note": report.get("metadata_note"),
                }
            )
            return 0

        parser.print_help()
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
