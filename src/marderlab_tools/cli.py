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
    run_parser.add_argument(
        "--pipeline",
        required=True,
        choices=["contracture", "nerve-evoked", "nerve_evoked", "hikcontrol", "hik-control"],
    )
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

    genai_window = sub.add_parser("genai-window", help="Launch Streamlit GenAI window.")
    genai_window.add_argument("--agent-config", default="configs/genai.yml", help="Path to agent config YAML.")
    genai_window.add_argument("--workspace-root", default=".", help="Workspace root for code context.")
    genai_window.add_argument("--host", default="127.0.0.1", help="Host address for Streamlit.")
    genai_window.add_argument("--port", default=8501, type=int, help="Port for Streamlit.")
    genai_window.add_argument(
        "--browser",
        action="store_true",
        help="Open browser automatically (headless off).",
    )

    genai_chat = sub.add_parser("genai-chat", help="Run one GenAI prompt in terminal mode.")
    genai_chat.add_argument("--agent-config", default="configs/genai.yml", help="Path to agent config YAML.")
    genai_chat.add_argument("--workspace-root", default=".", help="Workspace root for code context.")
    genai_chat.add_argument("--model", default="", help="Model name in agent config.")
    genai_chat.add_argument("--prompt", required=True, help="Prompt to run.")
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

    try:
        if args.command in {"doctor", "sync-metadata", "run", "run-all"}:
            config = load_config(args.config)

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

        if args.command == "genai-window":
            from marderlab_tools.app.genai_window import launch_streamlit_window

            return launch_streamlit_window(
                agent_config_path=args.agent_config,
                workspace_root=args.workspace_root,
                host=args.host,
                port=int(args.port),
                open_browser=bool(args.browser),
            )

        if args.command == "genai-chat":
            from marderlab_tools.app.genai_window import run_single_prompt

            result = run_single_prompt(
                prompt=args.prompt,
                agent_config_path=args.agent_config,
                workspace_root=args.workspace_root,
                model_name=args.model or None,
            )
            _print_json(result)
            return 0

        parser.print_help()
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
