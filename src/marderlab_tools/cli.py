from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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
    run_parser.add_argument("--pipeline", required=True)
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

    simulate = sub.add_parser("simulate", help="Run notebook-ported simulation models.")
    simulate.add_argument(
        "--model",
        required=True,
        choices=["hiksim", "modelfiber", "musclemodel", "untitled-model"],
        help="Simulation model to run.",
    )
    simulate.add_argument("--output", required=True, help="Output .npz file path.")
    simulate.add_argument("--duration-s", type=float, default=None)
    simulate.add_argument("--dt-s", type=float, default=None)
    simulate.add_argument("--temperature-c", type=float, default=None)

    stimgen = sub.add_parser("stimulus-gen", help="Generate burst stimulus file from StimulusGen port.")
    stimgen.add_argument("--output", required=True, help="Output CSV path.")
    stimgen.add_argument("--duration-s", type=float, default=60.0)
    stimgen.add_argument("--sample-rate-hz", type=float, default=10000.0)
    stimgen.add_argument("--burst-count", type=int, default=10)
    stimgen.add_argument("--burst-width-s", type=float, default=0.08)
    stimgen.add_argument("--burst-amplitude-v", type=float, default=5.0)
    stimgen.add_argument("--start-delay-s", type=float, default=2.0)
    stimgen.add_argument("--inter-burst-s", type=float, default=4.0)
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

        if args.command == "simulate":
            import numpy as np

            from marderlab_tools.modeling.hiksim import HiKSimParams, run_hiksim
            from marderlab_tools.modeling.modelfiber import FiberParams, run_modelfiber
            from marderlab_tools.modeling.musclemodelrealistic_vm import (
                MuscleVMParams,
                run_musclemodelrealistic_vm,
            )
            from marderlab_tools.modeling.untitled_model import UntitledParams, run_untitled_model

            model = str(args.model)
            if model == "hiksim":
                params = HiKSimParams(
                    duration_s=args.duration_s if args.duration_s is not None else 300.0,
                    dt_s=args.dt_s if args.dt_s is not None else 0.01,
                    temperature_c=args.temperature_c if args.temperature_c is not None else 12.0,
                )
                payload = run_hiksim(params)
            elif model == "modelfiber":
                params = FiberParams(
                    duration_s=args.duration_s if args.duration_s is not None else 20.0,
                    dt_s=args.dt_s if args.dt_s is not None else 0.001,
                    temperature_c=args.temperature_c if args.temperature_c is not None else 12.0,
                )
                payload = run_modelfiber(params)
            elif model == "musclemodel":
                params = MuscleVMParams(
                    duration_s=args.duration_s if args.duration_s is not None else 240.0,
                    dt_s=args.dt_s if args.dt_s is not None else 0.005,
                    temperatures_c=(args.temperature_c if args.temperature_c is not None else 12.0,),
                )
                payload = run_musclemodelrealistic_vm(params)
            else:
                params = UntitledParams(
                    duration_s=args.duration_s if args.duration_s is not None else 10.0,
                    dt_s=args.dt_s if args.dt_s is not None else 0.0005,
                    temperature_c=args.temperature_c if args.temperature_c is not None else 12.0,
                )
                payload = run_untitled_model(params)

            out = Path(args.output).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            arrays = {k: v for k, v in payload.items() if hasattr(v, "shape")}
            np.savez_compressed(out, **arrays, summary=np.array([json.dumps(payload.get("summary", {}))], dtype=object))
            print(str(out))
            return 0

        if args.command == "stimulus-gen":
            from marderlab_tools.stimulus.stimulusgen import StimulusSpec, generate_burst_train, write_stimulus_file

            spec = StimulusSpec(
                duration_s=float(args.duration_s),
                sample_rate_hz=float(args.sample_rate_hz),
                burst_count=int(args.burst_count),
                burst_width_s=float(args.burst_width_s),
                burst_amplitude_v=float(args.burst_amplitude_v),
                start_delay_s=float(args.start_delay_s),
                inter_burst_s=float(args.inter_burst_s),
            )
            payload = generate_burst_train(spec)
            out = write_stimulus_file(args.output, payload)
            print(str(out))
            return 0

        parser.print_help()
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
