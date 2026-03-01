from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from marderlab_tools.agent.context_service import ContextService
from marderlab_tools.checks.validators import CheckResult, check_channel_map, check_paths
from marderlab_tools.config.schema import ChannelMap, load_config
from marderlab_tools.io.experiment_discovery import discover_experiments
from marderlab_tools.metadata.cache import load_dataframe_csv


@dataclass
class ToolSpec:
    name: str
    description: str
    args: dict[str, str]


class ToolRegistry:
    """Safe, read-only tools exposed to the GenAI loop."""

    def __init__(
        self,
        context: ContextService,
        pipelines: list[str] | None = None,
        default_config_path: str | None = None,
    ) -> None:
        self.context = context
        self.pipelines = pipelines or [
            "contracture",
            "nerve_evoked",
            "hikcontrol",
            "control",
            "dualhik",
            "freqrange",
            "gm56acclim",
            "gm56weaklink",
            "muscle",
            "heartbeat",
            "rawheart",
        ]
        self.default_config_path = default_config_path or "configs/default.yml"
        self._tools: dict[str, tuple[ToolSpec, Callable[[dict[str, Any]], dict[str, Any]]]] = {
            "workspace_summary": (
                ToolSpec(
                    name="workspace_summary",
                    description="Return a high-level summary of the project workspace.",
                    args={},
                ),
                self._workspace_summary,
            ),
            "list_pipelines": (
                ToolSpec(
                    name="list_pipelines",
                    description="List supported analysis pipelines and canonical command names.",
                    args={},
                ),
                self._list_pipelines,
            ),
            "list_cli_commands": (
                ToolSpec(
                    name="list_cli_commands",
                    description="List key CLI commands relevant to this codebase.",
                    args={},
                ),
                self._list_cli_commands,
            ),
            "search_code": (
                ToolSpec(
                    name="search_code",
                    description="Search code/text files for a query string.",
                    args={"query": "string", "max_hits": "int (optional)"},
                ),
                self._search_code,
            ),
            "read_file_excerpt": (
                ToolSpec(
                    name="read_file_excerpt",
                    description="Read a line-bounded excerpt from a file path under workspace root.",
                    args={"path": "string", "start_line": "int", "end_line": "int"},
                ),
                self._read_file_excerpt,
            ),
            "resolve_request_context": (
                ToolSpec(
                    name="resolve_request_context",
                    description=(
                        "Infer likely pipelines/notebooks and missing inputs for a data or plotting request."
                    ),
                    args={"prompt": "string"},
                ),
                self._resolve_request_context,
            ),
            "validate_pipeline_config": (
                ToolSpec(
                    name="validate_pipeline_config",
                    description="Static config validation without writing files or running analysis.",
                    args={"config_path": "string (optional)"},
                ),
                self._validate_pipeline_config,
            ),
            "preview_pipeline_experiments": (
                ToolSpec(
                    name="preview_pipeline_experiments",
                    description=(
                        "Preview which experiments a pipeline would consider using local metadata cache + "
                        "raw-data folder discovery (no writes, no analysis execution)."
                    ),
                    args={
                        "pipeline": "string",
                        "config_path": "string (optional)",
                        "limit": "int (optional)",
                    },
                ),
                self._preview_pipeline_experiments,
            ),
            "build_run_command": (
                ToolSpec(
                    name="build_run_command",
                    description="Build a ready-to-run marder CLI command string for run/run-all.",
                    args={
                        "mode": "run | run-all",
                        "pipeline": "string (required when mode=run)",
                        "config_path": "string (optional)",
                        "pages": "comma-separated notebook_page list (optional)",
                        "max_experiments": "int (optional)",
                        "plots": "true|false (optional)",
                        "live": "true|false (optional)",
                    },
                ),
                self._build_run_command,
            ),
        }

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": spec.name, "description": spec.description, "args": spec.args}
            for spec, _ in self._tools.values()
        ]

    def as_openai_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for spec, _ in self._tools.values():
            properties: dict[str, Any] = {}
            required: list[str] = []
            for arg_name, arg_type in spec.args.items():
                properties[arg_name] = {"type": "string", "description": f"{arg_name} ({arg_type})"}
                if "optional" not in arg_type:
                    required.append(arg_name)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return tools

    def run_tool(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = args or {}
        entry = self._tools.get(name)
        if entry is None:
            return {"ok": False, "error": f"Unknown tool: {name}"}
        _, handler = entry
        try:
            payload = handler(args)
            return {"ok": True, "tool": name, "result": payload}
        except Exception as exc:
            return {"ok": False, "tool": name, "error": str(exc)}

    def _workspace_summary(self, _args: dict[str, Any]) -> dict[str, Any]:
        return self.context.workspace_summary()

    def _list_pipelines(self, _args: dict[str, Any]) -> dict[str, Any]:
        return {"pipelines": self.pipelines}

    def _list_cli_commands(self, _args: dict[str, Any]) -> dict[str, Any]:
        commands = [
            "marder doctor --config configs/default.yml",
            "marder sync-metadata --config configs/default.yml",
            "marder run --pipeline contracture --config configs/default.yml",
            "marder run --pipeline nerve-evoked --config configs/default.yml",
            "marder run --pipeline hikcontrol --config configs/default.yml",
            "marder run --pipeline dualhik --config configs/default.yml",
            "marder run --pipeline gm56weaklink --config configs/default.yml",
            "marder run --pipeline heartbeat --config configs/default.yml",
            "marder run-all --config configs/default.yml",
            "marder simulate --model hiksim --output outputs/hiksim_run.npz",
            "marder stimulus-gen --output outputs/burst_train.csv",
            "marder genai-window --agent-config configs/genai.yml",
        ]
        return {"commands": commands}

    def _search_code(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        max_hits = int(args.get("max_hits", 20))
        if not query:
            raise ValueError("search_code requires 'query'")
        hits = self.context.find_text(query, max_hits=max_hits)
        return {"query": query, "hits": hits}

    def _read_file_excerpt(self, args: dict[str, Any]) -> dict[str, Any]:
        path = str(args.get("path", "")).strip()
        if not path:
            raise ValueError("read_file_excerpt requires 'path'")
        start_line = int(args.get("start_line", 1))
        end_line = int(args.get("end_line", start_line + 80))
        text = self.context.file_excerpt(path, start_line=start_line, end_line=end_line)
        resolved = (self.context.workspace_root / Path(path)).resolve() if not Path(path).is_absolute() else Path(path)
        return {
            "path": str(resolved),
            "start_line": start_line,
            "end_line": end_line,
            "text": text,
        }

    def _resolve_request_context(self, args: dict[str, Any]) -> dict[str, Any]:
        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("resolve_request_context requires 'prompt'")
        prompt_l = prompt.lower()

        pipeline_aliases = {
            "contracture": "contracture",
            "nerve_evoked": "nerve_evoked",
            "nerve-evoked": "nerve_evoked",
            "pairedcontractions": "nerve_evoked",
            "hikcontrol": "hikcontrol",
            "hik": "hikcontrol",
            "control": "control",
            "dualhik": "dualhik",
            "dual10xk": "dualhik",
            "freqrange": "freqrange",
            "gm56acclim": "gm56acclim",
            "gm56weaklink": "gm56weaklink",
            "muscle": "muscle",
            "heartbeat": "heartbeat",
            "rawheart": "rawheart",
        }
        notebook_by_pipeline = {
            "contracture": ["contracture.ipynb"],
            "nerve_evoked": ["PairedContractions.ipynb"],
            "hikcontrol": ["control.ipynb", "HIKSIM.ipynb"],
            "control": ["control.ipynb"],
            "dualhik": ["dualhik.ipynb"],
            "freqrange": ["freqrange.ipynb"],
            "gm56acclim": ["gm56acclim.ipynb"],
            "gm56weaklink": ["gm56weaklink.ipynb"],
            "muscle": ["Muscle.ipynb"],
            "heartbeat": ["Heartbeat.ipynb"],
            "rawheart": ["rawheart.ipynb"],
        }

        candidate_pipelines: list[str] = []
        for alias, pipeline in pipeline_aliases.items():
            if alias in prompt_l and pipeline not in candidate_pipelines:
                candidate_pipelines.append(pipeline)

        candidate_notebooks: list[str] = []
        for pipeline in candidate_pipelines:
            for notebook in notebook_by_pipeline.get(pipeline, []):
                if notebook not in candidate_notebooks:
                    candidate_notebooks.append(notebook)

        is_data_request = any(
            token in prompt_l
            for token in (
                "plot",
                "graph",
                "figure",
                "amplitude",
                "data",
                "contracture",
                "process",
                "experiment",
                "dual10xk",
            )
        )

        missing_inputs: list[str] = []
        if is_data_request and not candidate_pipelines:
            missing_inputs.append("pipeline_or_experiment_type")
        if is_data_request and not candidate_notebooks:
            missing_inputs.append("notebook_confirmation")
        if is_data_request and "vs" not in prompt_l and "compare" not in prompt_l:
            missing_inputs.append("comparison_groups")

        return {
            "prompt": prompt,
            "is_data_request": is_data_request,
            "candidate_pipelines": candidate_pipelines,
            "candidate_notebooks": candidate_notebooks,
            "missing_inputs": missing_inputs,
            "defaults": {
                "prefer_processed_data": True,
                "missing_data_behavior": "ask_user",
                "default_stats_test": "t_test",
            },
        }

    def _resolve_config_path(self, path_arg: Any) -> Path:
        raw = str(path_arg).strip() if path_arg is not None else ""
        selected = raw or self.default_config_path
        path = Path(selected)
        if not path.is_absolute():
            path = (self.context.workspace_root / path).resolve()
        return path

    def _validate_pipeline_config(self, args: dict[str, Any]) -> dict[str, Any]:
        cfg_path = self._resolve_config_path(args.get("config_path"))
        cfg = load_config(cfg_path)
        checks = []
        checks.extend(check_paths(cfg))
        for name, settings in cfg.pipelines.items():
            checks.append(check_channel_map(ChannelMap(force=settings.force_channel, trigger=settings.trig_channel)))
            checks.append(
                CheckResult(
                    name=f"pipeline_{name}_has_type_values",
                    passed=bool(settings.experiment_type_values),
                    message=f"experiment_type_values={settings.experiment_type_values}",
                    severity="error",
                )
            )
        return {
            "config_path": str(cfg_path),
            "ok": all(item.passed for item in checks),
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                }
                for c in checks
            ],
        }

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default

    def _build_run_command(self, args: dict[str, Any]) -> dict[str, Any]:
        mode = str(args.get("mode", "run")).strip().lower()
        if mode not in {"run", "run-all"}:
            raise ValueError("mode must be 'run' or 'run-all'")
        cfg_path = self._resolve_config_path(args.get("config_path"))
        cfg_rel = cfg_path
        try:
            cfg_rel = cfg_path.relative_to(self.context.workspace_root)
        except ValueError:
            pass
        base = f"marder {mode} --config {cfg_rel}"
        if mode == "run":
            pipeline = str(args.get("pipeline", "")).strip()
            if not pipeline:
                raise ValueError("pipeline is required when mode=run")
            base += f" --pipeline {pipeline}"
        pages = str(args.get("pages", "")).strip()
        if pages:
            base += f" --pages {pages}"
        max_experiments = str(args.get("max_experiments", "")).strip()
        if max_experiments:
            base += f" --max-experiments {max_experiments}"
        if self._coerce_bool(args.get("plots"), default=False):
            base += " --plots"
        if self._coerce_bool(args.get("live"), default=False):
            base += " --live"
        return {"command": base}

    def _preview_pipeline_experiments(self, args: dict[str, Any]) -> dict[str, Any]:
        pipeline = str(args.get("pipeline", "")).strip()
        if not pipeline:
            raise ValueError("preview_pipeline_experiments requires 'pipeline'")
        limit = int(args.get("limit", 20))
        cfg_path = self._resolve_config_path(args.get("config_path"))
        cfg = load_config(cfg_path)
        pipeline_key = pipeline.replace("-", "_")
        if pipeline_key not in cfg.pipelines:
            raise ValueError(f"Pipeline not configured: {pipeline}")
        settings = cfg.pipelines[pipeline_key]

        if not cfg.metadata.cache_csv.exists():
            return {
                "pipeline": pipeline_key,
                "config_path": str(cfg_path),
                "note": f"metadata cache missing: {cfg.metadata.cache_csv}",
                "selected_pages": [],
                "selected_count": 0,
            }

        metadata_df = load_dataframe_csv(cfg.metadata.cache_csv)
        working = metadata_df.copy()
        if "source_tab" in working.columns and settings.metadata_tabs:
            tabs = {t.strip().lower() for t in settings.metadata_tabs if t.strip()}
            if tabs:
                mask = working["source_tab"].astype(str).str.strip().str.lower().isin(tabs)
                tab_subset = working.loc[mask].copy()
                if not tab_subset.empty:
                    working = tab_subset
        if "experiment_type" in working.columns and settings.experiment_type_values:
            types = {v.strip().lower() for v in settings.experiment_type_values if v.strip()}
            if types:
                mask = working["experiment_type"].astype(str).str.strip().str.lower().isin(types)
                type_subset = working.loc[mask].copy()
                if not type_subset.empty:
                    working = type_subset

        if "notebook_page" not in working.columns:
            return {
                "pipeline": pipeline_key,
                "config_path": str(cfg_path),
                "note": "metadata cache lacks notebook_page column",
                "selected_pages": [],
                "selected_count": 0,
            }

        discovered = discover_experiments(cfg.paths.raw_data_root)
        meta_pages = sorted(set(working["notebook_page"].dropna().astype(str)))
        selected_pages = [page for page in meta_pages if page in discovered]
        sample = selected_pages[: max(0, limit)]
        return {
            "pipeline": pipeline_key,
            "config_path": str(cfg_path),
            "raw_data_root": str(cfg.paths.raw_data_root),
            "metadata_cache": str(cfg.metadata.cache_csv),
            "selected_count": len(selected_pages),
            "selected_pages": sample,
        }
