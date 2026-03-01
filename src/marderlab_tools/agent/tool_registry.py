from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from marderlab_tools.agent.context_service import ContextService


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
    ) -> None:
        self.context = context
        self.pipelines = pipelines or ["contracture", "nerve_evoked", "hikcontrol"]
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
            "marder run-all --config configs/default.yml",
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
