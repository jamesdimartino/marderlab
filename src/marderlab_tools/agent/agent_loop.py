from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from marderlab_tools.agent.context_service import ContextService
from marderlab_tools.agent.model_router import ModelRouter
from marderlab_tools.agent.tool_registry import ToolRegistry


DEFAULT_SYSTEM_PROMPT = """You are a code-aware assistant for MarderLab pipelines.
You have read-only tools to inspect the workspace and help users run analysis safely.
When you need a tool, respond as JSON only:
{"action":"tool","name":"<tool_name>","args":{...}}
When you are ready to answer, respond as JSON only:
{"action":"answer","text":"..."}"""


@dataclass
class AgentResponse:
    text: str
    model: str
    provider: str
    steps: list[dict[str, Any]]
    raw: dict[str, Any]


class AgentLoop:
    def __init__(
        self,
        router: ModelRouter,
        context: ContextService,
        tools: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 4,
    ) -> None:
        self.router = router
        self.context = context
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def ask(
        self,
        prompt: str,
        model_name: str | None = None,
        conversation: list[dict[str, str]] | None = None,
    ) -> AgentResponse:
        history = list(conversation or [])
        prompt_context = self.context.build_prompt_context(prompt, max_hits=8)
        tool_manifest = json.dumps(self.tools.list_tools(), indent=2)

        system_content = (
            f"{self.system_prompt}\n\n"
            "Available tools:\n"
            f"{tool_manifest}\n\n"
            "Workspace context:\n"
            f"{prompt_context}"
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        steps: list[dict[str, Any]] = []
        grounded_steps = self._auto_ground(prompt)
        if grounded_steps:
            steps.extend(grounded_steps)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "AUTO_GROUNDED_TOOL_RESULTS\n"
                        + json.dumps(grounded_steps, indent=2)
                        + "\nUse these grounded facts when answering."
                    ),
                }
            )
        last_raw: dict[str, Any] = {}

        for _ in range(self.max_steps):
            raw = self.router.chat(
                messages,
                model_name=model_name,
                tools=self.tools.as_openai_tools(),
            )
            last_raw = raw
            native_calls = raw.get("tool_calls", [])
            if isinstance(native_calls, list) and native_calls:
                tool_feedback = self._run_native_tool_calls(native_calls)
                steps.extend(tool_feedback["steps"])
                messages.append({"role": "assistant", "content": str(raw.get("content", "") or "")})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "TOOL_RESULT\n"
                            + json.dumps(tool_feedback["results"], indent=2)
                            + "\nUse this result and continue."
                        ),
                    }
                )
                continue

            content = str(raw.get("content", "")).strip()
            parsed = self._parse_json_action(content)

            if parsed and parsed.get("action") == "tool":
                tool_name = str(parsed.get("name", "")).strip()
                args = parsed.get("args") if isinstance(parsed.get("args"), dict) else {}
                tool_result = self.tools.run_tool(tool_name, args)
                steps.append({"tool": tool_name, "args": args, "result": tool_result})
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "TOOL_RESULT\n"
                            + json.dumps(tool_result, indent=2)
                            + "\nUse this result and continue."
                        ),
                    }
                )
                continue

            if parsed and parsed.get("action") == "answer":
                text = str(parsed.get("text", "")).strip()
                return AgentResponse(
                    text=text or self._deterministic_fallback(prompt, steps),
                    model=str(raw.get("model", "")),
                    provider=str(raw.get("provider", "")),
                    steps=steps,
                    raw=raw,
                )

            # If model did not follow JSON format, return raw content.
            if not content:
                content = self._deterministic_fallback(prompt, steps)
            return AgentResponse(
                text=content,
                model=str(raw.get("model", "")),
                provider=str(raw.get("provider", "")),
                steps=steps,
                raw=raw,
            )

        return AgentResponse(
            text=self._deterministic_fallback(prompt, steps),
            model=str(last_raw.get("model", "")),
            provider=str(last_raw.get("provider", "")),
            steps=steps,
            raw=last_raw,
        )

    @staticmethod
    def _parse_json_action(content: str) -> dict[str, Any] | None:
        if not content:
            return None
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        if "action" not in parsed:
            return None
        return parsed

    def _run_native_tool_calls(self, native_calls: list[dict[str, Any]]) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for call in native_calls:
            name = str(call.get("name", "")).strip()
            args = call.get("args", {}) if isinstance(call.get("args", {}), dict) else {}
            result = self.tools.run_tool(name, args)
            steps.append({"tool": name, "args": args, "result": result})
            results.append({"name": name, "args": args, "result": result, "id": call.get("id", "")})
        return {"steps": steps, "results": results}

    def _auto_ground(self, prompt: str) -> list[dict[str, Any]]:
        p = prompt.lower()
        candidates: list[tuple[str, dict[str, Any]]] = []
        if ("pipeline" in p or "analysis" in p or "analyses" in p) and any(
            token in p for token in ("what", "which", "available", "can you run", "end to end")
        ):
            candidates.append(("list_pipelines", {}))
        if any(token in p for token in ("command", "cli", "how do i run", "options")):
            candidates.append(("list_cli_commands", {}))

        pipeline_hint = self._detect_pipeline_hint(p)
        if pipeline_hint and any(token in p for token in ("preview", "which experiments", "what experiments")):
            candidates.append(("preview_pipeline_experiments", {"pipeline": pipeline_hint, "limit": 20}))

        steps: list[dict[str, Any]] = []
        for name, args in candidates:
            result = self.tools.run_tool(name, args)
            steps.append({"tool": name, "args": args, "result": result, "source": "auto_ground"})
        return steps

    @staticmethod
    def _detect_pipeline_hint(prompt_lower: str) -> str | None:
        if "contracture" in prompt_lower:
            return "contracture"
        if "nerve" in prompt_lower:
            return "nerve_evoked"
        if "hik" in prompt_lower or "control" in prompt_lower:
            return "hikcontrol"
        return None

    def _deterministic_fallback(self, prompt: str, steps: list[dict[str, Any]]) -> str:
        prompt_l = prompt.lower()
        if "pipeline" in prompt_l or "analysis" in prompt_l or "analyses" in prompt_l:
            payload = self.tools.run_tool("list_pipelines", {})
            pipelines = payload.get("result", {}).get("pipelines", []) if payload.get("ok") else []
            if pipelines:
                as_text = ", ".join(str(x) for x in pipelines)
                return f"Current end-to-end pipelines available in this codebase: {as_text}."
        if "hello" in prompt_l or "hi" == prompt_l.strip():
            return "Hello. I can help with pipeline selection, command generation, and config validation."

        if steps:
            return "I could not format a full response, but I did run tool checks. Open 'Tool calls' to inspect results."
        return "I could not produce a complete answer from the model response. Please retry or ask a more specific question."
