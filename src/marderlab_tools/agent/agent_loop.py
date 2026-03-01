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
        last_raw: dict[str, Any] = {}

        for _ in range(self.max_steps):
            raw = self.router.chat(
                messages,
                model_name=model_name,
                tools=self.tools.as_openai_tools(),
            )
            last_raw = raw
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
                    text=text,
                    model=str(raw.get("model", "")),
                    provider=str(raw.get("provider", "")),
                    steps=steps,
                    raw=raw,
                )

            # If model did not follow JSON format, return raw content.
            return AgentResponse(
                text=content,
                model=str(raw.get("model", "")),
                provider=str(raw.get("provider", "")),
                steps=steps,
                raw=raw,
            )

        return AgentResponse(
            text="Agent stopped after max tool steps without final answer.",
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
