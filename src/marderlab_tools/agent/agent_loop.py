from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    contract: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPolicy:
    require_tool_for_data_requests: bool = True
    require_successful_tool_for_data_requests: bool = True
    ask_clarifying_questions_first: bool = True
    prefer_processed_data: bool = True
    missing_data_behavior: str = "ask_user"
    default_stats_test: str = "t_test"
    response_contract_version: str = "1.0"
    status_update_interval_minutes: int = 5

    @staticmethod
    def from_dict(raw: dict[str, Any] | None = None) -> "AgentPolicy":
        raw = raw or {}
        return AgentPolicy(
            require_tool_for_data_requests=_coerce_bool(raw.get("require_tool_for_data_requests"), default=True),
            require_successful_tool_for_data_requests=_coerce_bool(
                raw.get("require_successful_tool_for_data_requests"), default=True
            ),
            ask_clarifying_questions_first=_coerce_bool(raw.get("ask_clarifying_questions_first"), default=True),
            prefer_processed_data=_coerce_bool(raw.get("prefer_processed_data"), default=True),
            missing_data_behavior=str(raw.get("missing_data_behavior", "ask_user")).strip() or "ask_user",
            default_stats_test=str(raw.get("default_stats_test", "t_test")).strip() or "t_test",
            response_contract_version=str(raw.get("response_contract_version", "1.0")).strip() or "1.0",
            status_update_interval_minutes=max(1, int(raw.get("status_update_interval_minutes", 5))),
        )


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


class AgentLoop:
    def __init__(
        self,
        router: ModelRouter,
        context: ContextService,
        tools: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 4,
        policy: AgentPolicy | dict[str, Any] | None = None,
    ) -> None:
        self.router = router
        self.context = context
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        if isinstance(policy, dict):
            self.policy = AgentPolicy.from_dict(policy)
        else:
            self.policy = policy or AgentPolicy()

    def ask(
        self,
        prompt: str,
        model_name: str | None = None,
        conversation: list[dict[str, str]] | None = None,
    ) -> AgentResponse:
        history = list(conversation or [])
        is_data_request = self._is_data_request(prompt)
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
        if (
            is_data_request
            and self.policy.ask_clarifying_questions_first
            and not self._looks_like_clarification_answer(prompt)
        ):
            clarifying = self._build_clarifying_questions(prompt, grounded_steps)
            if clarifying:
                clarifying_text = self._render_clarifying_questions(clarifying)
                return AgentResponse(
                    text=clarifying_text,
                    model="policy",
                    provider="policy",
                    steps=steps,
                    raw={},
                    contract=self._build_contract(
                        status="needs_user_input",
                        steps=steps,
                        clarifying_questions=clarifying,
                        failure_reasons=[],
                    ),
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
                failure_reasons = self._tool_enforcement_failures(steps, is_data_request=is_data_request)
                if failure_reasons:
                    clarifying = self._build_clarifying_questions(prompt, steps)
                    text = self._render_enforcement_message(clarifying, failure_reasons)
                return AgentResponse(
                    text=text or self._deterministic_fallback(prompt, steps),
                    model=str(raw.get("model", "")),
                    provider=str(raw.get("provider", "")),
                    steps=steps,
                    raw=raw,
                    contract=self._build_contract(
                        status="needs_user_input" if failure_reasons else "completed",
                        steps=steps,
                        clarifying_questions=self._build_clarifying_questions(prompt, steps) if failure_reasons else [],
                        failure_reasons=failure_reasons,
                    ),
                )

            # If model did not follow JSON format, return raw content.
            if not content:
                content = self._deterministic_fallback(prompt, steps)
            failure_reasons = self._tool_enforcement_failures(steps, is_data_request=is_data_request)
            if failure_reasons:
                clarifying = self._build_clarifying_questions(prompt, steps)
                content = self._render_enforcement_message(clarifying, failure_reasons)
            return AgentResponse(
                text=content,
                model=str(raw.get("model", "")),
                provider=str(raw.get("provider", "")),
                steps=steps,
                raw=raw,
                contract=self._build_contract(
                    status="needs_user_input" if failure_reasons else "completed",
                    steps=steps,
                    clarifying_questions=self._build_clarifying_questions(prompt, steps) if failure_reasons else [],
                    failure_reasons=failure_reasons,
                ),
            )

        final_text = self._deterministic_fallback(prompt, steps)
        failure_reasons = self._tool_enforcement_failures(steps, is_data_request=is_data_request)
        if failure_reasons:
            clarifying = self._build_clarifying_questions(prompt, steps)
            final_text = self._render_enforcement_message(clarifying, failure_reasons)
        return AgentResponse(
            text=final_text,
            model=str(last_raw.get("model", "")),
            provider=str(last_raw.get("provider", "")),
            steps=steps,
            raw=last_raw,
            contract=self._build_contract(
                status="needs_user_input" if failure_reasons else "completed",
                steps=steps,
                clarifying_questions=self._build_clarifying_questions(prompt, steps) if failure_reasons else [],
                failure_reasons=failure_reasons,
            ),
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
        if self._is_data_request(prompt):
            candidates.append(("resolve_request_context", {"prompt": prompt}))
            candidates.append(("list_pipelines", {}))
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

    @staticmethod
    def _is_data_request(prompt: str) -> bool:
        prompt_l = prompt.lower()
        data_tokens = (
            "plot",
            "graph",
            "figure",
            "amplitude",
            "contracture",
            "dataset",
            "process data",
            "analyze",
            "analysis of",
            "experiment",
            "dual10xk",
            "hikcontrol",
        )
        browse_only_tokens = ("what analyses can i run", "what pipelines", "list pipelines", "available pipelines")
        if any(token in prompt_l for token in browse_only_tokens):
            return False
        return any(token in prompt_l for token in data_tokens)

    @staticmethod
    def _looks_like_clarification_answer(prompt: str) -> bool:
        prompt_l = prompt.lower()
        return any(token in prompt_l for token in ("yes", "confirmed", "use default", "proceed", "go ahead"))

    def _tool_enforcement_failures(self, steps: list[dict[str, Any]], is_data_request: bool) -> list[str]:
        failures: list[str] = []
        if not is_data_request:
            return failures
        if self.policy.require_tool_for_data_requests and not steps:
            failures.append("no_tool_calls_for_data_request")
        if self.policy.require_successful_tool_for_data_requests:
            successful = self._successful_tool_calls(steps)
            if successful < 1:
                failures.append("no_successful_tool_call_for_data_request")
        return failures

    @staticmethod
    def _successful_tool_calls(steps: list[dict[str, Any]]) -> int:
        count = 0
        for step in steps:
            result = step.get("result", {})
            if isinstance(result, dict) and result.get("ok") is True:
                count += 1
        return count

    def _build_clarifying_questions(self, prompt: str, steps: list[dict[str, Any]]) -> list[str]:
        resolve_result = {}
        for step in steps:
            if step.get("tool") != "resolve_request_context":
                continue
            result = step.get("result", {})
            if isinstance(result, dict) and result.get("ok"):
                payload = result.get("result", {})
                if isinstance(payload, dict):
                    resolve_result = payload
                    break

        notebooks = resolve_result.get("candidate_notebooks", []) if resolve_result else []
        pipelines = resolve_result.get("candidate_pipelines", []) if resolve_result else []
        questions: list[str] = []

        if notebooks:
            notebook_text = ", ".join(str(n) for n in notebooks)
            questions.append(f"I inferred these notebook sources: {notebook_text}. Are these correct?")
        else:
            questions.append("Which notebook(s) should be treated as source of truth for this request?")

        if pipelines:
            pipeline_text = ", ".join(str(p) for p in pipelines)
            questions.append(f"I inferred these pipeline(s): {pipeline_text}. Should I use these?")
        else:
            questions.append("Which pipeline should I run for this request?")

        if self.policy.prefer_processed_data:
            questions.append("Should I use existing processed outputs first and only reprocess missing data?")
        if self.policy.missing_data_behavior == "ask_user":
            questions.append("If required metadata or files are missing, should I stop and ask you before continuing?")

        if "plot" in prompt.lower() or "graph" in prompt.lower() or "figure" in prompt.lower():
            questions.append(
                f"Should I apply default statistics ({self.policy.default_stats_test}) unless you specify another test?"
            )

        return questions

    @staticmethod
    def _render_clarifying_questions(questions: list[str]) -> str:
        lines = ["Before I run analysis, please confirm the following:"]
        for idx, question in enumerate(questions, start=1):
            lines.append(f"{idx}. {question}")
        return "\n".join(lines)

    def _render_enforcement_message(self, clarifying: list[str], failure_reasons: list[str]) -> str:
        reason_text = ", ".join(failure_reasons) if failure_reasons else "policy_enforcement"
        message = (
            "I cannot finalize a data-analysis answer yet because policy requires successful relevant tool calls "
            f"({reason_text})."
        )
        if not clarifying:
            clarifying = self._build_clarifying_questions("", [])
        return f"{message}\n\n{self._render_clarifying_questions(clarifying)}"

    def _build_contract(
        self,
        status: str,
        steps: list[dict[str, Any]],
        clarifying_questions: list[str],
        failure_reasons: list[str],
    ) -> dict[str, Any]:
        requires_user_input = status == "needs_user_input" or bool(clarifying_questions)
        return {
            "version": self.policy.response_contract_version,
            "status": status,
            "requires_user_input": requires_user_input,
            "clarifying_questions": clarifying_questions,
            "failure_reasons": failure_reasons,
            "tool_call_count": len(steps),
            "successful_tool_call_count": self._successful_tool_calls(steps),
            "status_update_interval_minutes": self.policy.status_update_interval_minutes,
        }
