from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class ModelSpec:
    name: str
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    default_temperature: float = 0.1
    default_max_tokens: int = 1200
    supports_tools: bool = True


@dataclass
class RouterConfig:
    default_model: str
    fallback_model: str | None = None
    models: dict[str, ModelSpec] = field(default_factory=dict)


class ModelRouter:
    """Provider-agnostic chat router with optional fallback behavior."""

    def __init__(self, config: RouterConfig):
        if config.default_model not in config.models:
            raise ValueError(f"default_model '{config.default_model}' is not in model registry.")
        self.config = config

    @staticmethod
    def from_dict(config_dict: dict[str, Any]) -> "ModelRouter":
        models: dict[str, ModelSpec] = {}
        for key, item in config_dict.get("models", {}).items():
            models[key] = ModelSpec(
                name=key,
                provider=str(item.get("provider", "mock")),
                model=str(item.get("model", "mock-model")),
                base_url=item.get("base_url"),
                api_key_env=item.get("api_key_env"),
                default_temperature=float(item.get("default_temperature", 0.1)),
                default_max_tokens=int(item.get("default_max_tokens", 1200)),
                supports_tools=bool(item.get("supports_tools", True)),
            )
        cfg = RouterConfig(
            default_model=str(config_dict.get("default_model", "")),
            fallback_model=config_dict.get("fallback_model"),
            models=models,
        )
        return ModelRouter(cfg)

    def list_models(self) -> list[str]:
        return sorted(self.config.models.keys())

    def chat(
        self,
        messages: list[dict[str, str]],
        model_name: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        target = model_name or self.config.default_model
        try:
            return self._chat_one(target, messages, tools, temperature, max_tokens)
        except Exception as first_error:
            fallback = self.config.fallback_model
            if fallback and fallback != target:
                response = self._chat_one(fallback, messages, tools, temperature, max_tokens)
                response["fallback_from"] = target
                response["fallback_reason"] = str(first_error)
                return response
            raise

    def _chat_one(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        if model_name not in self.config.models:
            raise ValueError(f"Unknown model: {model_name}")
        spec = self.config.models[model_name]
        temp = spec.default_temperature if temperature is None else float(temperature)
        token_cap = spec.default_max_tokens if max_tokens is None else int(max_tokens)

        provider = spec.provider.strip().lower()
        if provider == "mock":
            return self._mock_chat(model_name, messages)
        if provider == "openai":
            return self._openai_chat(spec, messages, tools, temp, token_cap)
        if provider == "anthropic":
            return self._anthropic_chat(spec, messages, tools, temp, token_cap)
        if provider == "ollama":
            return self._ollama_chat(spec, messages, temp, token_cap)
        raise ValueError(f"Unsupported provider: {spec.provider}")

    @staticmethod
    def _mock_chat(model_name: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        prompt = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                prompt = m.get("content", "")
                break
        content = json.dumps(
            {
                "action": "answer",
                "text": f"[mock:{model_name}] {prompt}",
            }
        )
        return {"model": model_name, "provider": "mock", "content": content}

    def _openai_chat(
        self,
        spec: ModelSpec,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        base = spec.base_url or "https://api.openai.com"
        url = base.rstrip("/") + "/v1/chat/completions"
        api_key = self._api_key(spec.api_key_env)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools and spec.supports_tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        self._raise_for_status_with_body(resp, "openai")
        body = resp.json()
        message = body["choices"][0]["message"]
        tool_calls = self._extract_openai_tool_calls(message)
        content = message.get("content", "") or ""
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            ).strip()
        return {
            "model": spec.name,
            "provider": "openai",
            "content": content,
            "tool_calls": tool_calls,
            "raw": body,
        }

    def _anthropic_chat(
        self,
        spec: ModelSpec,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        base = spec.base_url or "https://api.anthropic.com"
        url = base.rstrip("/") + "/v1/messages"
        api_key = self._api_key(spec.api_key_env)
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        # Anthropic expects system prompt separately; use user/assistant exchange only.
        system_chunks = [m["content"] for m in messages if m.get("role") == "system"]
        non_system = [
            {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
            for m in messages
            if str(m.get("role", "")) in {"user", "assistant"} and str(m.get("content", "")).strip()
        ]
        payload: dict[str, Any] = {
            "model": spec.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": non_system,
            "system": "\n\n".join(system_chunks),
        }
        if tools and spec.supports_tools:
            anthropic_tools = self._to_anthropic_tools(tools)
            if anthropic_tools:
                payload["tools"] = anthropic_tools
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        self._raise_for_status_with_body(resp, "anthropic")
        body = resp.json()
        text_chunks = []
        tool_calls: list[dict[str, Any]] = []
        for item in body.get("content", []):
            if item.get("type") == "text":
                text_chunks.append(item.get("text", ""))
            if item.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": str(item.get("id", "")),
                        "name": str(item.get("name", "")),
                        "args": item.get("input", {}) if isinstance(item.get("input", {}), dict) else {},
                    }
                )
        return {
            "model": spec.name,
            "provider": "anthropic",
            "content": "\n".join(text_chunks).strip(),
            "tool_calls": tool_calls,
            "raw": body,
        }

    @staticmethod
    def _ollama_chat(
        spec: ModelSpec,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        base = spec.base_url or "http://127.0.0.1:11434"
        url = base.rstrip("/") + "/api/chat"
        payload = {
            "model": spec.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = requests.post(url, json=payload, timeout=90)
        ModelRouter._raise_for_status_with_body(resp, "ollama")
        body = resp.json()
        content = body.get("message", {}).get("content", "")
        return {
            "model": spec.name,
            "provider": "ollama",
            "content": content,
            "raw": body,
        }

    @staticmethod
    def _api_key(env_name: str | None) -> str:
        if not env_name:
            raise ValueError("api_key_env is required for this provider.")
        key = os.environ.get(env_name, "").strip()
        if not key:
            raise ValueError(f"Missing API key in environment variable: {env_name}")
        return key

    @staticmethod
    def _to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool.get("function", {})
                name = str(fn.get("name", "")).strip()
                if not name:
                    continue
                converted.append(
                    {
                        "name": name,
                        "description": str(fn.get("description", "")).strip(),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
                continue
            if "name" in tool and "input_schema" in tool:
                converted.append(tool)
        return converted

    @staticmethod
    def _raise_for_status_with_body(response: requests.Response, provider: str) -> None:
        if response.status_code < 400:
            return
        detail = ""
        try:
            detail_obj = response.json()
            detail = json.dumps(detail_obj)
        except Exception:
            detail = response.text.strip()
        if len(detail) > 500:
            detail = detail[:500] + "..."
        message = (
            f"{provider} api error {response.status_code} for {response.url}."
            + (f" details={detail}" if detail else "")
        )
        raise requests.HTTPError(message, response=response)

    @staticmethod
    def _extract_openai_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for call in message.get("tool_calls", []) or []:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            args_raw = fn.get("arguments", "{}")
            args: dict[str, Any] = {}
            if isinstance(args_raw, dict):
                args = args_raw
            elif isinstance(args_raw, str):
                try:
                    parsed = json.loads(args_raw)
                    if isinstance(parsed, dict):
                        args = parsed
                except json.JSONDecodeError:
                    args = {}
            calls.append(
                {
                    "id": str(call.get("id", "")),
                    "name": str(fn.get("name", "")),
                    "args": args,
                }
            )
        return calls
