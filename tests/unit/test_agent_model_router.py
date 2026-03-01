import requests

from marderlab_tools.agent.model_router import ModelRouter


def test_mock_router_echoes_prompt() -> None:
    router = ModelRouter.from_dict(
        {
            "default_model": "mock",
            "models": {"mock": {"provider": "mock", "model": "marder-mock"}},
        }
    )
    response = router.chat([{"role": "user", "content": "hello"}])
    assert response["provider"] == "mock"
    assert "hello" in response["content"]


def test_router_uses_fallback_when_default_fails() -> None:
    router = ModelRouter.from_dict(
        {
            "default_model": "broken",
            "fallback_model": "mock",
            "models": {
                "broken": {"provider": "not-a-provider", "model": "x"},
                "mock": {"provider": "mock", "model": "marder-mock"},
            },
        }
    )
    response = router.chat([{"role": "user", "content": "check fallback"}])
    assert response["provider"] == "mock"
    assert response["fallback_from"] == "broken"


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None, url: str = "https://example.test"):
        self.status_code = status_code
        self._payload = payload or {}
        self.url = url
        self.text = str(self._payload)

    def json(self):
        return self._payload


def test_anthropic_tools_are_converted_from_openai_shape(monkeypatch) -> None:
    captured = {}

    def fake_post(_url, headers=None, json=None, timeout=90):
        captured["payload"] = json
        return _FakeResponse(
            status_code=200,
            payload={"content": [{"type": "text", "text": '{"action":"answer","text":"ok"}'}]},
        )

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    router = ModelRouter.from_dict(
        {
            "default_model": "anthropic",
            "models": {
                "anthropic": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "supports_tools": True,
                }
            },
        }
    )
    response = router.chat(
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "list_pipelines",
                    "description": "List available pipelines.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    assert response["provider"] == "anthropic"
    tools = captured["payload"]["tools"]
    assert tools[0]["name"] == "list_pipelines"
    assert "input_schema" in tools[0]
    assert "function" not in tools[0]


def test_http_error_includes_provider_details() -> None:
    response = _FakeResponse(
        status_code=400,
        payload={"error": {"message": "bad_request", "type": "invalid_request_error"}},
        url="https://api.anthropic.com/v1/messages",
    )
    try:
        ModelRouter._raise_for_status_with_body(response, "anthropic")
    except requests.HTTPError as exc:
        text = str(exc)
        assert "anthropic api error 400" in text
        assert "invalid_request_error" in text
    else:
        raise AssertionError("Expected HTTPError")
