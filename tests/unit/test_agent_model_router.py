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
