from pathlib import Path

from marderlab_tools.agent.agent_loop import AgentLoop
from marderlab_tools.agent.context_service import ContextService
from marderlab_tools.agent.model_router import ModelRouter
from marderlab_tools.agent.tool_registry import ToolRegistry


def test_agent_loop_returns_plain_answer_with_mock_router(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir(parents=True)
    router = ModelRouter.from_dict(
        {
            "default_model": "mock",
            "models": {"mock": {"provider": "mock", "model": "marder-mock"}},
        }
    )
    loop = AgentLoop(
        router=router,
        context=ContextService(workspace),
        tools=ToolRegistry(ContextService(workspace)),
    )
    response = loop.ask("Explain run-all.")
    assert "run-all" in response.text
    assert response.provider == "mock"


class _ScriptedRouter:
    def __init__(self) -> None:
        self.calls = 0

    def list_models(self) -> list[str]:
        return ["scripted"]

    def chat(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return {
                "model": "scripted",
                "provider": "mock",
                "content": '{"action":"tool","name":"workspace_summary","args":{}}',
            }
        return {
            "model": "scripted",
            "provider": "mock",
            "content": '{"action":"answer","text":"done"}',
        }


def test_agent_loop_executes_tool_then_answers(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir(parents=True)
    loop = AgentLoop(
        router=_ScriptedRouter(),  # type: ignore[arg-type]
        context=ContextService(workspace),
        tools=ToolRegistry(ContextService(workspace)),
    )
    response = loop.ask("Use tool then answer.")
    assert response.text == "done"
    assert len(response.steps) == 1
    assert response.steps[0]["tool"] == "workspace_summary"


class _NativeToolRouter:
    def __init__(self) -> None:
        self.calls = 0

    def list_models(self) -> list[str]:
        return ["native-tool-router"]

    def chat(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return {
                "model": "native-tool-router",
                "provider": "anthropic",
                "content": "I'll check available pipelines.",
                "tool_calls": [{"id": "t1", "name": "list_pipelines", "args": {}}],
            }
        return {
            "model": "native-tool-router",
            "provider": "anthropic",
            "content": '{"action":"answer","text":"Available pipelines listed."}',
        }


def test_agent_loop_executes_native_tool_calls(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir(parents=True)
    loop = AgentLoop(
        router=_NativeToolRouter(),  # type: ignore[arg-type]
        context=ContextService(workspace),
        tools=ToolRegistry(ContextService(workspace), pipelines=["contracture"]),
    )
    response = loop.ask("What pipelines can you run end to end?")
    assert response.text == "Available pipelines listed."
    tools_used = [step["tool"] for step in response.steps]
    assert "list_pipelines" in tools_used


class _EmptyContentRouter:
    def list_models(self) -> list[str]:
        return ["empty-router"]

    def chat(self, *_args, **_kwargs):
        return {"model": "empty-router", "provider": "anthropic", "content": ""}


def test_agent_loop_never_returns_blank_content(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir(parents=True)
    loop = AgentLoop(
        router=_EmptyContentRouter(),  # type: ignore[arg-type]
        context=ContextService(workspace),
        tools=ToolRegistry(ContextService(workspace), pipelines=["contracture", "nerve_evoked"]),
    )
    response = loop.ask("What analyses can I run now?")
    assert response.text
    assert "contracture" in response.text
