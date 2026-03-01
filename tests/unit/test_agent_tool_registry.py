from pathlib import Path

from marderlab_tools.agent.context_service import ContextService
from marderlab_tools.agent.tool_registry import ToolRegistry


def test_tool_registry_lists_and_runs_tools(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir(parents=True)
    (workspace / "README.md").write_text("contracture pipeline\n", encoding="utf-8")

    registry = ToolRegistry(context=ContextService(workspace), pipelines=["contracture"])
    tools = registry.list_tools()
    names = {item["name"] for item in tools}
    assert "list_pipelines" in names
    assert "search_code" in names

    payload = registry.run_tool("list_pipelines")
    assert payload["ok"]
    assert payload["result"]["pipelines"] == ["contracture"]

    search = registry.run_tool("search_code", {"query": "contracture"})
    assert search["ok"]
    assert len(search["result"]["hits"]) == 1


def test_tool_registry_rejects_unknown_tool(tmp_path: Path) -> None:
    registry = ToolRegistry(context=ContextService(tmp_path))
    payload = registry.run_tool("does-not-exist")
    assert not payload["ok"]
