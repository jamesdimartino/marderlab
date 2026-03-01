from pathlib import Path

import yaml

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
    assert "resolve_request_context" in names

    payload = registry.run_tool("list_pipelines")
    assert payload["ok"]
    assert payload["result"]["pipelines"] == ["contracture"]

    search = registry.run_tool("search_code", {"query": "contracture"})
    assert search["ok"]
    assert len(search["result"]["hits"]) == 1


def test_resolve_request_context_maps_dual10xk_to_dualhik(tmp_path: Path) -> None:
    registry = ToolRegistry(context=ContextService(tmp_path))
    payload = registry.run_tool(
        "resolve_request_context",
        {"prompt": "generate a graph of contracture amplitude for gm6 vs cpv46p1 from dual10xk experiments"},
    )
    assert payload["ok"]
    result = payload["result"]
    assert result["is_data_request"] is True
    assert "dualhik" in result["candidate_pipelines"]
    assert "dualhik.ipynb" in result["candidate_notebooks"]


def test_tool_registry_rejects_unknown_tool(tmp_path: Path) -> None:
    registry = ToolRegistry(context=ContextService(tmp_path))
    payload = registry.run_tool("does-not-exist")
    assert not payload["ok"]


def _write_config(path: Path, raw_root: Path, cache_root: Path) -> None:
    payload = {
        "paths": {
            "raw_data_root": str(raw_root),
            "processed_root": str(raw_root / "processed"),
            "cache_root": str(cache_root),
        },
        "metadata": {
            "google_sheet_url": "https://example.com/sheet/pubhtml",
            "tabs": ["FTBath"],
            "cache_csv": str(cache_root / "metadata.csv"),
            "required_fields": [
                "notebook_page",
                "file_index",
                "stim_index",
                "temperature",
                "condition",
                "experiment_type",
                "season",
            ],
            "column_map": {},
        },
        "pipelines": {
            "contracture": {
                "metadata_tabs": ["FTBath"],
                "experiment_type_values": ["contracture"],
                "force_channel": "force",
                "trig_channel": "trig",
            }
        },
        "checks": {"fail_mode": "experiment_only"},
        "stats": {"default_rule": "by_group_count"},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_build_run_command_and_preview(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    raw = workspace / "raw"
    cache = workspace / "cache"
    raw.mkdir(parents=True)
    cache.mkdir(parents=True)
    (raw / "997_201").mkdir()
    (raw / "997_201" / "997_201_0001.abf").write_bytes(b"")
    (cache / "metadata.csv").write_text(
        "notebook_page,file_index,stim_index,temperature,condition,experiment_type,season,source_tab\n"
        "997_201,1,1,10.0,control,contracture,20,FTBath\n",
        encoding="utf-8",
    )

    cfg = workspace / "config.yml"
    _write_config(cfg, raw, cache)

    registry = ToolRegistry(
        context=ContextService(workspace),
        pipelines=["contracture"],
        default_config_path="config.yml",
    )

    cmd_payload = registry.run_tool(
        "build_run_command",
        {"mode": "run", "pipeline": "contracture", "pages": "997_201", "plots": "true"},
    )
    assert cmd_payload["ok"]
    command = cmd_payload["result"]["command"]
    assert "marder run" in command
    assert "--pipeline contracture" in command
    assert "--pages 997_201" in command
    assert "--plots" in command

    preview = registry.run_tool(
        "preview_pipeline_experiments",
        {"pipeline": "contracture", "config_path": "config.yml"},
    )
    assert preview["ok"]
    assert preview["result"]["selected_count"] == 1
    assert preview["result"]["selected_pages"] == ["997_201"]


def test_validate_pipeline_config_reports_checks(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    raw = workspace / "raw"
    cache = workspace / "cache"
    raw.mkdir(parents=True)
    cache.mkdir(parents=True)
    cfg = workspace / "config.yml"
    _write_config(cfg, raw, cache)

    registry = ToolRegistry(
        context=ContextService(workspace),
        pipelines=["contracture"],
        default_config_path="config.yml",
    )
    payload = registry.run_tool("validate_pipeline_config")
    assert payload["ok"]
    assert payload["result"]["ok"]
    assert len(payload["result"]["checks"]) >= 3
