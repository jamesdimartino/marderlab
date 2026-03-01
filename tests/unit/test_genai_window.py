from pathlib import Path

import yaml

from marderlab_tools.app.genai_window import (
    load_agent_config,
    run_single_prompt,
)


def test_load_agent_config_defaults_when_missing(tmp_path: Path) -> None:
    cfg = load_agent_config(tmp_path / "missing.yml")
    assert "router" in cfg
    assert cfg["router"]["default_model"] == "mock-local"


def test_run_single_prompt_with_custom_agent_config(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "router": {
                    "default_model": "mock",
                    "models": {"mock": {"provider": "mock", "model": "marder-mock"}},
                }
            }
        ),
        encoding="utf-8",
    )
    result = run_single_prompt(
        prompt="hello",
        agent_config_path=config_path,
        workspace_root=tmp_path,
    )
    assert result["provider"] == "mock"
    assert "hello" in result["text"]
    assert "contract" in result
    assert result["contract"]["version"] == "1.0"


def test_run_single_prompt_persists_user_preferences_file(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "router": {
                    "default_model": "mock",
                    "models": {"mock": {"provider": "mock", "model": "marder-mock"}},
                }
            }
        ),
        encoding="utf-8",
    )
    run_single_prompt(
        prompt="hello",
        agent_config_path=config_path,
        workspace_root=tmp_path,
    )
    prefs_path = tmp_path / "reports" / "user_preferences_context.yml"
    assert prefs_path.exists()
