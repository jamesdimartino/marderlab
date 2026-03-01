from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from marderlab_tools.agent import AgentLoop, ContextService, ModelRouter, ToolRegistry


USER_PREFERENCES_RELATIVE_PATH = Path("reports") / "user_preferences_context.yml"


DEFAULT_AGENT_CONFIG = {
    "system_prompt": "You are a practical assistant for the MarderLab codebase.",
    "max_steps": 4,
    "policy": {
        "require_tool_for_data_requests": True,
        "require_successful_tool_for_data_requests": True,
        "ask_clarifying_questions_first": True,
        "prefer_processed_data": True,
        "missing_data_behavior": "ask_user",
        "default_stats_test": "t_test",
        "response_contract_version": "1.0",
        "status_update_interval_minutes": 5,
    },
    "user_preferences": {
        "workflow_priorities": ["figure_generation", "contracture_processing", "hikcontrol"],
        "output_preferences": {
            "directories": "library_defaults",
            "plot_format": "library_defaults",
        },
        "analysis_preferences": {
            "default_stats_test": "t_test",
            "clarify_before_execution": True,
            "always_use_tools_for_data_requests": True,
        },
        "design_preferences": {
            "figure_style": "not_set",
            "theme": "not_set",
        },
    },
    "router": {
        "default_model": "mock-local",
        "fallback_model": "mock-local",
        "models": {
            "mock-local": {
                "provider": "mock",
                "model": "marder-mock",
                "supports_tools": True,
            }
        },
    },
}


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _user_preferences_path(workspace_root: Path) -> Path:
    return workspace_root / USER_PREFERENCES_RELATIVE_PATH


def load_user_preferences(workspace_root: str | Path, config_prefs: dict[str, Any] | None = None) -> dict[str, Any]:
    workspace = Path(workspace_root).expanduser().resolve()
    prefs_path = _user_preferences_path(workspace)
    disk_prefs: dict[str, Any] = {}
    if prefs_path.exists():
        with prefs_path.open("r", encoding="utf-8") as handle:
            disk_prefs = yaml.safe_load(handle) or {}
    prefs = _merge_dicts(DEFAULT_AGENT_CONFIG["user_preferences"], disk_prefs)
    if config_prefs:
        prefs = _merge_dicts(prefs, config_prefs)
    return prefs


def save_user_preferences(workspace_root: str | Path, preferences: dict[str, Any]) -> Path:
    workspace = Path(workspace_root).expanduser().resolve()
    path = _user_preferences_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(preferences, sort_keys=False), encoding="utf-8")
    return path


def load_agent_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        return DEFAULT_AGENT_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if "router" not in raw:
        raw["router"] = DEFAULT_AGENT_CONFIG["router"]
    router = dict(raw.get("router", {}))
    if "models" not in router or not router.get("models"):
        router["models"] = DEFAULT_AGENT_CONFIG["router"]["models"]
    if "default_model" not in router or not str(router.get("default_model", "")).strip():
        router["default_model"] = DEFAULT_AGENT_CONFIG["router"]["default_model"]
    if str(router["default_model"]) not in router["models"]:
        router["default_model"] = next(iter(router["models"].keys()))
    if "fallback_model" not in router:
        router["fallback_model"] = router["default_model"]
    raw["router"] = router
    if "system_prompt" not in raw:
        raw["system_prompt"] = DEFAULT_AGENT_CONFIG["system_prompt"]
    if "max_steps" not in raw:
        raw["max_steps"] = DEFAULT_AGENT_CONFIG["max_steps"]
    raw["policy"] = _merge_dicts(DEFAULT_AGENT_CONFIG["policy"], dict(raw.get("policy", {})))
    raw["user_preferences"] = _merge_dicts(
        DEFAULT_AGENT_CONFIG["user_preferences"],
        dict(raw.get("user_preferences", {})),
    )
    return raw


def make_agent(agent_config_path: str | Path, workspace_root: str | Path) -> AgentLoop:
    cfg = load_agent_config(agent_config_path)
    workspace_path = Path(workspace_root).expanduser().resolve()
    merged_preferences = load_user_preferences(workspace_path, cfg.get("user_preferences"))
    save_user_preferences(workspace_path, merged_preferences)
    router = ModelRouter.from_dict(cfg.get("router", {}))
    context = ContextService(workspace_path)
    pipelines = list(
        cfg.get(
            "pipelines",
            [
                "contracture",
                "nerve_evoked",
                "hikcontrol",
                "control",
                "dualhik",
                "freqrange",
                "gm56acclim",
                "gm56weaklink",
                "muscle",
                "heartbeat",
                "rawheart",
            ],
        )
    )
    default_pipeline_config = str(cfg.get("pipeline_config_path", "configs/default.yml"))
    tools = ToolRegistry(
        context=context,
        pipelines=pipelines,
        default_config_path=default_pipeline_config,
    )
    system_prompt = str(cfg.get("system_prompt", ""))
    max_steps = int(cfg.get("max_steps", 4))
    policy = _merge_dicts(
        dict(cfg.get("policy", {})),
        {
            "default_stats_test": str(
                merged_preferences.get("analysis_preferences", {}).get("default_stats_test", "t_test")
            ),
            "ask_clarifying_questions_first": bool(
                merged_preferences.get("analysis_preferences", {}).get("clarify_before_execution", True)
            ),
            "require_tool_for_data_requests": bool(
                merged_preferences.get("analysis_preferences", {}).get("always_use_tools_for_data_requests", True)
            ),
        },
    )
    agent_loop_init = inspect.signature(AgentLoop.__init__)
    kwargs: dict[str, Any] = {
        "router": router,
        "context": context,
        "tools": tools,
        "system_prompt": system_prompt,
        "max_steps": max_steps,
    }
    if "policy" in agent_loop_init.parameters:
        kwargs["policy"] = policy
    return AgentLoop(**kwargs)


def run_single_prompt(
    prompt: str,
    agent_config_path: str | Path,
    workspace_root: str | Path,
    model_name: str | None = None,
    conversation: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    agent = make_agent(agent_config_path=agent_config_path, workspace_root=workspace_root)
    response = agent.ask(prompt=prompt, model_name=model_name, conversation=conversation)
    contract = getattr(response, "contract", {}) or {}
    return {
        "text": response.text,
        "model": response.model,
        "provider": response.provider,
        "steps": response.steps,
        "raw": response.raw,
        "contract": contract,
    }


def launch_streamlit_window(
    agent_config_path: str | Path,
    workspace_root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8501,
    open_browser: bool = False,
) -> int:
    script_path = Path(__file__).resolve()
    workspace_path = Path(workspace_root).expanduser().resolve()
    env = os.environ.copy()
    env["MARDER_AGENT_CONFIG"] = str(Path(agent_config_path).expanduser().resolve())
    env["MARDER_WORKSPACE_ROOT"] = str(workspace_path)
    src_path = str((workspace_path / "src").resolve())
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        env["PYTHONPATH"] = src_path + os.pathsep + existing_pythonpath
    else:
        env["PYTHONPATH"] = src_path

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path),
        "--server.address",
        host,
        "--server.port",
        str(int(port)),
        "--server.headless",
        "false" if open_browser else "true",
    ]
    completed = subprocess.run(cmd, env=env, check=False)
    return int(completed.returncode)


def _chat_record_path(workspace_root: Path) -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return workspace_root / ".cache" / "marderlab" / "agent_chat" / f"chat_{stamp}.json"


def _save_chat_record(workspace_root: Path, payload: dict[str, Any]) -> Path:
    path = _chat_record_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _agent_audit_record_path(workspace_root: Path) -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return workspace_root / ".cache" / "marderlab" / "agent_audit" / f"audit_{stamp}.json"


def _save_agent_audit_record(workspace_root: Path, payload: dict[str, Any]) -> Path:
    path = _agent_audit_record_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _run_streamlit_ui() -> None:
    import streamlit as st  # type: ignore

    default_cfg = os.environ.get("MARDER_AGENT_CONFIG", "configs/genai.yml")
    default_workspace = os.environ.get("MARDER_WORKSPACE_ROOT", str(Path.cwd()))

    st.set_page_config(page_title="MarderLab GenAI Window", layout="wide")
    st.title("MarderLab GenAI Window")
    st.caption("Code-aware assistant with tool access and configurable model routing.")

    with st.sidebar:
        st.header("Session")
        agent_config = st.text_input("Agent Config", value=default_cfg)
        workspace_root = st.text_input("Workspace Root", value=default_workspace)
        if st.button("Reset Chat"):
            st.session_state["chat_messages"] = []
            st.session_state["chat_history"] = []
            st.session_state["last_record"] = ""

    workspace_path = Path(workspace_root).expanduser().resolve()
    prefs_path = _user_preferences_path(workspace_path)
    agent = make_agent(agent_config_path=agent_config, workspace_root=workspace_path)
    models = agent.router.list_models()
    selected_model = st.sidebar.selectbox("Model", models, index=0 if models else None)
    st.sidebar.caption(f"Preferences: {prefs_path}")

    st.subheader("Workspace + Tools")
    c1, c2 = st.columns(2)
    with c1:
        st.json(agent.context.workspace_summary())
    with c2:
        st.json({"tools": agent.tools.list_tools()})

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    prompt = st.chat_input("Ask about code, pipelines, commands, or debugging steps.")
    if prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    result = run_single_prompt(
                        prompt=prompt,
                        agent_config_path=agent_config,
                        workspace_root=workspace_path,
                        model_name=selected_model,
                        conversation=st.session_state["chat_history"],
                    )
                st.markdown(result["text"] or "_(no content)_")
                contract = result.get("contract", {})
                meta = (
                    f"model={result['model']} provider={result['provider']} "
                    f"tools_used={len(result['steps'])} status={contract.get('status', 'unknown')}"
                )
                st.caption(meta)
                if result["steps"]:
                    with st.expander("Tool calls"):
                        st.json(result["steps"])
                if contract.get("clarifying_questions"):
                    with st.expander("Clarifying Questions"):
                        st.json(contract.get("clarifying_questions"))
                if contract.get("failure_reasons"):
                    with st.expander("Failure Reasons"):
                        st.json(contract.get("failure_reasons"))
            except Exception as exc:
                result = {
                    "text": f"Provider request failed: {exc}",
                    "model": selected_model or "unknown",
                    "provider": "error",
                    "steps": [],
                    "raw": {},
                    "contract": {
                        "version": "1.0",
                        "status": "failed",
                        "requires_user_input": True,
                        "clarifying_questions": [],
                        "failure_reasons": [str(exc)],
                        "tool_call_count": 0,
                        "successful_tool_call_count": 0,
                        "status_update_interval_minutes": 5,
                    },
                }
                meta = f"model={result['model']} provider={result['provider']} tools_used=0"
                st.error(result["text"])
                st.caption(meta)

        st.session_state["chat_messages"].append(
            {"role": "assistant", "content": result["text"], "meta": meta}
        )
        st.session_state["chat_history"].append({"role": "assistant", "content": result["text"]})

        record = {
            "saved_at": datetime.now(tz=UTC).isoformat(),
            "agent_config": str(agent_config),
            "workspace_root": str(workspace_path),
            "model": selected_model,
            "messages": st.session_state["chat_messages"],
        }
        record_path = _save_chat_record(workspace_path, record)
        _save_agent_audit_record(
            workspace_path,
            {
                "saved_at": datetime.now(tz=UTC).isoformat(),
                "prompt": prompt,
                "model": result.get("model"),
                "provider": result.get("provider"),
                "steps": result.get("steps", []),
                "contract": result.get("contract", {}),
                "raw": result.get("raw", {}),
            },
        )
        st.session_state["last_record"] = str(record_path)

    if st.session_state.get("last_record"):
        st.info(f"Chat saved: {st.session_state['last_record']}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MarderLab GenAI utility")
    parser.add_argument("--prompt", default="", help="Run single prompt in CLI mode.")
    parser.add_argument("--agent-config", default="configs/genai.yml")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--model", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.prompt:
        result = run_single_prompt(
            prompt=args.prompt,
            agent_config_path=args.agent_config,
            workspace_root=args.workspace_root,
            model_name=args.model or None,
        )
        print(json.dumps(result, indent=2))
        return 0
    _run_streamlit_ui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
