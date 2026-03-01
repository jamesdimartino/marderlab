from pathlib import Path

from marderlab_tools.agent.context_service import ContextService


def test_find_text_and_excerpt(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    src = workspace / "src"
    src.mkdir(parents=True)
    file_path = src / "sample.py"
    file_path.write_text("line1\nrun_pipeline('contracture')\nline3\n", encoding="utf-8")

    ctx = ContextService(workspace_root=workspace)
    hits = ctx.find_text("run_pipeline", max_hits=5)
    assert len(hits) == 1
    assert hits[0]["line"] == 2

    excerpt = ctx.file_excerpt("src/sample.py", start_line=2, end_line=2)
    assert "run_pipeline" in excerpt


def test_workspace_summary_has_counts(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "a.py").write_text("x=1\n", encoding="utf-8")
    (workspace / "tests" / "t.py").write_text("def test_x():\n    assert 1\n", encoding="utf-8")

    ctx = ContextService(workspace_root=workspace)
    summary = ctx.workspace_summary()
    assert summary["python_files"] == 2
    assert summary["test_files"] == 1
