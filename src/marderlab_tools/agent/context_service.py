from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PATTERNS = ("*.py", "*.md", "*.yml", "*.yaml", "*.toml")
DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "processed_data",
}


@dataclass
class ContextService:
    workspace_root: Path
    max_file_size_bytes: int = 600_000
    include_patterns: tuple[str, ...] = DEFAULT_PATTERNS
    exclude_dirs: set[str] | None = None

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.expanduser().resolve()
        if self.exclude_dirs is None:
            self.exclude_dirs = set(DEFAULT_EXCLUDE_DIRS)

    def list_files(self, limit: int = 300) -> list[Path]:
        files: list[Path] = []
        for pattern in self.include_patterns:
            for path in self.workspace_root.rglob(pattern):
                if len(files) >= limit:
                    return sorted(set(files))
                if not path.is_file():
                    continue
                if any(part in self.exclude_dirs for part in path.parts):
                    continue
                try:
                    if path.stat().st_size > self.max_file_size_bytes:
                        continue
                except OSError:
                    continue
                files.append(path)
        return sorted(set(files))

    def workspace_summary(self) -> dict[str, Any]:
        py_files = list(self.workspace_root.rglob("*.py"))
        test_files = [p for p in py_files if "tests" in p.parts]
        src_files = [p for p in py_files if "src" in p.parts]
        return {
            "workspace_root": str(self.workspace_root),
            "python_files": len(py_files),
            "src_python_files": len(src_files),
            "test_files": len(test_files),
            "key_paths": [
                str(self.workspace_root / "src"),
                str(self.workspace_root / "tests"),
                str(self.workspace_root / "configs"),
                str(self.workspace_root / "README.md"),
            ],
        }

    def find_text(self, query: str, max_hits: int = 25, case_sensitive: bool = False) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(query), flags)
        hits: list[dict[str, Any]] = []

        for path in self.list_files():
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    hits.append(
                        {
                            "path": str(path),
                            "line": line_no,
                            "text": line.strip(),
                        }
                    )
                    if len(hits) >= max_hits:
                        return hits
        return hits

    def file_excerpt(self, path: str | Path, start_line: int = 1, end_line: int = 120) -> str:
        file_path = self._resolve_in_workspace(path)
        if start_line < 1:
            start_line = 1
        if end_line < start_line:
            end_line = start_line
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        selected = lines[start_line - 1 : end_line]
        return "\n".join(selected)

    def build_prompt_context(self, prompt: str, max_hits: int = 8) -> str:
        terms = self._extract_terms(prompt)
        all_hits: list[dict[str, Any]] = []
        for term in terms:
            all_hits.extend(self.find_text(term, max_hits=max_hits))
            if len(all_hits) >= max_hits:
                break
        hits = all_hits[:max_hits]
        summary = self.workspace_summary()

        lines = [
            "Workspace summary:",
            f"- root: {summary['workspace_root']}",
            f"- python files: {summary['python_files']}",
            f"- src files: {summary['src_python_files']}",
            f"- test files: {summary['test_files']}",
            "Relevant code hits:",
        ]
        if not hits:
            lines.append("- (no direct text hits found)")
        else:
            for hit in hits:
                rel = Path(hit["path"]).relative_to(self.workspace_root)
                lines.append(f"- {rel}:{hit['line']} :: {hit['text']}")
        return "\n".join(lines)

    def _resolve_in_workspace(self, path: str | Path) -> Path:
        given = Path(path)
        candidate = given if given.is_absolute() else self.workspace_root / given
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(f"Path is outside workspace: {path}") from exc
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return resolved

    @staticmethod
    def _extract_terms(prompt: str) -> list[str]:
        raw = re.findall(r"[A-Za-z_][A-Za-z0-9_\-]{2,}", prompt.lower())
        seen: set[str] = set()
        terms: list[str] = []
        for term in raw:
            if term in seen:
                continue
            seen.add(term)
            terms.append(term)
            if len(terms) >= 6:
                break
        return terms
