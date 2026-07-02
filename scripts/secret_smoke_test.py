#!/usr/bin/env python
"""Small repository smoke test for accidentally committed credential strings."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", ".venv", "data/downloaded"}
TEXT_SUFFIXES = {
    ".c",
    ".cpp",
    ".csv",
    ".js",
    ".json",
    ".md",
    ".py",
    ".r",
    ".sql",
    ".toml",
    ".txt",
    ".yml",
}


def sensitive_patterns() -> list[str]:
    """Build patterns without storing the literal needles in this file."""

    return [
        "olp" + "_",
        "gho" + "_",
        "git" + "." + "overleaf" + "." + "com",
        "649e" + "127da62fd2b03dd83d37",
        "64b2" + "6996fb5be5f760ea185b",
        "6438" + "f25609a95e051021d3b1",
        "640b" + "b5ef3aa60f30418d2b26",
    ]


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield path


def main() -> int:
    hits: list[str] = []
    for path in iter_text_files(ROOT):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in sensitive_patterns():
            if pattern in text:
                hits.append(f"{path.relative_to(ROOT)} contains blocked credential/source string")
    if hits:
        for hit in hits:
            print(hit, file=sys.stderr)
        return 1
    print("Secret smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
