#!/usr/bin/env python3
"""Validate the maintained Markdown documentation tree."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    ".worktrees",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "output",
    "vendor",
}
COMPATIBILITY_POINTER_MARKERS = (
    "compatibility page preserves an older link",
    "compatibility pointer",
    "compatibility reference",
)
ARCHIVE_STATUS_MARKER = (
    "archive status:** historical material; not part of the maintained product documentation"
)
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_archive_or_generated(relative_path: str) -> bool:
    parts = Path(relative_path).parts
    return relative_path.startswith("docs/archive/") or any(
        part.startswith("generated-") for part in parts
    )


def _is_compatibility_pointer(source: str) -> bool:
    lowered = source.lower()
    return any(marker in lowered for marker in COMPATIBILITY_POINTER_MARKERS)


def _is_archived_source(source: str) -> bool:
    return ARCHIVE_STATUS_MARKER in source.lower()


def _visible_lines(source: str) -> list[str]:
    visible: list[str] = []
    fence: tuple[str, int] | None = None

    for line in source.splitlines():
        match = FENCE_RE.match(line)
        if match is None:
            visible.append("" if fence else line)
            continue

        marker, suffix = match.groups()
        if fence is None:
            fence = (marker[0], len(marker))
        elif marker[0] == fence[0] and len(marker) >= fence[1] and not suffix.strip():
            fence = None
        visible.append("")

    return visible


def _link_destination(raw_target: str) -> str:
    target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
    return unquote(urlsplit(target).path)


def collect_markdown_files(root: Path) -> list[Path]:
    root = root.resolve()
    files: list[Path] = []
    for path in root.rglob("*.md"):
        relative_parts = path.relative_to(root).parts
        if any(part in IGNORED_DIRECTORIES for part in relative_parts[:-1]):
            continue
        files.append(path)
    return sorted(files)


def check_markdown_file(root: Path, path: Path) -> list[str]:
    root = root.resolve()
    path = path.resolve()
    relative_path = _relative(root, path)
    source = path.read_text(encoding="utf-8")
    failures: list[str] = []
    headings: list[tuple[int, int]] = []
    fence: tuple[str, int] | None = None

    for line_number, line in enumerate(source.splitlines(), start=1):
        fence_match = FENCE_RE.match(line)
        if fence_match is not None:
            marker, suffix = fence_match.groups()
            if fence is None:
                if not suffix.strip():
                    failures.append(
                        f"{relative_path}:{line_number}: opening code fence needs a language tag"
                    )
                fence = (marker[0], len(marker))
            elif (
                marker[0] == fence[0]
                and len(marker) >= fence[1]
                and not suffix.strip()
            ):
                fence = None
            continue

        if fence is not None:
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match is not None:
            headings.append((len(heading_match.group(1)), line_number))

    h1_count = sum(level == 1 for level, _ in headings)
    if h1_count != 1:
        failures.append(f"{relative_path}: expected exactly one H1, found {h1_count}")

    for (previous_level, _), (level, line_number) in zip(
        headings, headings[1:], strict=False
    ):
        if level > previous_level + 1:
            failures.append(
                f"{relative_path}:{line_number}: heading skips from H{previous_level} to H{level}"
            )

    for line_number, line in enumerate(_visible_lines(source), start=1):
        for match in LINK_RE.finditer(line):
            raw_target = match.group(1).strip()
            if raw_target.startswith("#") or re.match(
                r"^(?:[a-z][a-z0-9+.-]*:|//)", raw_target, re.IGNORECASE
            ):
                continue
            target = _link_destination(raw_target)
            if target and not (path.parent / target).resolve().exists():
                failures.append(
                    f'{relative_path}:{line_number}: relative link "{raw_target}" does not resolve'
                )

    return failures


def check_documentation_index(
    root: Path, files: list[Path], index_source: str
) -> list[str]:
    root = root.resolve()
    index_path = root / "docs/README.md"
    if not index_path.exists():
        index_path = root / "INDEX.md"
    index_relative = _relative(root, index_path)
    indexed_files: set[str] = set()

    for line in _visible_lines(index_source):
        for match in LINK_RE.finditer(line):
            raw_target = match.group(1).strip()
            if raw_target.startswith("#") or re.match(
                r"^(?:[a-z][a-z0-9+.-]*:|//)", raw_target, re.IGNORECASE
            ):
                continue
            target = _link_destination(raw_target)
            if target:
                resolved = (index_path.parent / target).resolve()
                try:
                    indexed_files.add(_relative(root, resolved))
                except ValueError:
                    continue

    failures: list[str] = []
    for path in files:
        relative_path = _relative(root, path)
        if relative_path == index_relative or _is_archive_or_generated(relative_path):
            continue
        source = path.read_text(encoding="utf-8")
        if _is_compatibility_pointer(source) or _is_archived_source(source):
            continue
        if relative_path not in indexed_files:
            failures.append(
                f"{relative_path}: maintained document is not linked from {index_relative}"
            )
    return failures


def check_documentation_tree(root: Path) -> list[str]:
    root = root.resolve()
    files = collect_markdown_files(root)
    failures: list[str] = []

    for path in files:
        relative_path = _relative(root, path)
        source = path.read_text(encoding="utf-8")
        if _is_archive_or_generated(relative_path) or _is_archived_source(source):
            continue
        failures.extend(check_markdown_file(root, path))

    index_path = root / "docs/README.md"
    if not index_path.exists():
        index_path = root / "INDEX.md"
    if not index_path.exists():
        return [*failures, "docs/README.md: canonical documentation index is missing"]

    documentation_files = [
        path for path in files if _relative(root, path).startswith("docs/")
    ]
    failures.extend(
        check_documentation_index(
            root,
            documentation_files,
            index_path.read_text(encoding="utf-8"),
        )
    )
    return failures


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = check_documentation_tree(root)
    if failures:
        print("\n".join(failures))
        print(f"\ncheck_docs: {len(failures)} problem(s)")
        return 1

    print(f"check_docs: ok ({len(collect_markdown_files(root))} Markdown file(s) checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
