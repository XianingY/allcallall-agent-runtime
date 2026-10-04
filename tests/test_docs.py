from pathlib import Path

from scripts.check_docs import (
    check_documentation_index,
    check_documentation_tree,
    check_markdown_file,
)
from scripts.portfolio_eval import DEFAULT_OUT

CANONICAL_DOCS = {
    "docs/architecture/overview.md",
    "docs/architecture/harness.md",
    "docs/reference/configuration.md",
    "docs/reference/tool-bridge-protocol.md",
    "docs/evaluation/methodology.md",
    "docs/evaluation/engineering-harness.md",
}


def write_files(root: Path, files: dict[str, str]) -> None:
    for relative_path, source in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def test_reports_broken_relative_link(tmp_path: Path) -> None:
    write_files(
        tmp_path,
        {
            "docs/README.md": "# Documentation\n\n- [Guide](guide.md)\n",
            "docs/guide.md": "# Guide\n\n[Missing](missing.md)\n",
        },
    )

    failures = check_documentation_tree(tmp_path)

    assert any("missing.md" in failure for failure in failures)


def test_reports_unindexed_maintained_document(tmp_path: Path) -> None:
    write_files(
        tmp_path,
        {
            "docs/README.md": "# Documentation\n",
            "docs/guides/orphan.md": "# Orphan\n",
        },
    )

    failures = check_documentation_tree(tmp_path)

    assert any("docs/guides/orphan.md" in failure for failure in failures)


def test_archive_is_excluded_from_index_requirement(tmp_path: Path) -> None:
    write_files(
        tmp_path,
        {
            "docs/README.md": "# Documentation\n",
            "docs/archive/report.md": "# Historical report\n",
        },
    )

    assert check_documentation_tree(tmp_path) == []


def test_reports_heading_and_code_fence_violations(tmp_path: Path) -> None:
    path = tmp_path / "docs/guide.md"
    write_files(
        tmp_path,
        {
            "docs/guide.md": "# Guide\n\n### Details\n\n# Duplicate\n\n```\nmake test\n```\n",
        },
    )

    failures = check_markdown_file(tmp_path, path)

    assert any("exactly one H1" in failure for failure in failures)
    assert any("skips from H1 to H3" in failure for failure in failures)
    assert any("language tag" in failure for failure in failures)


def test_documentation_index_accepts_indexed_document(tmp_path: Path) -> None:
    path = tmp_path / "docs/guides/guide.md"
    write_files(
        tmp_path,
        {
            "docs/README.md": "# Documentation\n",
            "docs/guides/guide.md": "# Guide\n",
        },
    )

    failures = check_documentation_index(
        tmp_path,
        [path],
        "# Documentation\n\n- [Guide](guides/guide.md)\n",
    )

    assert failures == []


def test_reports_missing_or_unlinked_governance_files(tmp_path: Path) -> None:
    write_files(
        tmp_path,
        {
            "README.md": "# Runtime\n\n[Contributing](CONTRIBUTING.md)\n",
            "docs/README.md": "# Documentation\n",
            "CONTRIBUTING.md": "# Contributing\n",
            "SECURITY.md": "# Security\n",
        },
    )

    failures = check_documentation_tree(tmp_path)

    for governance_file in (
        "LICENSE",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "SUPPORT.md",
    ):
        assert any(governance_file in failure for failure in failures)


def test_canonical_runtime_documents_exist_and_are_indexed() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [root / relative_path for relative_path in sorted(CANONICAL_DOCS)]

    missing = [path.relative_to(root).as_posix() for path in paths if not path.exists()]
    assert missing == []

    failures = check_documentation_index(
        root,
        paths,
        (root / "docs/README.md").read_text(encoding="utf-8"),
    )
    assert failures == []


def test_generated_portfolio_report_defaults_to_archive() -> None:
    root = Path(__file__).resolve().parents[1]

    assert DEFAULT_OUT.relative_to(root).as_posix() == (
        "docs/archive/reports/generated-ai-agent-portfolio-eval"
    )
