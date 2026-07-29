"""Tests for SkillRegistry manifest hardening (P2#22)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from allcallall_agent_runtime.skill_registry import (
    RISK_HIGH,
    RISK_LOW,
    ResolvedSkill,
    SkillRegistry,
    build_production_registry,
    stricter_risk,
)


def _write_skill(directory: Path, filename: str, *, risk_level: str) -> Path:
    path = directory / filename
    path.write_text(
        textwrap.dedent(
            f"""
            ---
            name: {filename[:-3]}
            risk_level: {risk_level}
            tools: [query_context_chunks]
            ---
            Do the thing.
            """
        ).strip(),
        encoding="utf-8",
    )
    return path


def test_stricter_risk_takes_max() -> None:
    assert stricter_risk(RISK_LOW, RISK_LOW) == RISK_LOW
    assert stricter_risk(RISK_LOW, RISK_HIGH) == RISK_HIGH
    assert stricter_risk(RISK_HIGH, RISK_LOW) == RISK_HIGH  # file cannot downgrade
    assert stricter_risk(RISK_HIGH, RISK_HIGH) == RISK_HIGH


def test_manifest_floor_blocks_self_downgrade(tmp_path: Path) -> None:
    # File self-reports LOW, but the manifest floor is HIGH -> resolved as HIGH.
    _write_skill(tmp_path, "bulk_writer.md", risk_level=RISK_LOW)
    manifest = tmp_path / "skills.yaml"
    manifest.write_text(
        textwrap.dedent(
            """
            skills:
              - name: bulk_writer
                path: bulk_writer.md
                risk_level: high
            """
        ).strip(),
        encoding="utf-8",
    )
    reg = SkillRegistry()
    reg.load_manifest(manifest)
    resolved = reg.resolve("bulk_writer")
    assert isinstance(resolved, ResolvedSkill)
    assert resolved.risk_level == RISK_HIGH
    assert resolved.requires_approval is True
    assert "SECURITY PLAN" in resolved.system_instructions


def test_manifest_cannot_be_made_looser_by_frontmatter(tmp_path: Path) -> None:
    # File self-reports HIGH, manifest floor is LOW -> still HIGH (cannot loosen).
    _write_skill(tmp_path, "risky.md", risk_level=RISK_HIGH)
    manifest = tmp_path / "skills.yaml"
    manifest.write_text(
        textwrap.dedent(
            """
            skills:
              - name: risky
                path: risky.md
                risk_level: low
            """
        ).strip(),
        encoding="utf-8",
    )
    reg = SkillRegistry()
    reg.load_manifest(manifest)
    assert reg.resolve("risky").risk_level == RISK_HIGH
    assert reg.resolve("risky").requires_approval is True


def test_manifest_low_plus_frontmatter_low_is_low(tmp_path: Path) -> None:
    _write_skill(tmp_path, "safe.md", risk_level=RISK_LOW)
    manifest = tmp_path / "skills.yaml"
    manifest.write_text(
        textwrap.dedent(
            """
            skills:
              - name: safe
                path: safe.md
                risk_level: low
            """
        ).strip(),
        encoding="utf-8",
    )
    reg = SkillRegistry()
    reg.load_manifest(manifest)
    assert reg.resolve("safe").requires_approval is False


def test_load_manifest_requires_name(tmp_path: Path) -> None:
    _write_skill(tmp_path, "x.md", risk_level=RISK_LOW)
    manifest = tmp_path / "bad.yaml"
    manifest.write_text("skills:\n  - path: x.md\n", encoding="utf-8")
    with pytest.raises(ValueError):
        SkillRegistry().load_manifest(manifest)


def test_build_production_registry_empty_without_manifest() -> None:
    reg = build_production_registry(None)
    assert reg.all() == []
    reg2 = build_production_registry(str(Path("/nonexistent/manifest.yaml")))
    assert reg2.all() == []
