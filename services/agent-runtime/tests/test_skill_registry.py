"""Tests for the dynamic SkillRegistry + high-risk security overlay (Module 5)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from allcallall_agent_runtime.skill_registry import (
    RISK_HIGH,
    RISK_LOW,
    ResolvedSkill,
    SecurityOverlay,
    Skill,
    SkillRegistry,
    parse_skill_md,
    sha256_text,
)

SAMPLE_MD = textwrap.dedent(
    """
    ---
    name: meeting_recap
    risk_level: low
    tools: [query_context_chunks, query_recent_meetings]
    scope: personal
    ---
    You are a meeting recap assistant. Summarize decisions and action items.
    """
).strip()


def test_parse_skill_md_frontmatter():
    skill = parse_skill_md(SAMPLE_MD)
    assert skill.name == "meeting_recap"
    assert skill.risk_level == RISK_LOW
    assert skill.tools == ["query_context_chunks", "query_recent_meetings"]
    assert "meeting recap assistant" in skill.instructions
    assert skill.instruction_sha256  # snapshot computed


def test_snapshot_is_stable_and_tamper_evident():
    a = parse_skill_md(SAMPLE_MD)
    b = parse_skill_md(SAMPLE_MD)
    assert a.instruction_sha256 == b.instruction_sha256
    # Changing the instruction body changes the snapshot.
    modified = SAMPLE_MD + "\nAdd a risk section."
    c = parse_skill_md(modified)
    assert c.instruction_sha256 != a.instruction_sha256


def test_register_and_get():
    reg = SkillRegistry()
    reg.register(Skill(name="x", instructions="do thing", tools=["t1"]))
    assert reg.get("x") is not None
    assert reg.get("missing") is None


def test_resolve_low_risk_no_approval():
    reg = SkillRegistry()
    reg.register(parse_skill_md(SAMPLE_MD))
    resolved = reg.resolve("meeting_recap")
    assert isinstance(resolved, ResolvedSkill)
    assert resolved.requires_approval is False
    assert "meeting recap assistant" in resolved.system_instructions
    assert "SECURITY PLAN" not in resolved.system_instructions


def test_resolve_high_risk_applies_security_overlay():
    high_md = textwrap.dedent(
        """
        ---
        name: bulk_writer
        risk_level: high
        tools: [write_conversation_message, create_follow_up_task]
        ---
        Write follow-ups on behalf of the user.
        """
    ).strip()
    reg = SkillRegistry()
    reg.register(parse_skill_md(high_md))
    resolved = reg.resolve("bulk_writer")
    assert resolved.requires_approval is True
    assert resolved.risk_level == RISK_HIGH
    assert SecurityOverlay.SAFETY_PLAN in resolved.system_instructions
    # The safety plan is force-appended, original instruction preserved.
    assert "Write follow-ups on behalf of the user." in resolved.system_instructions


def test_resolve_unknown_raises():
    import pytest
    with pytest.raises(KeyError):
        SkillRegistry().resolve("nope")


def test_load_from_file_and_directory(tmp_path: Path):
    skill_a = tmp_path / "a.md"
    skill_a.write_text(SAMPLE_MD, encoding="utf-8")
    skill_b = tmp_path / "b.md"
    skill_b.write_text(
        textwrap.dedent(
            """
            ---
            name: risk_reviewer
            risk_level: high
            tools: [query_context_chunks]
            ---
            Review risks in the conversation.
            """
        ).strip(),
        encoding="utf-8",
    )
    reg = SkillRegistry()
    loaded = reg.load_directory(tmp_path)
    assert len(loaded) == 2
    assert {s.name for s in loaded} == {"meeting_recap", "risk_reviewer"}
    # High-risk one resolves with overlay; low-risk without.
    assert reg.resolve("meeting_recap").requires_approval is False
    assert reg.resolve("risk_reviewer").requires_approval is True
