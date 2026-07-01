from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from allcallall_agent_runtime.eval_runner import run_eval as run_agent_eval
from allcallall_rag_runtime.eval_runner import (
    load_cases as load_rag_cases,
)
from allcallall_rag_runtime.eval_runner import (
    run_eval as run_rag_eval,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "generated-ai-agent-portfolio-eval"
RAG_FIXTURE = ROOT / "services" / "rag-runtime" / "evals" / "cases.json"


def build_report() -> dict[str, Any]:
    agent_report = run_agent_eval()
    rag_report = run_rag_eval(load_rag_cases(RAG_FIXTURE))
    return {
        "scope": (
            "deterministic regression and task-level eval; "
            "not online product-quality measurement"
        ),
        "agent_runtime": {
            "runtime": agent_report.runtime,
            "provider": agent_report.provider,
            "cases": {
                "passed": agent_report.summary.passed_cases,
                "total": agent_report.summary.total_cases,
            },
            "task_success_rate": agent_report.summary.task_success_rate,
            "route_accuracy": agent_report.summary.route_accuracy,
            "loop_completion_rate": agent_report.summary.loop_completion_rate,
            "stop_reason_valid_rate": agent_report.summary.stop_reason_valid_rate,
            "tool_intent_match_rate": agent_report.summary.tool_intent_match_rate,
            "approval_safety_rate": agent_report.summary.approval_safety_rate,
            "citation_coverage_rate": agent_report.summary.citation_coverage_rate,
            "grounding_check_rate": agent_report.summary.grounding_check_rate,
            "unsupported_claim_guard_rate": agent_report.summary.unsupported_claim_guard_rate,
            "memory_reflection_precision": agent_report.summary.memory_reflection_precision,
            "max_iteration_compliance_rate": agent_report.summary.max_iteration_compliance_rate,
        },
        "rag_runtime": {
            "runtime": rag_report.runtime,
            "provider": rag_report.provider,
            "cases": {
                "passed": rag_report.summary.passed_cases,
                "total": rag_report.summary.total_cases,
            },
            "rerank_top_match_rate": rag_report.summary.rerank_top_match_rate,
            "route_match_rate": rag_report.summary.route_match_rate,
            "graph_expansion_rate": rag_report.summary.graph_expansion_rate,
            "grounding_pass_rate": rag_report.summary.grounding_pass_rate,
            "sufficiency_pass_rate": rag_report.summary.sufficiency_pass_rate,
            "retrieval_refinement_success_rate": (
                rag_report.summary.retrieval_refinement_success_rate
            ),
        },
    }


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "portfolio-eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "portfolio-eval.md").write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    agent = report["agent_runtime"]
    rag = report["rag_runtime"]
    lines = [
        "# AI Agent Portfolio Eval",
        "",
        (
            "Scope: deterministic fixture eval for regression, safety boundaries, and interview "
            "discussion. These numbers are not online user-satisfaction or open-domain LLM "
            "quality claims."
        ),
        "",
        "## Agent Runtime",
        "",
        f"- Cases: `{agent['cases']['passed']}/{agent['cases']['total']}`",
        f"- Task success: `{agent['task_success_rate'] * 100:.1f}%`",
        f"- Route accuracy: `{agent['route_accuracy'] * 100:.1f}%`",
        f"- Loop completion: `{agent['loop_completion_rate'] * 100:.1f}%`",
        f"- Stop reason valid: `{agent['stop_reason_valid_rate'] * 100:.1f}%`",
        f"- Tool intent match: `{agent['tool_intent_match_rate'] * 100:.1f}%`",
        f"- Approval safety: `{agent['approval_safety_rate'] * 100:.1f}%`",
        f"- Citation coverage: `{agent['citation_coverage_rate'] * 100:.1f}%`",
        f"- Grounding check: `{agent['grounding_check_rate'] * 100:.1f}%`",
        f"- Unsupported-claim guard: `{agent['unsupported_claim_guard_rate'] * 100:.1f}%`",
        f"- Memory reflection precision: `{agent['memory_reflection_precision'] * 100:.1f}%`",
        f"- Max iteration compliance: `{agent['max_iteration_compliance_rate'] * 100:.1f}%`",
        "",
        "## RAG Runtime",
        "",
        f"- Cases: `{rag['cases']['passed']}/{rag['cases']['total']}`",
        f"- Rerank top-match rate: `{rag['rerank_top_match_rate'] * 100:.1f}%`",
        f"- Route match rate: `{rag['route_match_rate'] * 100:.1f}%`",
        f"- Graph expansion rate: `{rag['graph_expansion_rate'] * 100:.1f}%`",
        f"- Grounding pass rate: `{rag['grounding_pass_rate'] * 100:.1f}%`",
        f"- Sufficiency pass rate: `{rag['sufficiency_pass_rate'] * 100:.1f}%`",
        (
            "- Retrieval refinement success rate: "
            f"`{rag['retrieval_refinement_success_rate'] * 100:.1f}%`"
        ),
        "",
        "## Resume-Safe Wording",
        "",
        (
            "- Built a deterministic eval harness for a Python FastAPI + LangGraph Agent "
            "Runtime, covering task success, route accuracy, bounded-loop completion, "
            "stop-reason validity, tool intent match, approval safety, citation coverage, "
            "grounding, unsupported-claim guard, and memory reflection precision."
        ),
        (
            "- Built an Agentic RAG eval path covering route selection, multi-hop "
            "retrieval/refinement, graph expansion, rerank ordering, context sufficiency, "
            "and grounding checks."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AI Agent portfolio eval report")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = build_report()
    write_report(report, args.out)
    agent_cases = report["agent_runtime"]["cases"]
    rag_cases = report["rag_runtime"]["cases"]
    print(
        f"portfolio eval: agent {agent_cases['passed']}/{agent_cases['total']}, "
        f"rag {rag_cases['passed']}/{rag_cases['total']}"
    )
    passed = agent_cases["passed"] == agent_cases["total"] and (
        rag_cases["passed"] == rag_cases["total"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
