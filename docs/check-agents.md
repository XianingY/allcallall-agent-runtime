# Two-Tier CheckAgent Loop

The supervisor DAG runs a bounded quality/safety loop between `critic_check`
and `approval_gate`. Implementation: `nodes/check.py`; wiring: `dag.py`.

```text
... -> critic_check -> quality_check --revise--> synthesize (bounded retry)
                            |
                          safety
                            v
                       safety_check -> approval_gate -> ...
```

## L1: `quality_check`

Consumes the `CriticResult` produced by `critic_check` (fields: `passed`,
`issues`, `context_sufficient`, `citation_coverage`) and emits a
`CheckDecision`:

| Decision | Trigger | Effect |
| --- | --- | --- |
| `PASS` | critic passed | Proceed to `safety_check`. |
| `REVISE` | `insufficient_context_guarded` issue or `context_sufficient == false` | Loop back to `synthesize`; increments `critic_retries`. |
| `ESCALATE` | `grounding_failed` issue, or revise budget exhausted | Proceed to `safety_check` with an escalation flag for downstream surfacing. |

The revise budget is `PY_AGENT_MAX_QUALITY_RETRIES` (default `1`); the loop is
strictly bounded and can never spin.

## L2: `safety_check`

Independent of content quality, verifies the safety boundary before the
approval gate:

- write intents exist only as `ToolProposal` objects with
  `approval_required=true`
- no unauthorized tool usage appears in the trace

## Routing

`route_quality` and `route_safety` are pure functions used as LangGraph
conditional edges:

```python
graph.add_edge("critic_check", "quality_check")
graph.add_conditional_edges(
    "quality_check", route_quality, {"revise": "synthesize", "safety": "safety_check"}
)
graph.add_edge("safety_check", "approval_gate")
```

## State

`state.py` additions: `critic_retries: int`, `last_check_decision: str`,
`check_log: list[dict]` (append-only outcome log; each entry records the
check name, decision, and reason).

## Relationship to the Loop Contract

The role-level bounded loops (`LoopSpec` / `LoopBudget`, see
`docs/loop-engineering.md`) bound each agent role's tool usage. The two-tier
CheckAgent bounds the overall answer-quality loop at the workflow level. Both
budgets are independent and deterministic.

## Tests

`tests/test_check_agents.py` (11 cases): L1/L2 decision matrix, retry budget
exhaustion, routing outcomes, and graph wiring.
