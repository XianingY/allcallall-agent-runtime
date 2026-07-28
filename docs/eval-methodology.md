# Eval Methodology

The eval suite is deterministic and fixture-based.

It measures:

- task success
- tool intent match
- approval safety
- citation grounding
- retrieval refinement
- dynamic route match
- knowledge-graph expansion
- rerank ordering
- context sufficiency
- max-iteration compliance
- unsupported-claim guarding

These numbers are regression evidence for the runtime behavior and safety boundary. They are not open-domain LLM quality claims.

Recommended commands:

```bash
make agent-eval
make rag-eval
make portfolio-eval
```

Current deterministic evidence:

- Agent runtime: 9/9 fixtures pass across meeting brief, risk review, follow-up planning, context QA, approval safety, memory upsert proposals, and unsupported-claim guarding.
- RAG runtime: 3/3 fixtures pass with 100% route match and graph expansion success on graph-required cases.

IR-metric regression anchors (`engineering_harness.py`, 120-case fixture in
`tests/test_engineering_harness.py`; metrics mirrored from the Go evaluator
so both stacks report comparable numbers — see
`docs/engineering-harness.md`):

- `HitRate@5 = 0.9667`
- `MRR = 0.9083`

Portfolio report:

- `docs/generated-ai-agent-portfolio-eval/portfolio-eval.json`
- `docs/generated-ai-agent-portfolio-eval/portfolio-eval.md`

Manual pilot samples, if used in interviews, must be explicitly labeled illustrative and must not be mixed into reproducible metrics.
