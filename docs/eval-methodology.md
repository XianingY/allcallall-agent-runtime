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

Portfolio report:

- `docs/generated-ai-agent-portfolio-eval/portfolio-eval.json`
- `docs/generated-ai-agent-portfolio-eval/portfolio-eval.md`

Manual pilot samples, if used in interviews, must be explicitly labeled illustrative and must not be mixed into reproducible metrics.
