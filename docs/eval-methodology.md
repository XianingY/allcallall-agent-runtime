# Eval Methodology

The eval suite is deterministic and fixture-based.

It measures:

- task success
- tool intent match
- approval safety
- citation grounding
- retrieval refinement
- rerank ordering
- context sufficiency
- max-iteration compliance
- unsupported-claim guarding

These numbers are regression evidence for the runtime behavior and safety boundary. They are not open-domain LLM quality claims.

Recommended commands:

```bash
make agent-eval
make rag-eval
```

