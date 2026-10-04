# AI Agent Portfolio Eval

Scope: deterministic fixture eval for regression, safety boundaries, and interview discussion. These numbers are not online user-satisfaction or open-domain LLM quality claims.

## Agent Runtime

- Cases: `9/9`
- Task success: `100.0%`
- Route accuracy: `100.0%`
- Loop completion: `100.0%`
- Stop reason valid: `100.0%`
- Tool intent match: `100.0%`
- Approval safety: `100.0%`
- Citation coverage: `100.0%`
- Grounding check: `100.0%`
- Unsupported-claim guard: `100.0%`
- Memory reflection precision: `100.0%`
- Max iteration compliance: `100.0%`

## RAG Runtime

- Cases: `3/3`
- Rerank top-match rate: `100.0%`
- Route match rate: `100.0%`
- Graph expansion rate: `100.0%`
- Grounding pass rate: `100.0%`
- Sufficiency pass rate: `100.0%`
- Retrieval refinement success rate: `100.0%`

## Resume-Safe Wording

- Built a deterministic eval harness for a Python FastAPI + LangGraph Agent Runtime, covering task success, route accuracy, bounded-loop completion, stop-reason validity, tool intent match, approval safety, citation coverage, grounding, unsupported-claim guard, and memory reflection precision.
- Built an Agentic RAG eval path covering route selection, multi-hop retrieval/refinement, graph expansion, rerank ordering, context sufficiency, and grounding checks.
