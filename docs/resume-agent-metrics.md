# Resume Agent Metrics

Use the generated report as the source of truth:

```bash
make portfolio-eval
```

Output:

- `docs/generated-ai-agent-portfolio-eval/portfolio-eval.json`
- `docs/generated-ai-agent-portfolio-eval/portfolio-eval.md`

## Metric Boundary

These are deterministic fixture metrics. They are useful for showing engineering completeness and regression stability. They are not online user satisfaction, real-world model quality, or large-scale RAG benchmark numbers.

## Resume-Safe Bullet

> 将 Agent 编排层抽象为独立 Python FastAPI Runtime，基于 LangGraph 构建 Agent Runtime Harness，统一处理意图路由、上下文注入、bounded ReAct loop、critic/reflection、trace、citation 和 approval-only tool proposal；建设 deterministic eval harness，覆盖 task success、route accuracy、loop completion、tool intent match、approval safety、citation coverage、grounding check、unsupported-claim guard 等指标。

## Interview Explanation

The important point is not that every deterministic fixture passes. The stronger engineering point is that the project has a repeatable harness:

- Agent loop behavior is observable through structured `LoopTrace`.
- Agentic RAG decisions are visible through retrieval route, attempts, evidence pack, reranked hits, rejected chunks, and grounding result.
- Write tools remain proposal-only and must pass the Go approval boundary.
- Eval metrics are separated from manual pilot samples and are not presented as production quality claims.
