# Loop Engineering

AllCallAll Agent Runtime uses a bounded harness instead of an open-ended autonomous agent.

## Runtime Harness

`AllCallAllAgentHarness` is responsible for:

- request normalization and prompt-version selection
- modality metadata injection
- LangGraph workflow dispatch
- low-level trace collection
- loop trace projection
- citation and evidence pack output
- approval-only write tool proposal output
- deterministic eval replay

## Loop Contract

Each role loop is represented by:

- `LoopSpec`: role objective, max steps, allowed read tools, stop conditions
- `LoopStep`: thought summary, selected skill, tool input schema, observation, citation ids, confidence, stop reason
- `LoopBudget`: max steps, used steps, read tool calls, write proposals
- `LoopTrace`: role-level execution trace and completion status

Current bounded roles:

| role | max steps | write access | purpose |
| --- | ---: | --- | --- |
| searcher | 3 | no | collect meeting/conversation/knowledge evidence |
| risk_analyst | 2 | no | inspect approval, blocker, timeline, and policy risks |
| memory_agent | 1 | proposal only | summarize durable memory and suggest reflection writes |
| follow_up_planner | 2 | proposal only | extract action items and propose follow-up tasks |
| supervisor | 0 | no | route, merge, critic, and stop decision |

## Safety Boundary

Read tools can run automatically through the Go Tool Bridge. Write tools are never executed by Python. Python returns `ToolProposal` objects; Go validates schema, creates approvals, audits decisions, and executes accepted writes.

The critic node checks grounding, citation coverage, retrieval budget, and approval-only write safety before the approval gate.
