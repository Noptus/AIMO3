# LangGraph Solver Design (v1)

Date: 2026-02-13

## Objective

Add a graph-native orchestrator while preserving the existing `SolveResult` contract and CLI workflows.

## Implementation Summary

- New module: `src/aimo3/langgraph_solver.py`
- New class: `LangGraphAIMO3Solver` (subclasses `AIMO3Solver` for compatibility)
- New CLI switch: `--orchestrator {classic,langgraph}`
- Packaging: optional extra `agentic` adds `langgraph`

## Graph Topology

`START -> bootstrap -> draft -> tools -> (followup | commit) -> (draft | END)`

### Nodes

- `bootstrap`
  - parse modulus and profile
  - derive attempt/token limits
  - initialize graph state
- `draft`
  - initial prompt via `build_prompt`
  - follow-up prompt via `build_agent_followup_prompt`
  - call model backend
  - parse answer (`FINAL_ANSWER`, `\boxed{}`, line hints)
- `tools`
  - extract python blocks
  - execute in sandbox
  - collect numeric outputs + errors
  - produce compact observation for follow-up
- `followup`
  - increment round counter and loop to `draft`
- `commit`
  - build `Candidate`
  - weighted aggregation update
  - early-stop / budget / deadline checks

### Routing Conditions

- `tools -> followup` when answer is weak/absent, tool signal exists, and rounds/time permit.
- `commit -> END` when attempts exhausted, deadline near, or confidence threshold met.

## Robustness Controls

- Generation errors are captured into candidate traces (no graph crash).
- Fallback deterministic answer if all extraction paths fail.
- Works with or without stateful python tool context.
- Parallel code execution when stateful mode is disabled.

## Current Limitations

- Classic solver still has richer late-stage arbitration (selector/audit/probe).
- LangGraph v1 focuses on robust core loop first (draft/tool/followup/commit).

## Next Iteration Targets

1. Port consistency-audit and selector stages into graph branches.
2. Add checkpoint persistence for resume/replay on long runs.
3. Add graph event traces for per-node latency and failure diagnostics.
4. Add stricter sandbox telemetry and red-team policy tests.
