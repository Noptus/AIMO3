# AIMO3 Roadmap: Next 10 Feasible Upgrades

Date: 2026-02-13

Goal: maximize leaderboard score while keeping Kaggle submissions reliable and reproducible.

Latest targeted failure analysis:
- `roadmap/gptoss_error_analysis_2026-02-13.md`
- `roadmap/execution_optimization_ideas_2026-02-13.md`
- `project_prompts/gptoss120b_execution_playbook.md`

## What Is Now In Repo

- Dual orchestrators:
  - `classic` pipeline (`src/aimo3/solver.py`)
  - `langgraph` state machine (`src/aimo3/langgraph_solver.py`)
- Existing Kaggle hardened pipeline remains active (`kaggle-kernel-pipeline` with parquet validation).
- Proxy/smoke checks available for notebook and local inference harness.

## Top 10 Next Steps

1. LangGraph parity benchmark vs classic
- Run both orchestrators on the same reference splits.
- Keep a single leaderboard artifact with accuracy, fallback rate, and latency.
- Exit criteria: langgraph >= classic score with <=10% runtime overhead.

2. Stronger sandbox policy v2
- Add explicit filesystem isolation policy and denylist for file/network primitives.
- Add stricter AST checks for indirect import tricks and dynamic attribute chains.
- Add per-execution memory/time telemetry in debug traces.

3. Sandbox runner hardening on Linux/Kaggle
- Add optional process-level limits (`resource`, `prlimit`) + predictable failure classes.
- Normalize timeout/error signatures to reduce ambiguous retries.
- Add deterministic stdout truncation markers.

4. Tool quality scoring
- Score tool outputs by consistency with model rationale and modulus constraints.
- Downweight noisy tool outputs (multiple unrelated integers, traceback-heavy output).
- Track per-problem tool trust in debug JSON.

5. Geometry tool micro-kit
- Add safe helper templates for coordinates, vector dot/cross, circle/power checks.
- Expose these helpers in prompt policy for geometry category only.
- Regression pack: geometry-heavy failures from competitor analysis.

6. Adaptive budget manager v2
- Global runtime allocator across problems (easy/medium/hard buckets).
- Reserve budget for unresolved high-entropy problems late in run.
- Add hard stop guards to avoid starving final rows.

7. Structured selector output
- Force selector/verifier stages to produce compact JSON schema with confidence.
- Parse deterministically, reject malformed selector outputs, retry once.
- Improves reproducibility and reduces answer churn.

8. Modulus ambiguity resolver
- Multi-pass modulus extraction with confidence score.
- If confidence is low, run candidate-modulus checks before final normalization.
- Add parser stress tests with adversarial wording.

9. Submission risk gate
- Pre-submit gate combining:
  - fallback ratio,
  - tiny-answer ratio,
  - unresolved-problem count,
  - model/tool failure rate.
- Block low-confidence submissions automatically.

10. Expanded benchmark suite
- Build a larger internal benchmark from:
  - official reference rows,
  - synthetic AIME-style items,
  - known failure archetypes,
  - geometry/number-theory stress sets.
- Add daily regression command with pass/fail thresholds.

## Sandbox Hardening Backlog (Detailed)

- Add `SandboxPolicy` knobs for:
  - max AST depth,
  - max loop nesting,
  - max imports count,
  - max temporary file writes (should default to zero).
- Add deterministic execution transcript object:
  - code hash,
  - duration,
  - memory high-water estimate,
  - safety checks passed/failed.
- Add red-team tests for escape attempts:
  - `__subclasses__` traversals,
  - dynamic `getattr` chains,
  - exception-based exfil patterns,
  - encoded payload attempts.

## Immediate Run Plan (Next Session)

1. `benchmark-reference` with `--orchestrator classic` and `--orchestrator langgraph`.
2. `benchmark-sweep` to compare high-effort profiles under both orchestrators.
3. Keep Kaggle notebook preflight only until daily submission window opens.
4. Submit only if preflight + risk gate are both green.
