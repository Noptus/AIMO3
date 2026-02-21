# Execution Optimization Ideas (AIMO3)

Date: 2026-02-13

Goal: improve solve quality under real provider latency and long-run instability.

## Strong Practical Upgrades

1. Sharded benchmark by default
- Run reference/eval in small shards (`1-3` rows) and merge results.
- Prevents one stalled problem from invalidating a full run.

2. Dynamic stage budget floor
- Reserve explicit downstream time for verifier/audit/selector.
- Stop launching new initial attempts once the floor is threatened.

3. First-batch throttling under deadline mode
- Under hard per-problem budgets, execute one initial attempt first (not parallel burst).
- Then branch only if time remains after stage reserve.

4. Stage-level timeout ladder
- Initial attempts: moderate timeout.
- Arbitration stages: short timeout + retry.
- Reduces dead-time on expensive unresolved calls.

5. Evidence-aware retries
- Retry only when top answer has no code evidence and no stage diversity.
- Avoid wasting retries on already well-supported candidates.

6. Adaptive parallelism
- Lower `parallel_attempt_workers` when provider latency rises.
- Increase only when recent shard completion time is stable.

7. Mandatory contradiction check for hard number theory
- Require one of: modular contradiction, bound contradiction, parity contradiction.
- If absent, trigger one short targeted follow-up.

8. Candidate collapse protection
- If all candidates are small (`0/1`) or statement echoes, auto-trigger guard stage.
- Do not finalize without at least one additional challenge pass.

9. Structured decision telemetry
- Store skip reasons for each stage (`time_budget`, `not_triggered`, `no_candidates`).
- Use these for automated harness tuning.

10. Lightweight runbook automation
- One command to: generate stress set -> run sharded benchmark -> summarize failure archetypes.
- Keep daily iteration friction low.

## Immediate Recommended Config

- Profile: `aimo120b`
- `--per-problem-time-sec 90`
- `--parallel-attempt-workers 2`
- `--parallel-code-workers 2`
- `--agentic-tool-rounds 4`
- `--request-timeout 90`
- `--client-max-retries 2`
- Shard size: `2`

