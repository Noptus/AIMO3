# GPT-OSS 120B Error Analysis and Harness Upgrade Plan

Date: 2026-02-13  
Run analyzed: `artifacts/groq_long_reference_live_20260213_214155`

## Observed Failure Modes

1. Misclassification of AIMO problem type:
- `86e8e5` is divisor/number-theory heavy but was routed as `functional_equation` (`category=algebra`).
- Impact: weaker tactic prompt, lower chance of reliable modular/divisor workflow.

2. Complexity underestimated:
- `86e8e5` and `a295e9` were marked `easy`.
- Impact: attempt budget spent on initial traces without sufficient downstream arbitration.

3. Arbitration stages starved by wall-clock:
- For both wrong rows, only `initial` stage candidates were present.
- `verifier`, `consistency_audit`, `selector`, and `recovery` stages did not execute.
- Impact: fragile single-trace answers were selected (`0`, `125`) without cross-check.

4. Weak-evidence answer promotion:
- Wrong answers came from textual extraction (`answer_line` / `final_answer_tag`) without code verification.
- Impact: high-scoring but unsupported textual candidates can win.

## Harness Fixes Implemented in This Iteration

1. Better archetype/category routing (`src/aimo3/prompts.py`):
- Prioritize divisor/combinatorial patterns before functional-equation pattern.
- Add rectangle/partition/perimeter optimization cues for combinatorics routing.
- Reduce false `functional_equation` triggers from helper notation `f(n)`.

2. Better complexity scoring (`src/aimo3/prompts.py`):
- Fixed factorial and exponent regexes to catch AIMO-style large expressions.
- Added optimization/perimeter/rectangle/floor-pattern hardness cues.
- Added large-integer density bonus.

3. Stronger stage reserve behavior (`src/aimo3/solver.py`):
- Reserve now includes an additional attempt cushion when launching a new attempt.
- Prevents spending remaining time on one more initial attempt when downstream stages are enabled.

4. Deadline-aware parallel throttling (`src/aimo3/solver.py`):
- When per-problem deadlines and stage reserve are active, the first initial batch is forced to one attempt.
- Prevents an initial parallel burst from consuming downstream-stage budget.

5. Regression tests:
- `tests/test_prompts.py`: misclassification regressions for the two failing archetypes.
- `tests/test_solver.py`: reserve-cushion behavior to protect downstream stage time.
- `tests/test_solver.py`: parallel initial-batch throttling under stage reserve.

6. Kaggle mirror sync:
- Synced updated solver/prompt files to `kaggle_kernel_submission/src/aimo3/`.

## Validation Snapshots

- Baseline long run (before these fixes): `8/10` on local reference.
- Focused failure-pair probe after first patch set: `0/2` (`86e8e5`, `a295e9`).
- Focused failure-pair probe after reserve+throttling patch: `1/2` (`a295e9` fixed, `86e8e5` still wrong).

Interpretation:
- Classification/complexity routing is improved.
- Budget starvation is reduced but not fully solved for very hard long-form number-theory items.
- Remaining gap is mostly deep-solve reliability (not only extraction).

## Top 10 Next Improvements (Feasible, High-Impact)

1. Stage heartbeat telemetry:
- Log per-stage start/skip reason and remaining seconds in debug summary.

2. Fast emergency arbitration:
- If only initial-stage disagreement exists near deadline, run one ultra-short selector pass.

3. Anti-trivial confidence gate:
- Require independent support before accepting `0/1` in nontrivial statements.

4. Single-candidate verifier:
- Add a short “disprove this candidate” stage when only one unverified answer exists.

5. Score calibration by evidence diversity:
- Penalize candidates that win only via one extraction source and one stage.

6. Prompt-level hard constraints for optimization problems:
- Enforce bound checks and parity/mod sanity checks before final answer.

7. Timeout-adaptive max tokens:
- Scale per-attempt `max_tokens` by remaining time and problem complexity.

8. Category-specific tool templates:
- Rectangle/combinatorics helper templates for constructive counting and sanity checks.

9. Retry policy by stage criticality:
- More retries for verifier/selector than for early exploratory attempts.

10. Benchmark diff dashboard:
- Auto-compare two runs and report per-problem stage coverage and answer-source drift.
