# Full Groq Run Log (2026-02-13)

## Command profile

- Model: `openai/gpt-oss-120b` (forced policy)
- Provider: Groq OpenAI-compatible API
- Profile: `autonomous120b`
- Key controls:
  - `--force-full-problem-time`
  - `--mandatory-code-attempts 3`
  - `--force-tool-round-for-unverified`
  - `--agentic-tool-rounds 5`
  - `--parallel-attempt-workers 4`
  - `--parallel-code-workers 4`
  - `--strict-zero-one-policy`

## Run 1

- Output: `artifacts/groq_full_robust_20260213_230948`
- Result: `5/10` (`50.0%`)
- Wrong ids: `424e18`, `641659`, `86e8e5`, `a295e9`, `dd7f5e`
- Observed issue: stage-reserve throttling plus high per-attempt latency left many rows with single-candidate decisions.

## Run 2

- Output: `artifacts/groq_full_robust_v2_20260213_233550`
- Result: `6/10` (`60.0%`)
- Small-answer rate: `10%`
- Wrong ids: `0e644e`, `424e18`, `86e8e5`, `dd7f5e`
- Improvement: reduced small-answer collapse and fixed `a295e9`.

## Post-run harness fixes applied

- Added reserve-cap logic for realistic per-problem budgets.
- Added minimum-two-initial-attempts rule for realistic budgets before reserve throttling.
- Extended debug export with:
  - `method`
  - `independent_check_passed`
  - `missing_forced_code_check`

## Remaining bottleneck

- Hard rows still often converge to weak single-candidate outcomes under high-latency API conditions.
- Next tuning target: dynamic adaptation of
  - `min_time_for_attempt_sec`
  - `agentic_tool_rounds`
  - `max_tokens`
  by observed first-attempt runtime.
