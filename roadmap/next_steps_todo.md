# AIMO3 Next Roadmap (Top 10 Feasible Ideas)

Date: 2026-02-11

This list is ranked by expected score impact per implementation effort.

## 1) Two-model disagreement escalation

- Impact: High
- Effort: Medium
- Feasible because: pipeline already supports model/config switching.
- Plan: run 20B first, escalate only disagreement/low-confidence items to 120B.

## 2) Per-problem dynamic compute allocator

- Impact: High
- Effort: Medium
- Feasible because: solver already has complexity profile and stage controls.
- Plan: budget manager with easy/medium/hard caps + banked unused time.

## 3) Structured selector rubric (JSON)

- Impact: High
- Effort: Medium
- Feasible because: selector stage already exists.
- Plan: selector emits rubric fields (`validity`, `tool_support`, `contradictions`, `confidence`) used in aggregation.

## 4) Confidence-aware modulus parser

- Impact: Medium-High
- Effort: Low-Medium
- Feasible because: parser and debug summary already centralized.
- Plan: multi-pattern vote with confidence score and fallback alert when ambiguous.

## 5) Agent loop checkpointing/replay

- Impact: Medium-High
- Effort: Medium
- Feasible because: agentic rounds are now explicit in solver.
- Plan: persist per-problem stage trace and allow resume after API/runtime failure.

## 6) Geometry utility pack in sandbox

- Impact: Medium-High
- Effort: Medium
- Feasible because: sandbox execution and geometry stage already wired.
- Plan: add reusable helpers for similarity/power-of-point/radical-axis checks.

## 7) Targeted extraction stress-suite (50+ adversarial cases)

- Impact: Medium
- Effort: Low-Medium
- Feasible because: parsing tests already present.
- Plan: add malformed-final-line, tool-output, and truncation fixtures.

## 8) Stage ablation harness + markdown report

- Impact: Medium
- Effort: Medium
- Feasible because: CLI has stage toggles for all major components.
- Plan: one command to run A/B/C configs and output per-category deltas.

## 9) Submission confidence gate

- Impact: Medium
- Effort: Low
- Feasible because: debug summary already captures support features.
- Plan: block/flag submissions when too many answers come from fallback-only or unsupported tiny values.

## 10) Lightweight local benchmark expansion

- Impact: Medium
- Effort: Medium
- Feasible because: reference pipeline and scripts exist.
- Plan: build additional labeled AIME-style pack for offline tuning of weights/stages.

## Immediate next 3 experiments

1. A/B: `selector+audit` vs `selector+audit+adversarial_probe` on full reference.
2. A/B: `agentic_tool_rounds=1` vs `2` under equal token/time budgets.
3. A/B: dynamic escalation (20B->120B) vs pure 120B on same subset.

## Completed today

- Added bounded agentic tool loop in solver (python sandbox observations fed back to model).
- Hardened parsing for LaTeX modulus and stricter answer extraction behavior.
- Added contradiction-focused stages and stronger aggregation heuristics.
- Hardened Kaggle notebook output validation for required `submission.parquet`.
