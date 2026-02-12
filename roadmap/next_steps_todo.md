# AIMO3 High-Impact Roadmap (Top 10 Feasible Upgrades)

Date: 2026-02-11

Goal: improve solved-count under realistic runtime and cost constraints.

## Implemented Today (Delta)

- Added **parallel attempt + tool harness** controls in solver and CLI.
- Added **10-minute full-budget mode** defaults for `autonomous120b`.
- Added **uncertainty-triggered escalation stage** before arbitration.
- Added **`benchmark-sweep` harness** with:
  - curated trial variants,
  - leaderboard metrics,
  - best-trial export + recommended solve command.
- Added regression coverage for parallel/excalation/CLI sweep behavior.

## 1) Two-pass model cascade (20B -> 120B escalation)

- Why it works: most problems are not equally hard; 120B budget is best spent on uncertain cases.
- Feasibility: high (existing pipeline already supports model/profile switches).
- Success metric: same accuracy as pure-120B with >=35% lower token cost, or +1 solved at same cost.

## 2) Global runtime budget manager

- Why it works: avoids over-spending on easy items and under-spending on hard outliers.
- Feasibility: high (problem complexity + stage controls already exist).
- Success metric: +1 to +2 solved on reference with equal wall-clock.

## 3) Structured selector rubric (JSON output)

- Why it works: deterministic scoring beats free-form selector prose for calibration.
- Feasibility: medium (selector stage already in place).
- Success metric: improved stability across reruns; reduced answer churn.

## 4) Modulus confidence and ambiguity handling

- Why it works: wrong modulus silently corrupts otherwise-correct reasoning.
- Feasibility: high (parser is centralized and already robust).
- Success metric: 0 modulus-related regressions on reference/adversarial parser suite.

## 5) Agent checkpoint/restart and replay

- Why it works: long runs fail from API/network timeouts; replay avoids losing progress.
- Feasibility: medium.
- Success metric: >=95% completion rate on long 50-problem runs.

## 6) Geometry micro-toolkit for sandbox

- Why it works: geometry remains highest-variance category.
- Feasibility: medium (sandbox already supports sympy/numpy).
- Success metric: geometry subset accuracy delta > +10% relative.

## 7) Extraction torture-test suite (100 cases)

- Why it works: truncation and malformed final lines still cause silent defaults.
- Feasibility: high.
- Success metric: parse-failure rate cut by at least 50% on stress corpus.

## 8) Stage-ablation harness with report artifact

- Why it works: fast iteration needs reliable attribution of what helped.
- Feasibility: high (CLI already has stage toggles).
- Success metric: one-command report with per-stage marginal gains.

## 9) Submission confidence gate and risk scoring

- Why it works: prevents wasting submissions on clearly unstable outputs.
- Feasibility: high (debug summary already carries evidence signals).
- Success metric: lower variance and fewer low-confidence submissions.

## 10) Expanded benchmark pack (AIME-like + failure regressions)

- Why it works: 10-item reference is too small for robust tuning.
- Feasibility: medium.
- Success metric: statistically significant config choice over multiple slices.

## Immediate Experiments (Next 72h)

1. `aimo120b` vs `autonomous120b` on full reference with fixed randomization.
2. `agentic_tool_rounds=2` vs `4` with equal token cap.
3. cascade strategy: `20B-first + 120B escalation` vs pure `120B`.
4. selector free-form vs selector structured rubric prototype.

## Implemented Today

- Added bounded autonomous agent loop with iterative tool observation follow-ups.
- Added stateful Python context across agent rounds.
- Added `autonomous120b` maximum-budget profile.
- Updated docs for full execution model and high-budget commands.
