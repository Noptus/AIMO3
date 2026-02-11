# AIMO3 Forward Roadmap (TODO)

Date: 2026-02-11

## Completed today (2026-02-11)

- [x] Added contradiction-focused strategy stages (`consistency_audit`, `adversarial_probe`) and wired them into aggregation/ranking.
- [x] Strengthened anti-collapse safeguards (`small_answer_guard`, fallback behavior tuning).
- [x] Improved modulus parsing coverage for LaTeX-style expressions (`$10^{5}$`, `$5^7$`, `$99991$`).
- [x] Tightened extractor prompt contract and answer-line parsing to reduce false positives.
- [x] Added regression tests for new parsing and solver-stage behavior.
- [x] Synced Kaggle kernel mirror code with latest solver/prompt/parsing pipeline.

## P0 - Highest expected score impact

- [ ] Add a two-model ensemble pipeline (`openai/gpt-oss-120b` + fast 20B/32B) with disagreement-triggered escalation.
  - Why: boosts robustness on edge cases where one model mode collapses.
  - Deliverable: weighted answer fusion with per-model confidence calibration.

- [ ] Implement per-problem dynamic compute allocator with a global runtime budget.
  - Why: shift tokens/attempts from easy problems to hard geometry/number theory outliers.
  - Deliverable: budget manager with `easy/medium/hard` runtime caps and banked time.

- [ ] Add candidate-level verifier scoring using structured rubric outputs.
  - Why: current selector uses free-form rationale; structured rubric is easier to tune.
  - Deliverable: verifier emits JSON fields (`validity`, `consistency`, `tool_support`, `penalties`).

- [ ] Add symbolic-geometry helper tools in sandbox (SymPy geometry checks + invariant templates).
  - Why: geometry remains a high-variance category.
  - Deliverable: reusable geometry utility module used by code-first attempts.

## P1 - Reliability + extraction robustness

- [ ] Expand extraction hardening to full adversarial suite.
  - Status: base hardening shipped (stricter answer-line hints, extractor prompt tightening, fallback guard improvements).
  - Remaining deliverable: parser tests for 50+ adversarial output patterns and malformed tool-call outputs.

- [ ] Add modulus confidence scoring and multi-pass consistency vote.
  - Status: modulus parsing coverage improved for common AIMO3 formats.
  - Remaining deliverable: confidence score + warning flag in debug artifacts.

- [ ] Add API fault-tolerance strategy for long runs.
  - Deliverable: jittered retries, per-problem checkpointing, restart-from-checkpoint CLI.

- [ ] Add run manifest and deterministic seed logging.
  - Deliverable: `artifacts/<run_id>/manifest.json` with model, prompts, params, git SHA.

## P1 - Data + offline evaluation

- [ ] Build a local benchmark pack from reference + curated public AIME-style tasks.
  - Deliverable: `data/benchmarks/*.csv` + script to compute per-category score deltas.

- [ ] Add ablation runner for prompt/selector/verification toggles.
  - Deliverable: one command producing a markdown report with win/loss table.

- [ ] Add error taxonomy tagging (parse failure, wrong modulus, reasoning drift, tool failure).
  - Deliverable: post-run analysis notebook and summary CSV.

## P2 - Engineering quality and maintainability

- [ ] Split solver into stage modules (`attempt`, `repair`, `verify`, `geometry_recheck`, `selector`, `aggregate`).
  - Deliverable: lower-complexity code and easier experimentation.

- [ ] Add typed config schema and profile registry.
  - Deliverable: `profiles.yaml` + loader + CLI validation.

- [ ] Add integration tests with mocked chat backend and flaky network simulation.
  - Deliverable: CI job that validates retry/recovery behavior.

- [ ] Add precomputed prompt snapshots for regression checks.
  - Deliverable: tests asserting critical prompt invariants by archetype.

## P2 - Kaggle operations

- [ ] Build dedicated competition notebook template that reads Kaggle secrets natively.
  - Deliverable: `notebooks/kaggle_submission_template.ipynb` + auto-pack script.

- [ ] Add one-command kernel pack/push/status script.
  - Deliverable: `scripts/kaggle_kernel_push.sh` with version tagging and log retrieval.

- [ ] Add submission policy guardrails (daily quota, max pending submissions, confidence threshold).
  - Deliverable: submit gate in CLI before pushing low-confidence outputs.

## Immediate experiment backlog (next 3 runs)

- [ ] Run A/B: selector+audit vs selector+audit+adversarial_probe on full reference.
- [ ] Run A/B: geometry recheck `attempts=2` vs `attempts=4` with fixed budgets.
- [ ] Run A/B: aggressive high-temp tail (`0.7, 0.85`) only on unresolved problems after stage-1 consensus.
