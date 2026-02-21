# Robust Harness TODO (2026-02-13)

## Implemented in this pass

- [x] Mandatory code-first attempts:
  - Added `mandatory_code_attempts` and forced code-check follow-up prompt when no code evidence appears.
  - Files: `src/aimo3/solver.py`, `src/aimo3/prompts.py`, `src/aimo3/cli.py`

- [x] Verifier/selector/escalation-first robustness:
  - Kept multi-stage arbitration pipeline active in strong profiles.
  - Strengthened stage evidence use in scoring and tie-break.
  - Files: `src/aimo3/solver.py`, `src/aimo3/cli.py`

- [x] Verified-evidence weighting:
  - Increased weight for independent checks/code verification.
  - Downweighted text-only final-answer extraction with no evidence.
  - Files: `src/aimo3/solver.py`

- [x] Contradiction-aware selection:
  - Existing consistency/adversarial stages retained and weighted more in final aggregation.
  - Files: `src/aimo3/solver.py`, `src/aimo3/prompts.py`

- [x] Archetype mini-solvers:
  - Added deterministic mini-solvers for trivial/high-confidence arithmetic/remainder sub-cases.
  - Files: `src/aimo3/mini_solvers.py`, `src/aimo3/solver.py`

- [x] Structured output contract:
  - Added `RESULT_JSON` requirement and parser (`answer`, `method`, `independent_check_passed`).
  - Files: `src/aimo3/prompts.py`, `src/aimo3/parsing.py`, `src/aimo3/solver.py`

- [x] Better tie-break policy:
  - Prefer independent checks, stage/source diversity, and verified support.
  - Files: `src/aimo3/solver.py`

- [x] Strict 0/1 collapse controls:
  - Strong penalties and post-selection override when 0/1 lacks evidence.
  - Files: `src/aimo3/solver.py`

- [x] Submission quality gate:
  - Added script to run reference + hard synthetic checks with fail-fast thresholds.
  - Files: `scripts/run_submission_quality_gate.py`, `Makefile`

- [x] Parallel + caching maintained:
  - Kept parallel attempts/code and sandbox cache integration.
  - Files: `src/aimo3/solver.py`

## Next improvements (post-run)

- [ ] Add domain-specific deterministic solvers for harder number-theory/combinatorics archetypes.
- [ ] Add candidate-level calibration model trained on local debug traces.
- [ ] Add automatic per-problem retry policy based on entropy/disagreement signatures.
- [ ] Add stronger anti-echo detector with lexical overlap + derivation evidence scoring.
