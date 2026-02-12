# Research Strategy Update - 2026-02-11

## Primary Sources Reviewed

1. **Self-Consistency Improves Chain of Thought Reasoning in Language Models**  
   Wang et al., 2022  
   <https://arxiv.org/abs/2203.11171>

2. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**  
   Yao et al., 2023  
   <https://arxiv.org/abs/2305.10601>

3. **Self-Refine: Iterative Refinement with Self-Feedback**  
   Madaan et al., 2023  
   <https://arxiv.org/abs/2303.17651>

4. **Self-Evaluation Guided Beam Search for Reasoning**  
   Xie et al., 2023  
   <https://arxiv.org/abs/2305.00633>

## What Was Integrated Into AIMO3

- **Stronger self-consistency harness**:
  - multi-trial `benchmark-sweep` with diversified search settings and ranking metrics.
  - best-trial export + recommended solve command.

- **Uncertainty-aware escalation**:
  - new escalation stage launches extra attempts only when early candidates are weak or conflicting.
  - targeted to recover from fragile consensus and 0/1 collapse.

- **Long-runtime orchestration**:
  - explicit high-budget controls remain (`per_problem_time_sec`, force-full-time).
  - attempt-level parallel workers plus stateless code execution parallelism.

## Why This Should Help

- Self-consistency and beam-like candidate expansion improve reliability on hard reasoning tasks.
- Iterative refinement and verifier-style arbitration reduce brittle single-trace failures.
- A sweep harness converts runtime budget into measurable configuration gains instead of ad hoc tuning.

## Next High-Value Additions

1. Add stage-level confidence calibration trained from reference traces.
2. Add global run-level time banking across problems (not just per-problem deadline).
3. Add structured selector output schema for deterministic scoring.
4. Add richer geometry symbolic check library for sandbox verification.
