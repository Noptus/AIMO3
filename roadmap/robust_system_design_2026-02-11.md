# AIMO3 Robust System Design - 2026-02-11

## Competition strategy synthesis

- Top public AIMO3 runs converge on one pattern: tool-integrated reasoning (TIR), multi-sample self-consistency, then a selector/arbitration pass.
- Strong teams do not trust single-trace outputs; they use contradiction checks and stage-specific voting (verifier/selector/geometry specialists).
- Major failure modes are extraction drift, modulus mistakes, and trivial-answer collapse (`0/1`) when evidence is weak.

## What reference problems suggest

From `reference/ai-mathematical-olympiad-progress-prize-3/reference.csv`:

- Category spread is mixed, so single-style prompting is brittle.
  - `algebra`: 4
  - `geometry`: 2
  - `number_theory`: 2
  - `combinatorics`: 1
  - `general`: 1
- Modulus forms are heterogeneous:
  - `10^{5}` (LaTeX math mode),
  - `99991`,
  - `5^7`,
  - and some statements with implicit/no explicit modulus.
- Robust modulus parsing is critical because final normalization affects leaderboard score directly.

## Prompt strategy upgrades

- Keep archetype-specific guidance (valuation/combinatorics/geometry/functional equations).
- Enforce a reliability protocol in base prompts:
  - derive primary route,
  - run contradiction checks,
  - verify modulus and constraints,
  - resolve route disagreement before final line.
- Add an adversarial probe prompt stage:
  - explicitly tries to falsify the current leading candidate,
  - uses an independent derivation path,
  - emits corrected answer if refutation succeeds.
- Keep small-answer guard and geometry recheck as dedicated specialist prompts.

## Robust solver architecture

Current stage order:

1. Initial diversified attempts (proof-first/code-first mix).
2. Verification arbitration over top candidates.
3. Sparse recovery when extraction is too thin.
4. Consistency audit over top candidates.
5. Adversarial probe (new) to challenge fragile consensus.
6. Geometry recheck (geometry-only).
7. Small-answer guard (anti-`0/1` collapse).
8. Selector (GenSelect-style final arbitration).
9. Fallback guess only when all normal paths fail.

Aggregation upgrades:

- Stage-diversity bonus (answers supported across independent stages win over single-stage clusters).
- Extra weight for `consistency_audit`, `adversarial_probe`, `geometry_recheck`, `selector`.
- Continued penalties for weak tiny answers and problem-echo numbers.

## Recommended high-effort profile (Groq 120B)

Use `--profile aimo120b` (now includes adversarial probing and stronger arbitration defaults):

```bash
PYTHONPATH=src python -m aimo3.cli benchmark-reference \
  --reference-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
  --output-dir artifacts/reference_benchmark_120b_robust \
  --profile aimo120b \
  --model openai/gpt-oss-120b \
  --reasoning-effort high \
  --request-timeout 300 \
  --client-max-retries 2
```

## Next experiments

1. Run ablation: `adversarial_probe` on/off with fixed seed and equal budgets.
2. Compare selector-only vs selector+adversarial on geometry subset.
3. Add per-problem compute allocator (bank easy-problem time for hard outliers).
