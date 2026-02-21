# GPT-OSS-120B AIMO Execution Playbook

Date: 2026-02-13

Goal: maximize solve quality with reliable, fast tool-integrated reasoning.

## Model Behavior (Observed)

- Strong at structured modular arithmetic when prompts force explicit checks.
- Prone to:
  - early confident numeric guesses without enough verification,
  - statement-number echo (copying constants from prompt),
  - trivial-collapse (`0/1`) under uncertainty,
  - long unproductive prose without compact code checks.

## Prompt Strategy

1. Decompose first:
- Require a short plan with target quantity + constraints before derivation.

2. Force independent check:
- At least one of:
  - python arithmetic verification,
  - modular contradiction,
  - parity/bound contradiction.

3. Anti-shortcut guard:
- Explicitly forbid statement-constant echo unless derived.
- Require forcing justification for `0/1`.

4. Clean code policy:
- Use compact deterministic code blocks.
- No brute-force unless bounded and justified.
- Print only decisive integer outputs.

## Runtime Strategy

1. Keep provider-stable concurrency:
- `parallel_attempt_workers=2`, `parallel_code_workers=2`.

2. Preserve arbitration time:
- Enable stage reserve and avoid initial parallel bursts under strict deadlines.

3. Use sharded long runs:
- Split reference runs into small shards (1-3 rows), merge outputs.

4. Favor evidence-rich answers:
- Prioritize candidates with stage diversity and code agreement.

## Recommended Command Baseline

```bash
PYTHONPATH=src python -m aimo3.cli benchmark-reference \
  --reference-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
  --output-dir artifacts/reference_benchmark_gptoss120b \
  --profile aimo120b \
  --per-problem-time-sec 90 \
  --agentic-tool-rounds 4 \
  --parallel-attempt-workers 2 \
  --parallel-code-workers 2 \
  --reasoning-effort high \
  --request-timeout 90 \
  --client-max-retries 2
```

