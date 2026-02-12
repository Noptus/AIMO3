# AIMO3

AIMO3 is a production-oriented, agentic math-solving system for Kaggle competition `ai-mathematical-olympiad-progress-prize-3`.

## What This Repo Does

- Solves each problem with multiple diversified `gpt-oss` attempts.
- Runs an autonomous Python sandbox loop inside each attempt.
- Uses cross-stage arbitration (verifier, consistency audit, adversarial probe, geometry recheck, selector).
- Applies anti-collapse safeguards for weak `0/1` behavior.
- Produces Kaggle-ready outputs and automates Kaggle workflows.

## Is It Fully Agentic Right Now?

Yes, within controlled limits.

Current solver (`/Users/raphaelcaillon/Documents/GitHub/AIMO3/src/aimo3/solver.py`) runs a bounded autonomous loop:

1. Model drafts solution and emits python blocks.
2. Sandbox executes code.
3. Tool observations (stdout/errors) are fed back to the model.
4. Model continues reasoning and may emit more code.
5. Loop repeats for configurable rounds.

Agentic capabilities now include:

- iterative tool-use rounds (`--agentic-tool-rounds`)
- stateful python context across rounds (`--agentic-stateful-python`)
- persistent code state budget (`--agentic-state-chars`)
- bounded observation context (`--agentic-observation-chars`)
- parallel attempt workers so LLM calls and sandbox loops overlap (`--parallel-attempt-workers`)
- parallel code execution in stateless mode (`--parallel-code-workers`)
- uncertainty-triggered escalation stage for weak consensus (`--escalation-*`)
- stage-time reservation to guarantee arbitration runs (`--stage-time-reserve-sec`)
- forced tool follow-up for unverified medium/hard answers (`--force-tool-round-for-unverified`)

This is a bounded autonomous agent machine, not an unbounded open-ended planner.

## Core Components

- `/Users/raphaelcaillon/Documents/GitHub/AIMO3/src/aimo3/solver.py`: orchestration, agent loop, stage pipeline, aggregation.
- `/Users/raphaelcaillon/Documents/GitHub/AIMO3/src/aimo3/sandbox.py`: constrained Python execution.
- `/Users/raphaelcaillon/Documents/GitHub/AIMO3/src/aimo3/prompts.py`: archetype routing + stage prompt builders.
- `/Users/raphaelcaillon/Documents/GitHub/AIMO3/src/aimo3/parsing.py`: modulus/answer extraction.
- `/Users/raphaelcaillon/Documents/GitHub/AIMO3/src/aimo3/cli.py`: solve, benchmark, Kaggle automation.
- `/Users/raphaelcaillon/Documents/GitHub/AIMO3/kaggle_kernel_submission/`: offline competition notebook package.

## Solver Pipeline

Per problem:

1. Initial diversified attempts.
2. Agentic sandbox rounds per attempt.
3. Repair/extractor recovery (if needed).
4. Verification.
5. Sparse recovery.
6. Consistency audit.
7. Adversarial probe.
8. Geometry recheck.
9. Small-answer guard.
10. Selector.
11. Fallback guess.
12. Weighted final aggregation.

Aggregation boosts:

- code-verified support
- multi-stage consensus/diversity
- verifier/audit/probe/selector confirmations

Aggregation penalties:

- unsupported tiny outputs
- problem-echo numbers
- weak fallback-only evidence

## Profiles

- `cheap`: minimal cost smoke checks.
- `balanced`: practical default.
- `hard`: strong general high-effort profile.
- `aimo120b`: tuned 120B high-effort profile.
- `autonomous120b`: maximum autonomous profile (long runs, deep stage budget, stronger agent loop).
  - defaults to `600s` budget per problem and disables early stop.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
cp .env.example .env
```

Recommended `.env`:

```bash
AIMO_MODEL=openai/gpt-oss-120b
AIMO_BASE_URL=https://api.groq.com/openai/v1
AIMO_API_KEY=
GROQ_API_KEY=
KAGGLE_USERNAME=
KAGGLE_KEY=
# optional: KAGGLE_API_TOKEN=username:key
```

## Main Usage

### Solve

```bash
aimo3 solve \
  --input-csv examples/sample_problems.csv \
  --output-csv artifacts/submission.csv \
  --debug-json artifacts/debug_traces.json \
  --profile balanced
```

### High-Budget Autonomous Reference Run (Groq/Online)

```bash
PYTHONPATH=src python -m aimo3.cli benchmark-reference \
  --reference-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
  --output-dir artifacts/reference_benchmark_autonomous120b \
  --profile autonomous120b \
  --model openai/gpt-oss-120b \
  --per-problem-time-sec 600 \
  --force-full-problem-time \
  --agentic-tool-rounds 5 \
  --max-code-blocks-per-attempt 6 \
  --parallel-attempt-workers 4 \
  --parallel-code-workers 4 \
  --reasoning-effort high \
  --request-timeout 600 \
  --client-max-retries 2
```

### Full Competition Pipeline

```bash
PYTHONPATH=src python -m aimo3.cli kaggle-pipeline \
  --competition ai-mathematical-olympiad-progress-prize-3 \
  --input-csv data/raw/test.csv \
  --output-csv artifacts/submission_autonomous120b.csv \
  --debug-json artifacts/debug_autonomous120b.json \
  --profile autonomous120b \
  --model openai/gpt-oss-120b \
  --per-problem-time-sec 600 \
  --force-full-problem-time \
  --agentic-tool-rounds 5 \
  --max-code-blocks-per-attempt 6 \
  --parallel-attempt-workers 4 \
  --parallel-code-workers 4 \
  --reasoning-effort high \
  --request-timeout 600 \
  --client-max-retries 2 \
  --wait
```

### Multi-Hour Reference Harness Sweep (Recommended Before Final Submission)

```bash
PYTHONPATH=src python -m aimo3.cli benchmark-sweep \
  --reference-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
  --output-dir artifacts/reference_sweep_autonomous120b \
  --profile autonomous120b \
  --model openai/gpt-oss-120b \
  --trial-set standard \
  --per-problem-time-sec 600 \
  --force-full-problem-time \
  --agentic-tool-rounds 5 \
  --max-code-blocks-per-attempt 6 \
  --parallel-attempt-workers 4 \
  --parallel-code-workers 4 \
  --reasoning-effort high \
  --request-timeout 600 \
  --client-max-retries 2
```

This writes:
- `artifacts/reference_sweep_autonomous120b/sweep_leaderboard.csv`
- `artifacts/reference_sweep_autonomous120b/sweep_leaderboard.json`
- `artifacts/reference_sweep_autonomous120b/best_trial.json`
- per-trial predictions/debug under `artifacts/reference_sweep_autonomous120b/trials/`

## Critical Agent Controls

- `--agentic-tool-rounds`
- `--agentic-observation-chars`
- `--agentic-stateful-python` / `--no-agentic-stateful-python`
- `--agentic-state-chars`
- `--max-code-blocks-per-attempt`
- `--parallel-attempt-workers`
- `--parallel-code-workers`
- `--escalation-attempts`
- `--escalation-temperature`
- `--escalation-trigger-ratio`
- `--escalation-min-valid`
- `--stage-time-reserve-sec`
- `--force-tool-round-for-unverified` / `--no-force-tool-round-for-unverified`
- `--per-problem-time-sec`
- `--min-time-for-attempt-sec`
- `--min-time-for-stage-sec`
- `--force-full-problem-time` / `--no-force-full-problem-time`

## Reliability Fixes Applied

Recent failure analysis against reference/debug traces identified:
- transient provider errors (notably HTTP 524) causing weak single-vote outcomes,
- attempt loops consuming full budget and starving verifier/selector stages,
- unverified `FINAL_ANSWER` outputs skipping tool confirmation.

Fixes now in code:
- transient retry coverage expanded (`408/409/429/5xx/52x`),
- stage-time reserve gate in initial-attempt scheduler,
- forced tool round on unverified medium/hard candidates,
- uncertainty-triggered escalation attempts before arbitration.

## Kaggle Notebook Notes

The competition notebook in `/Users/raphaelcaillon/Documents/GitHub/AIMO3/kaggle_kernel_submission/aimo3_submission.ipynb` is hardened to:

- always write `/kaggle/working/submission.parquet`
- validate output file existence, schema, and row count
- run with internet disabled mode for scoring compatibility
- use stronger offline fallback logic (exact reference hit, retrieval, pattern solver, diversified hash fallback)

If Kaggle says parquet is missing, ensure you select the latest notebook version that logs `Saved required output: /kaggle/working/submission.parquet`.

## Common Log Warnings (Non-blocking)

These usually do not indicate failure:

- frozen modules debugger warning
- `mistune` / `nbconvert` syntax/future warnings

Blocking issue is typically only missing/invalid `submission.parquet`.

### Sample Validation Logs

Kaggle often validates notebook output using a 3-row sample test with ids `000aaa`, `111bbb`, `222ccc`.
Those sample problems have answer `0`, so seeing all-zero outputs there is expected and not by itself an error.

## Development

```bash
make install-dev
make lint
make test
make check
```

## Limitation To Keep In Mind

Online API mode (Groq/OpenAI) is excellent for development and external runs, but Kaggle competition scoring notebooks run offline. For leaderboard-maximal offline performance, a local in-notebook model runtime is still required.

## License

MIT (`/Users/raphaelcaillon/Documents/GitHub/AIMO3/LICENSE`).
