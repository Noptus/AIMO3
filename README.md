# AIMO3

Composable Kaggle solver framework for `ai-mathematical-olympiad-progress-prize-3` with:

- multi-attempt LLM solving (`gpt-oss-120b` first, any OpenAI-compatible endpoint)
- agentic Python-sandbox execution loop (model -> code -> sandbox output -> model follow-up)
- stage-based arbitration (`verification`, `consistency_audit`, `adversarial_probe`, `geometry_recheck`, `selector`)
- anti-collapse guards for trivial outputs (`0/1` penalties + dedicated guard stages)
- Kaggle automation for competition files, notebook packaging, and submission plumbing

## Is The LLM Agentic Right Now?

Yes, partially and explicitly.

Current solver behavior (`src/aimo3/solver.py`):

1. model generates a candidate solution
2. fenced Python code blocks are extracted and executed in the local sandbox
3. solver builds tool observations from stdout/errors
4. model receives a follow-up prompt with those observations (`agentic_tool_rounds`)
5. final answer is aggregated with additional verifier/recheck/selector stages

This is a practical agent loop with bounded tool rounds. It is not an unrestricted autonomous tool planner; it is a controlled math-solver agent with sandboxed Python execution.

## Core Architecture

- `src/aimo3/prompts.py`
  - problem classification + archetype routing
  - base solve prompt + specialist prompts (repair, verification, selector, adversarial probe, extractor)
- `src/aimo3/sandbox.py`
  - constrained Python execution for tool-integrated reasoning
- `src/aimo3/solver.py`
  - multi-attempt orchestration
  - agentic tool rounds
  - stage pipeline and weighted aggregation
- `src/aimo3/parsing.py`
  - modulus extraction (including LaTeX forms like `$10^{5}$`, `$5^7$`, `$99991$`)
  - robust final answer extraction
- `src/aimo3/cli.py`
  - end-to-end commands for solve, benchmark, and Kaggle operations
- `kaggle_kernel_submission/`
  - notebook package used for competition kernel execution (internet disabled mode)

## Solver Stage Pipeline

Per problem, the current default flow is:

1. `initial` diversified attempts (proof-first/code-first mix)
2. bounded agentic tool follow-up rounds (`--agentic-tool-rounds`)
3. optional repair pass
4. optional strict extractor pass
5. optional verification arbitration
6. sparse-recovery attempts when evidence is too thin
7. consistency audit stage
8. adversarial probe stage
9. geometry recheck stage (geometry only)
10. small-answer guard stage
11. selector stage
12. fallback guess (last resort)

Final aggregation rewards:

- code-verified traces
- cross-stage agreement/diversity
- verifier/audit/probe/selector support

And penalizes:

- weak tiny answers
- problem-echo numbers
- unsupported trivial consensus

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
cp .env.example .env
```

Minimal `.env`:

```bash
AIMO_MODEL=openai/gpt-oss-120b
AIMO_BASE_URL=https://api.groq.com/openai/v1
AIMO_API_KEY=
GROQ_API_KEY=
KAGGLE_USERNAME=
KAGGLE_KEY=
# optional: KAGGLE_API_TOKEN=username:key
```

## Main Commands

### Solve CSV

```bash
aimo3 solve \
  --input-csv examples/sample_problems.csv \
  --output-csv artifacts/submission.csv \
  --debug-json artifacts/debug_traces.json \
  --profile balanced
```

### Benchmark on labeled reference

```bash
PYTHONPATH=src python -m aimo3.cli benchmark-reference \
  --reference-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
  --output-dir artifacts/reference_benchmark \
  --profile aimo120b \
  --model openai/gpt-oss-120b \
  --reasoning-effort high
```

### Kaggle API (file submission path)

```bash
aimo3 kaggle-download --competition ai-mathematical-olympiad-progress-prize-3 --output-dir data/raw
aimo3 kaggle-submit --competition ai-mathematical-olympiad-progress-prize-3 --submission-csv artifacts/submission.csv --wait
```

### Kaggle notebook push (code competition workflow)

```bash
kaggle kernels push -p kaggle_kernel_submission
```

Then choose the pushed notebook version in Kaggle competition UI.

## Profiles And Cost Control

- `cheap`
  - minimal attempts/tokens, reduced stages
  - use for wiring checks
- `balanced`
  - practical default
- `hard`
  - stronger stage budget and verification
- `aimo120b`
  - maximum practical 120B configuration in this repo

Recommended low-cost iteration:

1. run `openai/gpt-oss-20b` on full set
2. rerun uncertain/hard slice with `openai/gpt-oss-120b`
3. submit best merged result

## Agentic Controls

New/important knobs:

- `--agentic-tool-rounds` (default `1`)
- `--agentic-observation-chars` (default `1200`)
- `--max-code-blocks-per-attempt`
- `--repair-passes`
- `--final-extractor-passes`
- `--consistency-audit-attempts`
- `--adversarial-probe-attempts`
- `--selector-attempts`

Example (high effort):

```bash
PYTHONPATH=src python -m aimo3.cli benchmark-reference \
  --reference-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
  --output-dir artifacts/reference_benchmark_120b_robust \
  --profile aimo120b \
  --model openai/gpt-oss-120b \
  --reasoning-effort high \
  --agentic-tool-rounds 2 \
  --max-code-blocks-per-attempt 4 \
  --consistency-audit-attempts 2 \
  --adversarial-probe-attempts 2 \
  --geometry-recheck-attempts 2 \
  --selector-attempts 2 \
  --request-timeout 300
```

## Kaggle Notebook Constraints

Competition notebook constraints matter:

- internet disabled during scoring
- required output filename: `submission.parquet`
- output must exist in `/kaggle/working`

Notebook in `kaggle_kernel_submission/aimo3_submission.ipynb` is hardened to:

- always write `/kaggle/working/submission.parquet`
- validate parquet existence, schema, and row count
- emit clear logs for output discovery

## Interpreting Common Kaggle Logs

Warnings like these are usually non-blocking:

- debugger frozen module warning
- `mistune` / `nbconvert` syntax/future warnings

Blocking issue is typically only:

- missing or invalid `submission.parquet`

## Development

```bash
make install-dev
make lint
make test
make check
```

CI and standards:

- `.github/workflows/ci.yml`
- `CONTRIBUTING.md`
- `SECURITY.md`

## Current Limitations

- Hosted API usage is unavailable in official offline Kaggle scoring runs.
- True top-tier competition performance requires robust offline in-notebook inference stack (local model weights/runtime), not only API-mode orchestration.
- Agent loop is intentionally bounded for reliability and runtime control.

## License

MIT (`LICENSE`).
