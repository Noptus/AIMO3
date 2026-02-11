# AIMO3

Production-ready baseline for the Kaggle competition `ai-mathematical-olympiad-progress-prize-3` with:

- `gpt-oss-120b`-first inference (or any OpenAI-compatible model endpoint)
- multi-attempt prompting and weighted answer aggregation
- GenSelect-style selector pass over top candidate answers
- adversarial probe stage to refute fragile consensus candidates
- geometry-specialist recheck stage for geometry candidate arbitration
- tool-integrated reasoning via constrained Python sandbox execution
- robust modulus + answer parsing (including LaTeX forms like `$10^{5}$`, `$5^7$`, `$99991$`)
- Kaggle API automation for download, submit, and score polling

## Project standards

- Source layout: `src/` package + explicit CLI entrypoint
- Quality checks: `ruff` linting + unit tests
- CI: GitHub Actions on push and pull request (`.github/workflows/ci.yml`)
- Contribution and security process: `CONTRIBUTING.md` and `SECURITY.md`

## Repository layout

- `src/aimo3/prompts.py`: typed prompt routing and templates
- `src/aimo3/parsing.py`: modulus extraction, answer extraction, weighted mode
- `src/aimo3/sandbox.py`: code safety validation + constrained execution
- `src/aimo3/solver.py`: orchestration across attempts and code verification
- `src/aimo3/pipeline.py`: dataframe batch solve + artifact writers
- `src/aimo3/kaggle_api.py`: Kaggle API wrapper with submission polling
- `src/aimo3/cli.py`: end-to-end CLI (`solve`, `kaggle-download`, `kaggle-submit`, `kaggle-pipeline`)
- `notebooks/aimo3_first_attempt.ipynb`: polished first notebook for iterative development

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
cp .env.example .env
```

Set environment values in `.env`:

```bash
AIMO_BASE_URL=http://127.0.0.1:8000/v1
AIMO_MODEL=openai/gpt-oss-120b
AIMO_API_KEY=
GROQ_API_KEY=
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```

If `GROQ_API_KEY` is set and `AIMO_BASE_URL` is not set, the CLI automatically targets `https://api.groq.com/openai/v1`.
For Groq + `openai/gpt-oss-*`, the CLI also auto-enables hosted `code_interpreter` to avoid tool-call failures on hard math prompts.

## Baseline usage

### 1) Solve problems -> create submission CSV

```bash
aimo3 solve \
  --input-csv examples/sample_problems.csv \
  --output-csv artifacts/submission.csv \
  --debug-json artifacts/debug_traces.json \
  --attempts 8
```

Low-cost smoke test on hosted API:

```bash
aimo3 solve \
  --input-csv examples/sample_problems.csv \
  --output-csv artifacts/submission.csv \
  --debug-json artifacts/debug_traces.json \
  --attempts 1 \
  --max-tokens 256 \
  --temperatures 0.2 \
  --reasoning-effort low
```

### 2) Submit CSV to Kaggle and wait for score

```bash
aimo3 kaggle-submit \
  --competition ai-mathematical-olympiad-progress-prize-3 \
  --submission-csv artifacts/submission.csv \
  --message "first composable baseline" \
  --wait
```

### 3) Fully automated pipeline (solve + submit)

```bash
aimo3 kaggle-pipeline \
  --competition ai-mathematical-olympiad-progress-prize-3 \
  --input-csv /path/to/test.csv \
  --output-csv artifacts/submission.csv \
  --debug-json artifacts/debug_traces.json \
  --attempts 10 \
  --wait
```

Equivalent shortcuts are available through `Makefile`:

```bash
make test
make solve-sample
make solve-groq-cheap
make benchmark-reference-groq-budget
make benchmark-reference-groq-120b-max
make kaggle-submit-120b
make kaggle-download
make kaggle-submit
```

## Development workflow

```bash
make install-dev
make lint
make test
make check
```

Optional local hook installation:

```bash
pre-commit install
```

## Kaggle API automation

Download competition files:

```bash
aimo3 kaggle-download \
  --competition ai-mathematical-olympiad-progress-prize-3 \
  --output-dir data/raw
```

Automation behavior:

- authenticates using standard `KAGGLE_USERNAME` + `KAGGLE_KEY`
- also accepts `KAGGLE_API_TOKEN=username:key`
- submits with custom message
- polls latest submission status until complete

## Notebook

Open `notebooks/aimo3_first_attempt.ipynb` for an interactive workflow:

1. configure model endpoint
2. run smoke test
3. run batch inference
4. auto-submit to Kaggle API

## Testing

```bash
python3 -m unittest discover -s tests
```

## Notes on model strategy

This baseline is intentionally composable and tuned for reliable first submissions:

- prompt routing by problem type
- multiple temperatures + diversified proof-first/code-first attempt modes
- ratio-based early stopping by consensus confidence
- contradiction-focused consistency audit + adversarial probe passes
- geometry-aware prompt checklist (invariants + analytic fallback + independent cross-check)
- code block extraction and sandbox verification for higher-confidence answers
- anti-collapse safeguards (small-answer guard + fallback guess only when required)
- answer aggregation that up-ranks code-verified/verifier/audit/probe/geometry-recheck/selector-backed candidates and down-ranks weak priors (tiny/problem-echo answers)

Recommended next iteration steps:

1. add model ensemble (secondary 32B model)
2. increase attempt budget on hard-problem detector only
3. calibrate answer weighting from offline validation (AIME/HMMT-like set)
4. add stronger modulus disambiguation pass for edge-format statements

## Cheapest way to run now

For very low spend while validating pipeline plumbing:

1. use `openai/gpt-oss-20b` for broad sweeps
2. set `--attempts 1` and `--max-tokens 256`
3. keep `--reasoning-effort low`
4. rerun only uncertain/hard problems with `openai/gpt-oss-120b`

For high-budget Groq runs on complex problems, use larger completion budgets (`--max-tokens 4096`) to avoid truncation before `FINAL_ANSWER`.

Reference benchmark command (AIMO-focused, high budget):

```bash
PYTHONPATH=src python -m aimo3.cli benchmark-reference \
  --reference-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
  --output-dir artifacts/reference_benchmark_20b_budget \
  --profile balanced \
  --model openai/gpt-oss-20b \
  --reasoning-effort medium \
  --attempts 2 \
  --max-tokens 4096 \
  --repair-passes 1 \
  --final-extractor-passes 2 \
  --verification-attempts 1 \
  --verification-top-k 3 \
  --max-code-blocks-per-attempt 3
```

Maximum-performance 120B reference benchmark:

```bash
PYTHONPATH=src python -m aimo3.cli benchmark-reference \
  --reference-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
  --output-dir artifacts/reference_benchmark_120b_max \
  --profile aimo120b \
  --model openai/gpt-oss-120b \
  --reasoning-effort high \
  --attempts 8 \
  --max-tokens 4096 \
  --repair-passes 1 \
  --final-extractor-passes 2 \
  --verification-attempts 3 \
  --verification-top-k 4 \
  --consistency-audit-attempts 2 \
  --consistency-audit-top-k 4 \
  --consistency-audit-temperature 0.09 \
  --adversarial-probe-attempts 2 \
  --adversarial-probe-top-k 4 \
  --adversarial-probe-temperature 0.14 \
  --geometry-recheck-attempts 2 \
  --geometry-top-k 4 \
  --geometry-recheck-temperature 0.07 \
  --small-answer-guard-attempts 2 \
  --small-answer-guard-top-k 3 \
  --small-answer-guard-temperature 0.10 \
  --selector-attempts 2 \
  --selector-top-k 4 \
  --selector-temperature 0.05 \
  --sparse-recovery-attempts 4 \
  --sparse-recovery-temperature 0.1 \
  --max-code-blocks-per-attempt 4 \
  --request-timeout 300 \
  --client-max-retries 2
```

Recommended pre-submission profile:

- `--profile aimo120b`
- 120B-first settings with diversified attempts, repair/extractor passes, geometry recheck + verifier + selector arbitration, and sparse-recovery attempts for timeout-heavy problems.
- Pair with `--request-timeout 300 --client-max-retries 1` for long olympiad prompts.

## License

MIT. See `LICENSE`.
