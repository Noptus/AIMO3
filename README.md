# AIMO3 - First High-Quality Baseline

Composable baseline for the Kaggle competition `ai-mathematical-olympiad-progress-prize-3` with:

- `gpt-oss-120b`-first inference (or any OpenAI-compatible model endpoint)
- multi-attempt prompting and weighted answer aggregation
- tool-integrated reasoning via constrained Python sandbox execution
- robust modulus + answer parsing
- Kaggle API automation for download, submit, and score polling

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
make kaggle-download
make kaggle-submit
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
- multiple temperatures + early stopping by consensus
- code block extraction and sandbox verification for higher-confidence answers
- weighted vote that up-ranks code-verified candidates

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
