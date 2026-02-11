PYTHON ?= python3
SRC_DIRS ?= src tests scripts
PACKAGE ?= aimo3

.PHONY: install install-dev lint format test check build clean \
	solve-sample solve-groq-cheap benchmark-reference-groq-budget benchmark-reference-groq-120b-max benchmark-reference-groq-autonomous120b \
	kaggle-submit-120b kaggle-submit-autonomous120b kaggle-download kaggle-submit kaggle-pipeline

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check $(SRC_DIRS)

format:
	$(PYTHON) -m ruff format $(SRC_DIRS)

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests

check: lint test

build:
	$(PYTHON) -m build

clean:
	rm -rf build dist .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

solve-sample:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli solve \
		--input-csv examples/sample_problems.csv \
		--output-csv artifacts/submission.csv \
		--debug-json artifacts/debug_traces.json

solve-groq-cheap:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli solve \
		--input-csv examples/sample_problems.csv \
		--output-csv artifacts/submission_groq_cheap.csv \
		--debug-json artifacts/debug_groq_cheap.json \
		--model openai/gpt-oss-20b \
		--attempts 1 \
		--max-tokens 256 \
		--temperatures 0.2 \
		--reasoning-effort low \
		--top-p 0.9

benchmark-reference-groq-budget:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli benchmark-reference \
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

benchmark-reference-groq-120b-max:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli benchmark-reference \
		--reference-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
		--output-dir artifacts/reference_benchmark_120b_max \
		--profile aimo120b \
		--model openai/gpt-oss-120b \
		--reasoning-effort high \
		--attempts 8 \
		--max-tokens 4096 \
		--agentic-tool-rounds 2 \
		--agentic-observation-chars 1400 \
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

benchmark-reference-groq-autonomous120b:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli benchmark-reference \
		--reference-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
		--output-dir artifacts/reference_benchmark_autonomous120b \
		--profile autonomous120b \
		--model openai/gpt-oss-120b \
		--reasoning-effort high \
		--request-timeout 420 \
		--client-max-retries 2

kaggle-submit-120b:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli kaggle-pipeline \
		--competition ai-mathematical-olympiad-progress-prize-3 \
		--input-csv data/raw/test.csv \
		--output-csv artifacts/submission_120b.csv \
		--debug-json artifacts/debug_120b.json \
		--profile aimo120b \
		--model openai/gpt-oss-120b \
		--reasoning-effort high \
		--agentic-tool-rounds 2 \
		--request-timeout 300 \
		--client-max-retries 1 \
		--wait

kaggle-submit-autonomous120b:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli kaggle-pipeline \
		--competition ai-mathematical-olympiad-progress-prize-3 \
		--input-csv data/raw/test.csv \
		--output-csv artifacts/submission_autonomous120b.csv \
		--debug-json artifacts/debug_autonomous120b.json \
		--profile autonomous120b \
		--model openai/gpt-oss-120b \
		--reasoning-effort high \
		--request-timeout 420 \
		--client-max-retries 2 \
		--wait

kaggle-download:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli kaggle-download \
		--competition ai-mathematical-olympiad-progress-prize-3 \
		--output-dir data/raw

kaggle-submit:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli kaggle-submit \
		--competition ai-mathematical-olympiad-progress-prize-3 \
		--submission-csv artifacts/submission.csv \
		--wait

kaggle-pipeline:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli kaggle-pipeline \
		--competition ai-mathematical-olympiad-progress-prize-3 \
		--input-csv /path/to/test.csv \
		--output-csv artifacts/submission.csv \
		--debug-json artifacts/debug_traces.json \
		--attempts 8 \
		--wait
