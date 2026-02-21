PYTHON ?= python3
SRC_DIRS ?= src tests scripts
PACKAGE ?= aimo3

.PHONY: install install-dev install-agentic lint format test check build clean \
	solve-sample solve-groq-cheap benchmark-reference-groq-budget benchmark-reference-groq-120b-max benchmark-reference-groq-autonomous120b benchmark-reference-groq-autonomous120b-robust benchmark-sweep-groq-autonomous120b \
	solve-sample-langgraph kaggle-submit-120b kaggle-submit-autonomous120b kaggle-download kaggle-submit kaggle-submit-code kaggle-kernel-preflight kaggle-kernel-submit kaggle-pipeline \
	notebook-proxy-eval kaggle-kernel-preflight-next kaggle-submit-v27 quality-gate \
	preflight-44-local preflight-44-kaggle preflight-44 submit-44

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

install-agentic:
	$(PYTHON) -m pip install -e ".[agentic,dev]"

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

solve-sample-langgraph:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli solve \
		--input-csv examples/sample_problems.csv \
		--output-csv artifacts/submission_langgraph.csv \
		--debug-json artifacts/debug_langgraph.json \
		--orchestrator langgraph

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
		--per-problem-time-sec 600 \
		--force-full-problem-time \
		--stage-time-reserve-sec 45 \
		--force-tool-round-for-unverified \
		--agentic-tool-rounds 5 \
		--max-code-blocks-per-attempt 6 \
		--parallel-attempt-workers 4 \
		--parallel-code-workers 4 \
		--request-timeout 600 \
		--client-max-retries 2

benchmark-reference-groq-autonomous120b-robust:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli benchmark-reference \
		--reference-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
		--output-dir artifacts/reference_benchmark_autonomous120b_robust \
		--profile autonomous120b \
		--model openai/gpt-oss-120b \
		--reasoning-effort high \
		--per-problem-time-sec 600 \
		--force-full-problem-time \
		--mandatory-code-attempts 3 \
		--stage-time-reserve-sec 45 \
		--force-tool-round-for-unverified \
		--agentic-tool-rounds 5 \
		--max-code-blocks-per-attempt 6 \
		--parallel-attempt-workers 4 \
		--parallel-code-workers 4 \
		--mini-solver-enabled \
		--strict-zero-one-policy \
		--request-timeout 600 \
		--client-max-retries 2

quality-gate:
	PYTHONPATH=src $(PYTHON) scripts/run_submission_quality_gate.py \
		--output-dir artifacts/quality_gate \
		--profile autonomous120b \
		--reasoning-effort high

benchmark-sweep-groq-autonomous120b:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli benchmark-sweep \
		--reference-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
		--output-dir artifacts/reference_sweep_autonomous120b \
		--profile autonomous120b \
		--model openai/gpt-oss-120b \
		--trial-set standard \
		--per-problem-time-sec 600 \
		--force-full-problem-time \
		--stage-time-reserve-sec 45 \
		--force-tool-round-for-unverified \
		--agentic-tool-rounds 5 \
		--max-code-blocks-per-attempt 6 \
		--parallel-attempt-workers 4 \
		--parallel-code-workers 4 \
		--request-timeout 600 \
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
		--per-problem-time-sec 600 \
		--force-full-problem-time \
		--stage-time-reserve-sec 45 \
		--force-tool-round-for-unverified \
		--agentic-tool-rounds 5 \
		--max-code-blocks-per-attempt 6 \
		--parallel-attempt-workers 4 \
		--parallel-code-workers 4 \
		--request-timeout 600 \
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

kaggle-submit-code:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli kaggle-submit-code \
		--competition ai-mathematical-olympiad-progress-prize-3 \
		--kernel-dir kaggle_kernel_submission \
		--output-file-name submission.parquet \
		--wait

kaggle-kernel-preflight:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli kaggle-kernel-pipeline \
		--competition ai-mathematical-olympiad-progress-prize-3 \
		--kernel-dir kaggle_kernel_submission \
		--output-dir artifacts/kaggle_kernel_output_latest \
		--required-output-file submission.parquet \
		--no-submit

kaggle-kernel-submit:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli kaggle-kernel-pipeline \
		--competition ai-mathematical-olympiad-progress-prize-3 \
		--kernel-dir kaggle_kernel_submission \
		--output-dir artifacts/kaggle_kernel_output_latest \
		--required-output-file submission.parquet \
		--wait

kaggle-kernel-preflight-next:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli kaggle-kernel-pipeline \
		--competition ai-mathematical-olympiad-progress-prize-3 \
		--kernel-dir kaggle_kernel_submission \
		--output-dir artifacts/kaggle_kernel_output_next_ready \
		--required-output-file submission.parquet \
		--no-submit

kaggle-submit-v27:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli kaggle-submit-code \
		--competition ai-mathematical-olympiad-progress-prize-3 \
		--kernel raphaelcaillon/aimo3-progress-prize-3-working \
		--kernel-version 27 \
		--output-file-name submission.parquet \
		--message "AIMO3 v27 preflight-validated offline iterative solver" \
		--wait

preflight-44-local:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli kaggle-kernel-preflight-44 \
		--competition ai-mathematical-olympiad-progress-prize-3 \
		--notebook-dir kaggle_kernel_submission_44_50 \
		--stage local \
		--status-file artifacts/preflight_44_status.json

preflight-44-kaggle:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli kaggle-kernel-preflight-44 \
		--competition ai-mathematical-olympiad-progress-prize-3 \
		--notebook-dir kaggle_kernel_submission_44_50 \
		--stage kaggle \
		--strict-runtime-health \
		--output-dir artifacts/kaggle_kernel_output_44_latest \
		--required-output-file submission.parquet \
		--status-file artifacts/preflight_44_status.json

preflight-44:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli kaggle-kernel-preflight-44 \
		--competition ai-mathematical-olympiad-progress-prize-3 \
		--notebook-dir kaggle_kernel_submission_44_50 \
		--stage all \
		--strict-runtime-health \
		--output-dir artifacts/kaggle_kernel_output_44_latest \
		--required-output-file submission.parquet \
		--status-file artifacts/preflight_44_status.json

submit-44:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli kaggle-submit-44 \
		--competition ai-mathematical-olympiad-progress-prize-3 \
		--notebook-dir kaggle_kernel_submission_44_50 \
		--status-file artifacts/preflight_44_status.json \
		--output-file-name submission.parquet \
		--max-submissions-per-day 1 \
		--wait

notebook-proxy-eval:
	PYTHONPATH=src $(PYTHON) scripts/eval_notebook_proxy.py \
		--notebook kaggle_kernel_submission/aimo3_submission.ipynb \
		--input-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
		--output-dir artifacts/notebook_proxy_eval \
		--max-hash-fallback-rate 0.6 \
		--min-accuracy 0.1

kaggle-pipeline:
	PYTHONPATH=src $(PYTHON) -m aimo3.cli kaggle-pipeline \
		--competition ai-mathematical-olympiad-progress-prize-3 \
		--input-csv /path/to/test.csv \
		--output-csv artifacts/submission.csv \
		--debug-json artifacts/debug_traces.json \
		--attempts 8 \
		--wait
