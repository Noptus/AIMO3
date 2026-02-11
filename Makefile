PYTHON ?= python3

.PHONY: install test solve-sample solve-groq-cheap kaggle-download kaggle-submit kaggle-pipeline

install:
	$(PYTHON) -m pip install -e .

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests

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
