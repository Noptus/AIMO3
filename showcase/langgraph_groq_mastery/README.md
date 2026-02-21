# LangGraph Groq Mastery Showcase

This folder is a standalone showcase solver for AIMO-style problems, separate from the production harness.

## What it demonstrates

- Groq-first LLM execution (`gpt-oss-120b` by default).
- Optional LiteLLM backend path for compatible model routing.
- LangGraph graph with explicit nodes:
  - bootstrap and problem profiling,
  - deterministic pattern solver node,
  - parallel candidate generation,
  - verifier arbitration node,
  - confidence-based refinement loop.
- Tool-integrated reasoning with sandboxed Python execution.
- Forced program-synthesis fallback when regular attempts are weak.
- Strict final-output contract:
  - `RESULT_JSON: {"answer": ..., "method": "...", "independent_check_passed": ...}`
  - `FINAL_ANSWER: <integer>`

## Files

- `langgraph_groq_solver.py`: solver implementation.
- `run_two_problem_demo.py`: executes a 2-problem benchmark and writes artifact JSON.
- `artifacts/`: output JSON from demo runs.

## Run

From repo root:

```bash
set -a; source .env; set +a
PYTHONPATH=src ./.venv/bin/python showcase/langgraph_groq_mastery/run_two_problem_demo.py \
  --reference-csv examples/hard_synthetic_problems.csv \
  --id-a hard_001 \
  --id-b hard_009 \
  --backend groq \
  --model openai/gpt-oss-120b \
  --time-budget-sec 220 \
  --max-attempt-workers 4
```

Optional LiteLLM backend:

```bash
pip install litellm
PYTHONPATH=src ./.venv/bin/python showcase/langgraph_groq_mastery/run_two_problem_demo.py \
  --backend litellm \
  --model openai/gpt-oss-120b
```

## Latest local check

- Artifact: `showcase/langgraph_groq_mastery/artifacts/two_problem_demo_strong.json`
- Demo pair: `hard_001`, `hard_009`
- Result: `2/2` correct in Groq mode (`openai/gpt-oss-120b`)
