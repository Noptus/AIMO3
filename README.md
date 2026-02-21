# AIMO3

AIMO3 is a production-oriented, agentic math-solving system for Kaggle competition `ai-mathematical-olympiad-progress-prize-3`.

## What This Repo Does

- Solves each problem with multiple diversified `gpt-oss` attempts.
- Runs an autonomous Python sandbox loop inside each attempt.
- Uses cross-stage arbitration (verifier, consistency audit, adversarial probe, geometry recheck, selector).
- Enforces mandatory code-first attempts and structured `RESULT_JSON` outputs for stronger evidence tracking.
- Applies anti-collapse safeguards for weak `0/1` behavior.
- Produces Kaggle-ready outputs and automates Kaggle workflows.
- Ships a hardened Kaggle code-submission pipeline with strict preflight validation.

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

Two orchestrators are now available:

- `classic`: existing handcrafted stage pipeline (`solver.py`).
- `langgraph`: explicit state-machine runtime (`langgraph_solver.py`) with graph transitions:
  draft -> tools -> followup -> commit -> (loop/end).

## Core Components

- `/Users/raphaelcaillon/Documents/GitHub/AIMO3/src/aimo3/solver.py`: orchestration, agent loop, stage pipeline, aggregation.
- `/Users/raphaelcaillon/Documents/GitHub/AIMO3/src/aimo3/langgraph_solver.py`: LangGraph-native orchestration with the same `SolveResult` contract.
- `/Users/raphaelcaillon/Documents/GitHub/AIMO3/src/aimo3/sandbox.py`: constrained Python execution.
- `/Users/raphaelcaillon/Documents/GitHub/AIMO3/src/aimo3/prompts.py`: archetype routing + stage prompt builders.
- `/Users/raphaelcaillon/Documents/GitHub/AIMO3/src/aimo3/parsing.py`: modulus/answer extraction.
- `/Users/raphaelcaillon/Documents/GitHub/AIMO3/src/aimo3/cli.py`: solve, benchmark, Kaggle automation.
- `/Users/raphaelcaillon/Documents/GitHub/AIMO3/kaggle_kernel_submission/`: offline competition notebook package.
- `/Users/raphaelcaillon/Documents/GitHub/AIMO3/roadmap/langgraph_system_design_2026-02-13.md`: graph design notes and migration plan.
- `/Users/raphaelcaillon/Documents/GitHub/AIMO3/roadmap/next_submission_playbook_2026-02-13.md`: exact, validated next-submission procedure.

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
  - defaults to mandatory code-first attempts and strict `0/1` policy.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
cp .env.example .env
```

Install LangGraph orchestrator support:

```bash
pip install -e '.[agentic]'
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

Model policy:
- Runtime is pinned to `openai/gpt-oss-120b` for all solve/benchmark flows.
- Passing another `--model` value is ignored and automatically overridden.

## Main Usage

### Solve

```bash
aimo3 solve \
  --input-csv examples/sample_problems.csv \
  --output-csv artifacts/submission.csv \
  --debug-json artifacts/debug_traces.json \
  --profile balanced \
  --orchestrator classic
```

LangGraph runtime:

```bash
aimo3 solve \
  --input-csv examples/sample_problems.csv \
  --output-csv artifacts/submission_langgraph.csv \
  --debug-json artifacts/debug_langgraph.json \
  --profile autonomous120b \
  --orchestrator langgraph
```

`solve` now enforces submission safety automatically:
- aligns output ids to the input CSV order,
- fixes invalid/non-integer answers,
- normalizes answers to `0..99999`,
- guarantees final CSV schema is exactly `id,answer`.

### High-Budget Autonomous Reference Run (Groq/Online)

```bash
PYTHONPATH=src python -m aimo3.cli benchmark-reference \
  --reference-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
  --output-dir artifacts/reference_benchmark_autonomous120b \
  --profile autonomous120b \
  --model openai/gpt-oss-120b \
  --per-problem-time-sec 600 \
  --force-full-problem-time \
  --mandatory-code-attempts 3 \
  --agentic-tool-rounds 5 \
  --max-code-blocks-per-attempt 6 \
  --parallel-attempt-workers 4 \
  --parallel-code-workers 4 \
  --reasoning-effort high \
  --request-timeout 600 \
  --client-max-retries 2
```

### Pre-Submission Quality Gate

```bash
PYTHONPATH=src python scripts/run_submission_quality_gate.py \
  --output-dir artifacts/quality_gate \
  --profile autonomous120b \
  --reasoning-effort high
```

This runs:
- full reference benchmark,
- hard synthetic benchmark,
- small-answer collapse check,
- independent-check-rate threshold.

If any gate fails, it exits non-zero.

### Harder Local Stress Benchmarks

Generate a harder/extreme synthetic AIMO-style set:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/generate_hard_synthetic_problems.py \
  --difficulty extreme \
  --count 120 \
  --seed 20260213 \
  --output-csv examples/extreme_synthetic_problems.csv \
  --output-unlabeled-csv examples/extreme_synthetic_problems_unlabeled.csv
```

Run a heavy benchmark on that local set:

```bash
PYTHONPATH=src python -m aimo3.cli benchmark-reference \
  --reference-csv examples/extreme_synthetic_problems.csv \
  --output-dir artifacts/extreme_synthetic_benchmark \
  --profile aimo120b \
  --model openai/gpt-oss-120b \
  --per-problem-time-sec 120 \
  --agentic-tool-rounds 4 \
  --max-code-blocks-per-attempt 5 \
  --parallel-attempt-workers 2 \
  --parallel-code-workers 2 \
  --reasoning-effort high \
  --request-timeout 90 \
  --client-max-retries 2
```

For long unstable runs, use sharded execution:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/run_sharded_reference_benchmark.py \
  --reference-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
  --output-dir artifacts/reference_sharded_groq \
  --shard-size 2 \
  --continue-on-error \
  -- \
  --profile aimo120b \
  --model openai/gpt-oss-120b \
  --per-problem-time-sec 90 \
  --agentic-tool-rounds 4 \
  --parallel-attempt-workers 2 \
  --parallel-code-workers 2 \
  --reasoning-effort high \
  --request-timeout 90 \
  --client-max-retries 2
```

### File Submission Pipeline (CSV Upload)

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

`kaggle-submit` also sanitizes the CSV before upload. If fixes are needed, it writes
`<name>.validated.csv` and submits that file.

### Hardened Kaggle Code-Competition Pipeline (Recommended)

Use this for AIMO3 competition submissions (notebook output `submission.parquet`):

```bash
PYTHONPATH=src python -m aimo3.cli kaggle-kernel-pipeline \
  --competition ai-mathematical-olympiad-progress-prize-3 \
  --kernel-dir kaggle_kernel_submission \
  --output-dir artifacts/kaggle_kernel_output_latest \
  --required-output-file submission.parquet \
  --wait
```

What it enforces before submitting:
- pushes a fresh notebook version,
- waits for the kernel run to complete,
- downloads outputs,
- hard-validates `submission.parquet` schema/content (`id,answer`, valid range),
- validates `submission_debug_sources.csv` and kernel log for runtime health (no disabled model / OOM fallback),
- checks daily submission quota (UTC) to avoid guaranteed API-limit errors,
- submits exact kernel version output via Kaggle code-submission API.

Preflight-only (no submit, useful when daily quota is already used):

```bash
PYTHONPATH=src python -m aimo3.cli kaggle-kernel-pipeline \
  --competition ai-mathematical-olympiad-progress-prize-3 \
  --kernel-dir kaggle_kernel_submission \
  --output-dir artifacts/kaggle_kernel_output_latest \
  --required-output-file submission.parquet \
  --no-submit
```

### Staged Gate for 44/50 Notebook (Recommended Workflow)

Run the hardened staged gate for `/Users/raphaelcaillon/Documents/GitHub/AIMO3/kaggle_kernel_submission_44_50`:

```bash
PYTHONPATH=src python -m aimo3.cli kaggle-kernel-preflight-44 \
  --competition ai-mathematical-olympiad-progress-prize-3 \
  --notebook-dir kaggle_kernel_submission_44_50 \
  --stage all \
  --output-dir artifacts/kaggle_kernel_output_44_latest \
  --required-output-file submission.parquet \
  --status-file artifacts/preflight_44_status.json
```

This staged command does:
- local gate: contract/runtime tests (`tests.test_notebook_44_50_contract`, `tests.test_notebook_44_50_local_gate`, pipeline/CLI tests),
- Kaggle gate: push notebook, wait run, download outputs, strict validate `submission.parquet` + `submission_debug_sources.csv`,
- writes status artifact: `artifacts/preflight_44_status.json`.

P100 (`sm_60`) compatibility policy used by the 44/50 notebook:
- GPT-OSS (`mxfp4`) is hard-disabled when `sm < AIMO_DISABLE_GPT_OSS_ON_SM_LT` (default `80`).
- DeepSeek transformers backend becomes primary on `sm_60`.
- Runtime health is written to `runtime_health.json` with keys:
  `solver_warmup_ok`, `selected_model_family`, `selected_model_path`, `gpu_sm`, `backend`, `incompatible_models_skipped`.
- Strict preflight rejects non-sample outputs if warmup is not healthy or safe-mode/fatal markers are detected.

Optional runtime knobs:
- `AIMO_FORCE_MODEL_FAMILY=auto|gpt_oss|deepseek`
- `AIMO_DISABLE_GPT_OSS_ON_SM_LT=80`
- `AIMO_DEEPSEEK_ATTEMPTS_HIGH|AIMO_DEEPSEEK_ATTEMPTS_MED|AIMO_DEEPSEEK_ATTEMPTS_LOW`
- `AIMO_DEEPSEEK_VERIFY_TOP_K`

Submit only after a fresh successful staged preflight:

```bash
PYTHONPATH=src python -m aimo3.cli kaggle-submit-44 \
  --competition ai-mathematical-olympiad-progress-prize-3 \
  --notebook-dir kaggle_kernel_submission_44_50 \
  --status-file artifacts/preflight_44_status.json \
  --output-file-name submission.parquet \
  --wait
```

`kaggle-submit-44` hard-fails if the preflight status is missing, failed, stale, or for a different notebook/competition.
Default workflow policy is one gated submit per UTC day (`--max-submissions-per-day 1`).

Equivalent Make targets:
- `make preflight-44-local`
- `make preflight-44-kaggle`
- `make preflight-44`
- `make submit-44`

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
- `--mandatory-code-attempts`
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
- `--mini-solver-enabled` / `--no-mini-solver-enabled`
- `--mini-solver-min-confidence`
- `--strict-zero-one-policy` / `--no-strict-zero-one-policy`

## Reliability Fixes Applied

Recent failure analysis against reference/debug traces identified:
- transient provider errors (notably HTTP 524) causing weak single-vote outcomes,
- attempt loops consuming full budget and starving verifier/selector stages,
- unverified `FINAL_ANSWER` outputs skipping tool confirmation.

Fixes now in code:
- transient retry coverage expanded (`408/409/429/5xx/52x`),
- stage-time reserve gate in initial-attempt scheduler,
- forced tool round on unverified medium/hard candidates,
- mandatory code-first attempts with forced verification fallback prompt,
- structured `RESULT_JSON` parsing (`answer`, `method`, `independent_check_passed`),
- deterministic mini-solvers for trivial/remainder sub-cases,
- uncertainty-triggered escalation attempts before arbitration.

## Kaggle Notebook Notes

The competition notebook in `/Users/raphaelcaillon/Documents/GitHub/AIMO3/kaggle_kernel_submission/aimo3_submission.ipynb` is hardened to:

- use `kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer`
- call `serve()` in rerun mode (`KAGGLE_IS_COMPETITION_RERUN=1`)
- call `run_local_gateway()` in notebook validation mode
- always write `/kaggle/working/submission.parquet`
- validate output file existence, schema, and row count
- run with internet disabled mode for scoring compatibility
- use stronger offline fallback logic (exact reference hit, retrieval, pattern solver, diversified hash fallback)
- print `/kaggle/input` mount snapshots and vLLM model-discovery notes for faster failure diagnosis

If Kaggle says parquet is missing, ensure you select the latest notebook version that logs `Saved required output: /kaggle/working/submission.parquet`.

To avoid selecting the wrong version manually, use `kaggle-kernel-pipeline` which submits by kernel/version through API.

Important: this competition uses hidden reruns via inference server. A notebook that only reads `/kaggle/input/.../test.csv` may pass local validation but still fail real submissions with a generic Kaggle system error.

### Offline Model Runtime (Kaggle-Safe)

Submission kernel metadata currently uses:

- `danielhanchen/gpt-oss-120b/transformers/default/1`

The submission notebook uses lazy loading:

- `AIMO3InferenceServer.serve()` starts quickly.
- local model loading happens inside `predict()` on first call.
- in strict competition mode, weak/no-model configurations now hard-fail early instead of silently emitting heuristic fallback answers.

Strict competition guards now enforce:

- minimum GPU capability (`AIMO_MIN_REQUIRED_CUDA_MAJOR`, default `8`),
- allowed model hints (`AIMO_ALLOWED_MODEL_HINTS`, default includes `gpt-oss-120b`/`gpt-oss-20b`/Qwen3),
- no heuristic fallback in rerun mode when strict guard is enabled.
- no heuristic fallback in Kaggle offline mode (including local gateway validation) when strict guard is enabled.

The offline solver path is now agentic:

- first-pass reasoning with multiple prompts,
- controlled Python tool execution from extracted ` ```python ` blocks in a subprocess sandbox (`python3 -I`, timeout, blocked dangerous imports/APIs),
- tool output fed back into a second-pass generation,
- weighted majority vote over parsed candidates (`FINAL_ANSWER`, `\boxed{}`, tool outputs, weak tail extraction).
- chat-template prompting when tokenizer supports it (closer to the official Kaggle reference notebook flow).
- parallel traces with iterative follow-up rounds (Kaggle-style multi-trace TIR loop).

Useful notebook runtime knobs:

- `AIMO_LOCAL_MODEL_PATH` (explicit model directory override),
- `AIMO_HF_MAX_INPUT_TOKENS`, `AIMO_HF_MAX_NEW_TOKENS`,
- `AIMO_HF_TEMPERATURE`, `AIMO_HF_TOP_P`,
- `AIMO_HF_MAX_PARALLEL_TRACES`, `AIMO_HF_MAX_ROUNDS`,
- `AIMO_HF_PER_PROBLEM_SEC`, `AIMO_HF_MIN_PER_PROBLEM_SEC`, `AIMO_HF_MAX_PER_PROBLEM_SEC`,
- `AIMO_ESTIMATED_TEST_ROWS` (dynamic per-problem budget allocator),
- `AIMO_HF_MAX_TOOL_BLOCKS`,
- `AIMO_TOOL_TIMEOUT_SEC`, `AIMO_TOOL_MAX_WORKERS`.

P100-safe defaults:

- utility dependency path injection is disabled by default,
- vLLM loading is disabled by default,
- notebook stays on base Kaggle runtime unless explicitly enabled.

Opt-in flags when you intentionally attach a utility dependency kernel and/or run on stronger GPUs:

- `AIMO_ENABLE_UTILITY_PATHS=1`
- `AIMO_HF_PREFER_VLLM=1`
- `AIMO_VLLM_MAX_MODEL_LEN`, `AIMO_VLLM_MAX_NUM_SEQS`, `AIMO_VLLM_GPU_MEMORY_UTILIZATION`

Note: Kaggle submission notebook remains self-contained and does not depend on LangGraph.
LangGraph orchestration is for local/online harness runs.

Recommended push order:

1. `kaggle_kernel_submission` kernel preflight (`--no-submit`).
2. `kaggle_kernel_submission` final run (`kaggle-kernel-submit`) when daily quota is available.

## Score 0 Debug Playbook

If a submission is accepted but scores `0`, validate offline quality before using the next daily slot:

```bash
cd /Users/raphaelcaillon/Documents/GitHub/AIMO3
PYTHONPATH=src .venv/bin/python scripts/eval_notebook_proxy.py \
  --notebook kaggle_kernel_submission/aimo3_submission.ipynb \
  --input-csv reference/ai-mathematical-olympiad-progress-prize-3/reference.csv \
  --output-dir artifacts/notebook_proxy_eval
```

This reports:
- `accuracy` on the proxy reference set,
- `hash_fallback_rate` (critical),
- `source_counts` breakdown.

If `hash_fallback_rate` is high (for example `100%`), the notebook is not truly solving hard problems offline and leaderboard score will likely stay near zero.

If Kaggle run logs show `CUDA capability: (6, 0)` (P100) with strict guard enabled, treat that as misconfigured for this strategy and fix runtime/model setup before spending the next submission.

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
