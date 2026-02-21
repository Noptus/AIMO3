# Next Submission Playbook

Date: 2026-02-14
Competition: `ai-mathematical-olympiad-progress-prize-3`
Kernel: `raphaelcaillon/aimo3-progress-prize-3-working`

## Current Status

- `v33`: kernel completed and wrote `submission.parquet`, but runtime health was bad:
  - mounted model: `danielhanchen/gpt-oss-120b/transformers/default/1`
  - GPU in Kaggle run: `CUDA capability: (6, 0)` (P100)
  - model load failed with OOM, predictions fell back to pattern solver on sample rows
- `v34`: strict guard now fails fast with:
  - `RuntimeError: Competition preflight failed: weak GPU detected (CUDA capability=(6, 0))`

Conclusion: the pipeline is now hardened to block unsafe submissions, but this environment is not currently valid for offline `gpt-oss-120b` execution.

## Safe Rule (Do Not Skip)

Do not submit until preflight passes all of:

1. kernel run status is `complete`
2. `submission.parquet` exists and validates
3. runtime health check passes (no `disabled:*` model statuses, no model OOM markers)

## Preflight Command (No Daily Slot Burn)

```bash
cd /Users/raphaelcaillon/Documents/GitHub/AIMO3
PYTHONPATH=src .venv/bin/python -m aimo3.cli kaggle-kernel-pipeline \
  --competition ai-mathematical-olympiad-progress-prize-3 \
  --kernel-dir kaggle_kernel_submission \
  --output-dir artifacts/kaggle_kernel_output_next_ready \
  --required-output-file submission.parquet \
  --no-submit
```

If this command exits non-zero, the setup is not submission-safe.

## Submit Command (Only After Passing Preflight)

```bash
cd /Users/raphaelcaillon/Documents/GitHub/AIMO3
PYTHONPATH=src .venv/bin/python -m aimo3.cli kaggle-kernel-pipeline \
  --competition ai-mathematical-olympiad-progress-prize-3 \
  --kernel-dir kaggle_kernel_submission \
  --output-dir artifacts/kaggle_kernel_output_next_ready \
  --required-output-file submission.parquet \
  --wait
```

This submits the exact kernel version produced by the validated run.

## Note on Sample Validation Zeros

Kaggle local gateway sample ids (`000aaa`, `111bbb`, `222ccc`) are expected to evaluate to `0`.
This is normal and not evidence of hidden-set quality.
