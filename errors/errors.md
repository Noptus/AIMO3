# Kaggle Submission Error Summary (2026-02-14)

## What Failed

- Kernel `v33` completed and produced `submission.parquet`, but runtime inference was not healthy.
- Kaggle allocated GPU capability `CUDA (6, 0)` (P100-class environment).
- The mounted model was detected correctly:
  - `/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1`
- Model load then failed with OOM / unsupported quantization path on this GPU class.
- Predictions fell back to non-model logic (`pattern_solver`) on sample rows.

## Root Cause

The offline notebook attempted to run `gpt-oss-120b` on a weak Kaggle GPU runtime (P100 / compute capability 6.0), which is insufficient for this setup.

## Why This Is Critical

- A technically valid `submission.parquet` can still be low quality if model inference never actually ran.
- This is a major cause of accepted submissions that score `0`.

## Hardening Added

- `model_sources` updated to an accessible Kaggle model source:
  - `danielhanchen/gpt-oss-120b/transformers/default/1`
- Strict notebook preflight now hard-fails on weak GPU in Kaggle offline mode.
- CLI kernel pipeline now validates runtime health (debug sources + logs) and aborts on:
  - `disabled:model_*` statuses
  - OOM / model load failure markers
- On kernel failure, pipeline now auto-downloads failure artifacts/logs.

## Current Status

- `v34` / `v35` now fail fast with:
  - `Competition preflight failed: weak GPU detected (CUDA capability=(6, 0))`
- This is the intended safe behavior until a stronger runtime is available.

## Next Submission Gate

Do not submit unless preflight passes with:

1. kernel status `complete`
2. valid `submission.parquet`
3. healthy runtime evidence (no disabled model status, no OOM/model_load_failed markers)
