#!/usr/bin/env python3
"""Run benchmark-reference in shards to improve long-run stability.

This is useful when provider latency or request variance makes single long runs brittle.
Each shard writes its own logs/artifacts, then merged outputs are produced.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd


def _validate_passthrough(args: list[str]) -> None:
    blocked = {"--reference-csv", "--output-dir"}
    for token in args:
        if token in blocked:
            raise ValueError(f"Do not pass {token} in passthrough args; it is managed per-shard.")


def _run_shard(
    *,
    python_bin: str,
    shard_csv: Path,
    shard_out: Path,
    passthrough_args: list[str],
    env: dict[str, str],
) -> tuple[int, str]:
    cmd = [
        python_bin,
        "-m",
        "aimo3.cli",
        "benchmark-reference",
        "--reference-csv",
        str(shard_csv),
        "--output-dir",
        str(shard_out),
        *passthrough_args,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, log


def main() -> int:
    parser = argparse.ArgumentParser(description="Shard benchmark-reference into smaller runs.")
    parser.add_argument(
        "--reference-csv",
        default="reference/ai-mathematical-olympiad-progress-prize-3/reference.csv",
        help="Labeled reference CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/reference_sharded_run",
        help="Directory to store shard artifacts and merged outputs.",
    )
    parser.add_argument("--shard-size", type=int, default=2, help="Rows per shard.")
    parser.add_argument("--python-bin", default="./.venv/bin/python", help="Python binary to use.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining shards even if one shard fails.",
    )
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Extra args for benchmark-reference (prefix with --).",
    )
    args = parser.parse_args()

    passthrough = list(args.passthrough)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    _validate_passthrough(passthrough)

    ref_path = Path(args.reference_csv).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ref_df = pd.read_csv(ref_path)
    shard_size = max(1, int(args.shard_size))
    total_rows = len(ref_df)

    # Ensure local source imports are available even without editable install.
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    py_path = str(repo_root / "src")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = py_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = py_path

    shard_results: list[dict[str, object]] = []
    merged_frames: list[pd.DataFrame] = []
    failed = False

    for start in range(0, total_rows, shard_size):
        end = min(start + shard_size, total_rows)
        shard_idx = start // shard_size
        shard_df = ref_df.iloc[start:end].copy()

        shard_csv = tmp_dir / f"reference_shard_{shard_idx:03d}.csv"
        shard_out = out_dir / f"run_shard_{shard_idx:03d}"
        shard_out.mkdir(parents=True, exist_ok=True)
        shard_df.to_csv(shard_csv, index=False)

        code, log = _run_shard(
            python_bin=args.python_bin,
            shard_csv=shard_csv,
            shard_out=shard_out,
            passthrough_args=passthrough,
            env=env,
        )
        (shard_out / "run.log").write_text(log, encoding="utf-8")

        pred_path = shard_out / "reference_predictions.csv"
        solved = 0
        total = 0
        if pred_path.exists():
            pred_df = pd.read_csv(pred_path)
            merged_frames.append(pred_df)
            total = len(pred_df)
            if "correct" in pred_df.columns:
                solved = int(pred_df["correct"].sum())

        shard_info = {
            "shard": shard_idx,
            "start_row": start,
            "end_row": end,
            "rows": end - start,
            "exit_code": code,
            "solved": solved,
            "total": total,
            "output_dir": str(shard_out),
        }
        shard_results.append(shard_info)
        print(
            f"[shard {shard_idx:03d}] rows={end-start} exit={code} solved={solved}/{total} "
            f"out={shard_out}"
        )

        if code != 0:
            failed = True
            if not args.continue_on_error:
                break

    merged_path = out_dir / "reference_predictions_merged.csv"
    summary_path = out_dir / "sharded_summary.json"
    shard_summary_path = out_dir / "shard_results.json"

    solved_all = 0
    total_all = 0
    accuracy = 0.0
    if merged_frames:
        merged = pd.concat(merged_frames, ignore_index=True)
        merged.to_csv(merged_path, index=False)
        total_all = len(merged)
        if "correct" in merged.columns:
            solved_all = int(merged["correct"].sum())
            accuracy = float(merged["correct"].mean()) if total_all else 0.0
    else:
        merged = pd.DataFrame()

    shard_summary = {
        "failed_any_shard": failed,
        "total_rows": total_rows,
        "rows_merged": int(total_all),
        "solved": int(solved_all),
        "accuracy": accuracy,
        "shard_size": shard_size,
        "num_shards": len(shard_results),
        "passthrough_args": passthrough,
    }
    shard_summary_path.write_text(json.dumps(shard_results, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(shard_summary, indent=2), encoding="utf-8")

    print(f"Merged predictions: {merged_path}")
    print(f"Shard results: {shard_summary_path}")
    print(f"Summary: {summary_path}")
    if total_all:
        print(f"Global solved: {solved_all}/{total_all} ({accuracy:.1%})")

    return 1 if failed and not args.continue_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
