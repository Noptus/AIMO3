#!/usr/bin/env python3
"""Run a lightweight quality gate before submission."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="AIMO3 submission quality gate")
    parser.add_argument("--output-dir", default="artifacts/quality_gate")
    parser.add_argument(
        "--reference-csv",
        default="reference/ai-mathematical-olympiad-progress-prize-3/reference.csv",
    )
    parser.add_argument(
        "--hard-csv",
        default="examples/extreme_synthetic_problems.csv",
    )
    parser.add_argument("--profile", default="autonomous120b")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--min-reference-accuracy", type=float, default=0.6)
    parser.add_argument("--max-reference-small-answer-rate", type=float, default=0.2)
    parser.add_argument("--min-independent-check-rate", type=float, default=0.4)
    parser.add_argument("--hard-limit", type=int, default=24)
    parser.add_argument("--reference-limit", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        sys.executable,
        "-m",
        "aimo3.cli",
        "benchmark-reference",
        "--profile",
        args.profile,
        "--reasoning-effort",
        args.reasoning_effort,
        "--model",
        "openai/gpt-oss-120b",
    ]

    ref_out = out / "reference"
    ref_cmd = base_cmd + [
        "--reference-csv",
        args.reference_csv,
        "--output-dir",
        str(ref_out),
    ]
    if args.reference_limit > 0:
        ref_cmd.extend(["--limit", str(args.reference_limit)])
    _run(ref_cmd)

    hard_out = out / "hard"
    hard_cmd = base_cmd + [
        "--reference-csv",
        args.hard_csv,
        "--output-dir",
        str(hard_out),
        "--answer-col",
        "answer",
        "--id-col",
        "id",
        "--problem-col",
        "problem",
    ]
    if args.hard_limit > 0:
        hard_cmd.extend(["--limit", str(args.hard_limit)])
    _run(hard_cmd)

    ref_summary = _load_json(ref_out / "reference_summary.json")
    ref_accuracy = float(ref_summary.get("accuracy", 0.0))

    ref_pred = pd.read_csv(ref_out / "reference_predictions.csv")
    small_rate = float(ref_pred["answer"].isin([0, 1]).mean()) if len(ref_pred) else 1.0

    ref_debug = _load_json(ref_out / "reference_debug.json")
    rows = ref_debug.get("rows", []) if isinstance(ref_debug, dict) else []
    independent_rates: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        summary = row.get("summary", {})
        if not isinstance(summary, dict):
            continue
        cands = float(summary.get("candidate_count", 0.0) or 0.0)
        indep = float(summary.get("independent_check_candidates", 0.0) or 0.0)
        if cands > 0:
            independent_rates.append(indep / cands)
    independent_rate = _mean(independent_rates)

    hard_summary = _load_json(hard_out / "reference_summary.json")
    hard_accuracy = float(hard_summary.get("accuracy", 0.0))

    checks = [
        (
            ref_accuracy >= args.min_reference_accuracy,
            f"reference_accuracy={ref_accuracy:.3f} (min {args.min_reference_accuracy:.3f})",
        ),
        (
            small_rate <= args.max_reference_small_answer_rate,
            f"small_answer_rate={small_rate:.3f} (max {args.max_reference_small_answer_rate:.3f})",
        ),
        (
            independent_rate >= args.min_independent_check_rate,
            f"independent_check_rate={independent_rate:.3f} (min {args.min_independent_check_rate:.3f})",
        ),
        (
            hard_accuracy > 0.0,
            f"hard_synthetic_accuracy={hard_accuracy:.3f} (> 0.000)",
        ),
    ]

    failed = [msg for ok, msg in checks if not ok]
    print("\nQuality gate summary:")
    for ok, msg in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"- {mark}: {msg}")

    if failed:
        print("\nQuality gate failed.", file=sys.stderr)
        return 2

    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", "src")
    raise SystemExit(main())
