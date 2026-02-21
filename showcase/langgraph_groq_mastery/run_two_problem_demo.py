#!/usr/bin/env python3
"""Run the LangGraph Groq showcase solver on two AIMO-style problems."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(override=False)


def main() -> int:
    _load_dotenv_if_present()
    LangGraphGroqShowcaseSolver = importlib.import_module(
        "langgraph_groq_solver"
    ).LangGraphGroqShowcaseSolver

    parser = argparse.ArgumentParser(description="LangGraph Groq showcase two-problem runner")
    parser.add_argument(
        "--reference-csv",
        default="examples/hard_synthetic_problems.csv",
    )
    parser.add_argument("--id-a", default="hard_001")
    parser.add_argument("--id-b", default="hard_009")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--backend", choices=["groq", "litellm"], default="groq")
    parser.add_argument("--time-budget-sec", type=int, default=180)
    parser.add_argument("--max-attempt-workers", type=int, default=4)
    parser.add_argument(
        "--output-json",
        default="showcase/langgraph_groq_mastery/artifacts/two_problem_demo_results.json",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.reference_csv)
    chosen = df[df["id"].isin([args.id_a, args.id_b])].copy()
    if len(chosen) != 2:
        raise ValueError(
            f"Could not find both ids in {args.reference_csv}. Requested: {args.id_a}, {args.id_b}"
        )

    solver = LangGraphGroqShowcaseSolver(
        model=args.model,
        backend=args.backend,
        api_key=os.getenv("GROQ_API_KEY") or os.getenv("AIMO_API_KEY"),
        max_attempt_workers=args.max_attempt_workers,
        time_budget_sec=args.time_budget_sec,
    )

    records: list[dict[str, object]] = []
    for _, row in chosen.iterrows():
        outcome = solver.solve(problem_id=str(row["id"]), problem_text=str(row["problem"]))
        truth = int(row["answer"]) if "answer" in row else None
        correct = bool(truth is not None and int(outcome.answer) == int(truth))
        records.append(
            {
                "id": str(row["id"]),
                "truth_answer": truth,
                "predicted_answer": int(outcome.answer),
                "correct": correct,
                "modulus": outcome.modulus,
                "final_reason": outcome.final_reason,
                "elapsed_sec": round(outcome.elapsed_sec, 3),
                "verifier_votes": outcome.verifier_votes,
                "candidate_count": len(outcome.candidates),
                "candidates": outcome.candidates,
            }
        )
        print(
            f"id={row['id']} predicted={outcome.answer} truth={truth} "
            f"correct={correct} elapsed={outcome.elapsed_sec:.1f}s"
        )

    solved = sum(1 for r in records if bool(r["correct"]))
    total = len(records)
    summary = {
        "solver": "LangGraphGroqShowcaseSolver",
        "backend": args.backend,
        "model": args.model,
        "solved": solved,
        "total": total,
        "accuracy": float(solved / total) if total else 0.0,
        "time_budget_sec": args.time_budget_sec,
        "records": records,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved demo results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
