"""Dataframe-level utilities for AIMO3 inference runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .solver import AIMO3Solver


def run_inference(
    solver: AIMO3Solver,
    problems_df: pd.DataFrame,
    *,
    id_col: str = "id",
    problem_col: str = "problem",
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run solver across a dataframe of problems."""

    required = {id_col, problem_col}
    missing = required - set(problems_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    total = len(problems_df)
    for idx, row in enumerate(problems_df.itertuples(index=False), start=1):
        problem_id = getattr(row, id_col)
        problem_text = getattr(row, problem_col)

        result = solver.solve(problem_text=problem_text, problem_id=problem_id)

        rows.append({"id": problem_id, "answer": int(result.predicted_answer)})
        debug_rows.append(
            {
                "id": problem_id,
                "summary": result.debug_summary,
                "candidates": [
                    {
                        "attempt": c.attempt,
                        "stage": c.stage,
                        "temperature": c.temperature,
                        "answer": c.answer,
                        "answer_source": c.answer_source,
                        "category": c.category,
                        "complexity": c.complexity,
                        "code_verified": c.code_verified,
                        "repair_used": c.repair_used,
                        "code_answers": c.code_answers,
                        "sandbox_errors": c.sandbox_errors,
                        "generation_error": c.generation_error,
                        "score": c.score,
                    }
                    for c in result.candidates
                ],
            }
        )

        if verbose:
            print(f"[{idx:02d}/{total:02d}] id={problem_id} answer={result.predicted_answer}")

    return pd.DataFrame(rows), debug_rows


def save_submission(submission_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save submission with the exact Kaggle-required columns."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    expected = ["id", "answer"]
    if list(submission_df.columns) != expected:
        submission_df = submission_df[expected]

    submission_df.to_csv(output, index=False)
    return output


def save_debug(debug_rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Persist full solve traces for error analysis."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(debug_rows, indent=2), encoding="utf-8")
    return output
