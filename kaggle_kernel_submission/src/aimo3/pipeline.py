"""Dataframe-level utilities for AIMO3 inference runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
                        "archetype": c.archetype,
                        "complexity": c.complexity,
                        "code_verified": c.code_verified,
                        "agent_rounds": c.agent_rounds,
                        "repair_used": c.repair_used,
                        "extractor_used": c.extractor_used,
                        "code_answers": c.code_answers,
                        "sandbox_errors": c.sandbox_errors,
                        "generation_error": c.generation_error,
                        "answer_in_problem": c.answer_in_problem,
                        "method": c.method,
                        "independent_check_passed": c.independent_check_passed,
                        "missing_forced_code_check": c.missing_forced_code_check,
                        "score": c.score,
                    }
                    for c in result.candidates
                ],
            }
        )

        if verbose:
            print(
                f"[{idx:02d}/{total:02d}] id={problem_id} answer={result.predicted_answer}",
                flush=True,
            )

    return pd.DataFrame(rows), debug_rows


@dataclass(frozen=True)
class SubmissionValidationSummary:
    rows_before: int
    rows_after: int
    duplicate_ids: int
    missing_ids_filled: int
    invalid_answers_fixed: int
    out_of_range_normalized: int
    empty_ids_fixed: int
    changed: bool


@dataclass(frozen=True)
class SubmissionFileValidationResult:
    file_format: str
    rows: int
    columns_ok: bool
    changed_by_sanitization: bool
    summary: SubmissionValidationSummary


def sanitize_submission(
    submission_df: pd.DataFrame,
    *,
    input_ids: list[str] | None = None,
    default_answer: int = 0,
) -> tuple[pd.DataFrame, SubmissionValidationSummary]:
    """Normalize a submission to Kaggle-safe `id,answer` format.

    - Keeps exactly two columns: `id`, `answer`
    - Casts ids to non-empty strings
    - Casts answers to integers
    - Normalizes answers to [0, 99999]
    - De-duplicates ids (first occurrence wins)
    - If `input_ids` is provided, aligns output rows/order exactly to those ids
    """

    if "id" not in submission_df.columns or "answer" not in submission_df.columns:
        raise ValueError("Submission must include columns: id, answer")

    rows_before = len(submission_df)
    df = submission_df[["id", "answer"]].copy()

    # Normalize ids and patch empties with deterministic placeholders.
    raw_ids = df["id"].astype(str).str.strip()
    empty_mask = raw_ids.eq("") | raw_ids.str.lower().isin({"nan", "none"})
    empty_ids_fixed = int(empty_mask.sum())
    if empty_ids_fixed:
        for idx in df.index[empty_mask]:
            raw_ids.loc[idx] = f"missing_id_{idx}"
    df["id"] = raw_ids

    # Parse answers robustly and patch invalid values.
    numeric = pd.to_numeric(df["answer"], errors="coerce")
    invalid_mask = numeric.isna()
    invalid_answers_fixed = int(invalid_mask.sum())
    numeric = numeric.fillna(float(default_answer))
    numeric = numeric.astype(int)

    out_of_range_mask = (numeric < 0) | (numeric > 99_999)
    out_of_range_normalized = int(out_of_range_mask.sum())
    numeric = numeric % 100_000
    df["answer"] = numeric.astype("int64")

    duplicate_ids = int(df["id"].duplicated(keep="first").sum())
    if duplicate_ids:
        df = df.drop_duplicates(subset=["id"], keep="first")

    missing_ids_filled = 0
    if input_ids is not None:
        expected_ids = [str(x).strip() for x in input_ids]
        expected_df = pd.DataFrame({"id": expected_ids})
        df = expected_df.merge(df, on="id", how="left")
        miss_mask = df["answer"].isna()
        missing_ids_filled = int(miss_mask.sum())
        if missing_ids_filled:
            df.loc[miss_mask, "answer"] = int(default_answer)
        df["answer"] = df["answer"].astype("int64")

    rows_after = len(df)
    changed = any(
        [
            duplicate_ids > 0,
            missing_ids_filled > 0,
            invalid_answers_fixed > 0,
            out_of_range_normalized > 0,
            empty_ids_fixed > 0,
            rows_after != rows_before,
        ]
    )
    summary = SubmissionValidationSummary(
        rows_before=rows_before,
        rows_after=rows_after,
        duplicate_ids=duplicate_ids,
        missing_ids_filled=missing_ids_filled,
        invalid_answers_fixed=invalid_answers_fixed,
        out_of_range_normalized=out_of_range_normalized,
        empty_ids_fixed=empty_ids_fixed,
        changed=changed,
    )
    return df[["id", "answer"]], summary


def load_submission_file(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Load a CSV/Parquet submission file into a dataframe."""

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Submission file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source), "csv"
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(source), "parquet"
        except ImportError:
            # Keep the pipeline usable without optional parquet engines by
            # validating the sidecar CSV many Kaggle notebooks emit for debug.
            csv_fallback = source.with_suffix(".csv")
            if csv_fallback.exists():
                return pd.read_csv(csv_fallback), "parquet(csv-fallback)"
            raise
    raise ValueError(f"Unsupported submission file format: {source.suffix}")


def validate_submission_file(
    submission_path: str | Path,
    *,
    input_ids: list[str] | None = None,
    default_answer: int = 0,
    strict: bool = True,
) -> tuple[pd.DataFrame, SubmissionFileValidationResult]:
    """Validate a saved submission file and optionally enforce strict no-fix policy.

    `strict=True` means the file must already be Kaggle-safe with no sanitizer repairs.
    """

    raw_df, file_format = load_submission_file(submission_path)
    columns_ok = list(raw_df.columns) == ["id", "answer"]

    sanitized_df, summary = sanitize_submission(
        raw_df,
        input_ids=input_ids,
        default_answer=default_answer,
    )
    result = SubmissionFileValidationResult(
        file_format=file_format,
        rows=len(sanitized_df),
        columns_ok=columns_ok,
        changed_by_sanitization=summary.changed,
        summary=summary,
    )

    if strict and (not columns_ok or summary.changed):
        raise ValueError(
            "Submission file failed strict validation: "
            f"columns_ok={columns_ok}, "
            f"duplicates={summary.duplicate_ids}, "
            f"missing_filled={summary.missing_ids_filled}, "
            f"invalid_fixed={summary.invalid_answers_fixed}, "
            f"range_normalized={summary.out_of_range_normalized}, "
            f"empty_ids_fixed={summary.empty_ids_fixed}"
        )

    return sanitized_df, result


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
