#!/usr/bin/env python3
"""Evaluate notebook offline predict quality on proxy datasets.

This executes the Kaggle notebook logic locally (without starting the inference server)
and reports answer quality / fallback behavior on reference-style datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _load_notebook_predict(notebook_path: Path) -> tuple[dict[str, object], object]:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    if len(nb.get("cells", [])) < 4:
        raise RuntimeError(f"Unexpected notebook shape: {notebook_path}")

    cell1 = "".join(nb["cells"][1]["source"])
    cell2 = "".join(nb["cells"][2]["source"])
    cell3 = "".join(nb["cells"][3]["source"])
    cell3 = cell3.split("AIMO3_SERVER = _load_inference_server_module()", 1)[0]

    # Dev environments may not have polars installed; use a tiny shim.
    # Newer notebook versions already contain an internal polars fallback.
    if "try:\n    import polars as pl" not in cell3:
        polars_shim = """
try:
    import polars as pl
except Exception:
    class _DummyPLDataFrame(pd.DataFrame):
        pass
    class _DummyPLSeries(pd.Series):
        pass
    class _DummyPL:
        DataFrame = _DummyPLDataFrame
        Series = _DummyPLSeries
    pl = _DummyPL()
"""
        cell3 = cell3.replace("import polars as pl", polars_shim)

    glb: dict[str, object] = {"pd": pd}
    exec(cell1, glb)
    exec(cell2, glb)
    exec(cell3, glb)

    glb["USE_MODEL_API"] = False
    predict = glb.get("predict")
    if predict is None:
        raise RuntimeError("Notebook predict() not found.")

    return glb, predict


def _evaluate(
    *,
    predict,
    glb: dict[str, object],
    input_df: pd.DataFrame,
    id_col: str,
    problem_col: str,
    answer_col: str | None,
    disable_reference: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    debug_rows = glb.get("DEBUG_ROWS")
    if isinstance(debug_rows, list):
        debug_rows.clear()

    if disable_reference:
        glb["REFERENCE_ROWS"] = []
        glb["REFERENCE_ANSWER_MAP"] = {}

    pred = predict(input_df[id_col], input_df[problem_col])
    if hasattr(pred, "to_pandas"):
        pred = pred.to_pandas()
    if not isinstance(pred, pd.DataFrame):
        pred = pd.DataFrame(pred)

    if id_col not in pred.columns:
        if "id" in pred.columns:
            pred[id_col] = pred["id"]
        else:
            raise RuntimeError(f"Prediction frame missing id column `{id_col}` and fallback `id`.")

    if "answer" not in pred.columns:
        raise RuntimeError("Prediction frame missing `answer` column.")

    out = pred[[id_col, "answer"]].copy()
    out[id_col] = out[id_col].astype(str)
    out["answer"] = pd.to_numeric(out["answer"], errors="coerce").fillna(0).astype("int64")

    if answer_col and answer_col in input_df.columns:
        truth_col = "__truth_answer__"
        truth = input_df[[id_col, answer_col]].rename(columns={answer_col: truth_col})
        merged = out.merge(truth, on=id_col, how="left")
        merged["correct"] = merged["answer"].astype("int64") == merged[truth_col].astype("int64")
    else:
        merged = out.copy()
        merged["correct"] = pd.NA

    debug_df = pd.DataFrame(debug_rows) if isinstance(debug_rows, list) else pd.DataFrame()
    return merged, debug_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Proxy-evaluate notebook offline solver quality.")
    parser.add_argument(
        "--notebook",
        default="kaggle_kernel_submission/aimo3_submission.ipynb",
        help="Path to Kaggle submission notebook.",
    )
    parser.add_argument(
        "--input-csv",
        default="reference/ai-mathematical-olympiad-progress-prize-3/reference.csv",
        help="Proxy dataset with id/problem/(optional)answer.",
    )
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--problem-col", default="problem")
    parser.add_argument("--answer-col", default="answer")
    parser.add_argument("--output-dir", default="artifacts/notebook_proxy_eval")
    parser.add_argument("--max-hash-fallback-rate", type=float, default=0.6)
    parser.add_argument("--min-accuracy", type=float, default=0.1)
    parser.add_argument(
        "--disable-reference",
        action="store_true",
        help="Clear notebook reference maps before prediction (recommended for synthetic tests).",
    )
    args = parser.parse_args()

    notebook_path = Path(args.notebook).expanduser()
    input_path = Path(args.input_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    glb, predict = _load_notebook_predict(notebook_path)
    data = pd.read_csv(input_path)
    required = {args.id_col, args.problem_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    answer_col = args.answer_col if args.answer_col in data.columns else None
    merged, debug_df = _evaluate(
        predict=predict,
        glb=glb,
        input_df=data,
        id_col=args.id_col,
        problem_col=args.problem_col,
        answer_col=answer_col,
        disable_reference=bool(args.disable_reference),
    )

    accuracy = None
    if answer_col:
        accuracy = float(merged["correct"].mean()) if len(merged) else 0.0

    hash_fallback_rate = 0.0
    source_counts: dict[str, int] = {}
    if not debug_df.empty and "source" in debug_df.columns:
        source_counts = debug_df["source"].astype(str).value_counts().to_dict()
        hash_count = int(source_counts.get("hash_fallback", 0))
        hash_fallback_rate = hash_count / max(1, len(debug_df))

    small_answer_rate = float(merged["answer"].isin([0, 1]).mean()) if len(merged) else 0.0

    summary = {
        "rows": int(len(merged)),
        "accuracy": accuracy,
        "hash_fallback_rate": hash_fallback_rate,
        "small_answer_rate": small_answer_rate,
        "source_counts": source_counts,
        "notebook": str(notebook_path),
        "input_csv": str(input_path),
    }

    merged.to_csv(output_dir / "predictions.csv", index=False)
    debug_df.to_csv(output_dir / "debug_sources.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))

    if hash_fallback_rate > float(args.max_hash_fallback_rate):
        print(
            f"FAIL: hash_fallback_rate={hash_fallback_rate:.2%} "
            f"> max {args.max_hash_fallback_rate:.2%}",
            file=sys.stderr,
        )
        return 2

    if accuracy is not None and accuracy < float(args.min_accuracy):
        print(
            f"FAIL: accuracy={accuracy:.2%} < min {args.min_accuracy:.2%}",
            file=sys.stderr,
        )
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
