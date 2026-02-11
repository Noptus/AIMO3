"""Command-line interface for AIMO3 experimentation and Kaggle automation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd

from .client import OpenAICompatChatClient
from .kaggle_api import KaggleAutomation
from .pipeline import run_inference, save_debug, save_submission
from .solver import AIMO3Solver, SolverConfig


def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    load_dotenv(override=False)


def _add_solver_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)

    parser.add_argument("--profile", choices=["cheap", "balanced", "hard"], default="balanced")
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.45])
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None)

    parser.add_argument("--min-consensus", type=int, default=3)
    parser.add_argument("--early-stop-attempt", type=int, default=4)
    parser.add_argument("--max-code-blocks-per-attempt", type=int, default=2)
    parser.add_argument("--default-answer", type=int, default=0)

    parser.add_argument("--adaptive-complexity", action="store_true", default=True)
    parser.add_argument("--no-adaptive-complexity", action="store_false", dest="adaptive_complexity")
    parser.add_argument("--hard-mode", action="store_true")
    parser.add_argument("--hard-attempts", type=int, default=12)
    parser.add_argument("--hard-max-tokens", type=int, default=1536)

    parser.add_argument("--repair-passes", type=int, default=0)
    parser.add_argument("--verification-attempts", type=int, default=0)
    parser.add_argument("--verification-top-k", type=int, default=3)

    parser.add_argument("--request-timeout", type=int, default=180)


def _profile_overrides(args: argparse.Namespace) -> dict[str, int | bool]:
    """Return profile-specific overrides for stronger/cheaper behavior."""

    if args.profile == "cheap":
        return {
            "attempts": min(args.attempts, 2),
            "max_tokens": min(args.max_tokens, 384),
            "hard_mode": False,
            "repair_passes": 0,
            "verification_attempts": 0,
            "adaptive_complexity": False,
        }

    if args.profile == "hard":
        return {
            "attempts": max(args.attempts, 12),
            "max_tokens": max(args.max_tokens, 1536),
            "max_code_blocks_per_attempt": max(args.max_code_blocks_per_attempt, 4),
            "hard_mode": True,
            "hard_attempts": max(args.hard_attempts, 12),
            "hard_max_tokens": max(args.hard_max_tokens, 1536),
            "repair_passes": max(args.repair_passes, 1),
            "verification_attempts": max(args.verification_attempts, 2),
            "adaptive_complexity": True,
        }

    return {}


def _build_solver_from_args(args: argparse.Namespace) -> AIMO3Solver:
    groq_key = os.getenv("GROQ_API_KEY")

    model = args.model or os.getenv("AIMO_MODEL") or "openai/gpt-oss-120b"

    if args.base_url:
        base_url = args.base_url
    elif os.getenv("AIMO_BASE_URL"):
        base_url = os.getenv("AIMO_BASE_URL", "")
    elif groq_key:
        base_url = "https://api.groq.com/openai/v1"
    else:
        base_url = "http://127.0.0.1:8000/v1"

    api_key = args.api_key or os.getenv("AIMO_API_KEY") or groq_key

    extra_body = {"top_p": args.top_p}
    if args.reasoning_effort:
        extra_body["reasoning_effort"] = args.reasoning_effort

    client = OpenAICompatChatClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_sec=args.request_timeout,
        extra_body=extra_body,
    )

    overrides = _profile_overrides(args)

    def value(name: str):
        return overrides.get(name, getattr(args, name))

    config = SolverConfig(
        attempts=int(value("attempts")),
        temperatures=tuple(args.temperatures),
        max_tokens=int(value("max_tokens")),
        min_consensus=args.min_consensus,
        early_stop_attempt=args.early_stop_attempt,
        max_code_blocks_per_attempt=int(value("max_code_blocks_per_attempt")),
        default_answer=args.default_answer,
        adaptive_complexity=bool(value("adaptive_complexity")),
        hard_mode=bool(value("hard_mode")),
        hard_attempts=int(value("hard_attempts")),
        hard_max_tokens=int(value("hard_max_tokens")),
        repair_passes=int(value("repair_passes")),
        verification_attempts=int(value("verification_attempts")),
        verification_top_k=args.verification_top_k,
    )

    return AIMO3Solver(client=client, config=config)


def _validate_input_path(input_csv: str) -> Path:
    input_path = Path(input_csv).expanduser()
    if str(input_path).startswith("/path/to/"):
        raise FileNotFoundError(
            "You passed a placeholder path ('/path/to/...'). "
            "Use a real CSV path, e.g. 'examples/sample_problems.csv' "
            "or your downloaded Kaggle test file."
        )
    if not input_path.exists():
        local_csvs = sorted(Path.cwd().glob("**/*.csv"))
        preview = ", ".join(str(p) for p in local_csvs[:5]) if local_csvs else "none found"
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}. "
            f"Examples in this repo: {preview}"
        )
    return input_path


def cmd_solve(args: argparse.Namespace) -> None:
    solver = _build_solver_from_args(args)

    input_path = _validate_input_path(args.input_csv)
    problems = pd.read_csv(input_path)

    submission_df, debug_rows = run_inference(
        solver,
        problems,
        id_col=args.id_col,
        problem_col=args.problem_col,
        verbose=not args.quiet,
    )

    out = save_submission(submission_df, args.output_csv)
    print(f"Saved submission file: {out}")

    if args.debug_json:
        debug_out = save_debug(debug_rows, args.debug_json)
        print(f"Saved debug traces: {debug_out}")


def cmd_benchmark_reference(args: argparse.Namespace) -> None:
    solver = _build_solver_from_args(args)
    reference_path = _validate_input_path(args.reference_csv)

    reference_df = pd.read_csv(reference_path)
    required = {args.id_col, args.problem_col, args.answer_col}
    missing = required - set(reference_df.columns)
    if missing:
        raise ValueError(f"Reference CSV missing columns: {sorted(missing)}")

    if args.limit and args.limit > 0:
        reference_df = reference_df.head(args.limit)

    submission_df, debug_rows = run_inference(
        solver,
        reference_df,
        id_col=args.id_col,
        problem_col=args.problem_col,
        verbose=not args.quiet,
    )

    truth = reference_df[[args.id_col, args.answer_col]].rename(
        columns={args.id_col: "id", args.answer_col: "answer_true"}
    )
    merged = submission_df.merge(truth, on="id", how="left")
    merged["correct"] = merged["answer"] == merged["answer_true"]

    accuracy = float(merged["correct"].mean()) if len(merged) else 0.0
    solved = int(merged["correct"].sum())
    total = int(len(merged))
    print(f"Reference benchmark: {solved}/{total} solved ({accuracy:.1%})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / "reference_predictions.csv"
    debug_path = output_dir / "reference_debug.json"
    summary_path = output_dir / "reference_summary.json"

    merged.to_csv(pred_path, index=False)
    save_debug(debug_rows, debug_path)
    summary = {
        "solved": solved,
        "total": total,
        "accuracy": accuracy,
        "profile": args.profile,
        "model": args.model or os.getenv("AIMO_MODEL") or "openai/gpt-oss-120b",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved predictions: {pred_path}")
    print(f"Saved debug traces: {debug_path}")
    print(f"Saved summary: {summary_path}")


def cmd_kaggle_download(args: argparse.Namespace) -> None:
    api = KaggleAutomation(args.competition)
    extracted = api.download_competition_files(args.output_dir, unzip=not args.no_unzip)
    if extracted:
        print(f"Downloaded and extracted {len(extracted)} files into {Path(args.output_dir).resolve()}")
    else:
        print(f"Downloaded archive into {Path(args.output_dir).resolve()}")


def cmd_kaggle_submit(args: argparse.Namespace) -> None:
    api = KaggleAutomation(args.competition)

    message = args.message
    if not message:
        timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        message = f"AIMO3 automated submission ({timestamp})"

    response = api.submit(args.submission_csv, message=message)
    print(response)

    if args.wait:
        final_state = api.wait_for_latest(
            poll_seconds=args.poll_seconds,
            timeout_minutes=args.timeout_minutes,
        )
        if final_state:
            print(
                "Final status: "
                f"{final_state.status} public={final_state.public_score} private={final_state.private_score}"
            )


def cmd_kaggle_pipeline(args: argparse.Namespace) -> None:
    solve_args = argparse.Namespace(**vars(args))
    cmd_solve(solve_args)

    submit_args = argparse.Namespace(
        competition=args.competition,
        submission_csv=args.output_csv,
        message=args.message,
        wait=args.wait,
        poll_seconds=args.poll_seconds,
        timeout_minutes=args.timeout_minutes,
    )
    cmd_kaggle_submit(submit_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIMO3 solver and Kaggle automation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("solve", help="Run model inference and create submission.csv")
    solve.add_argument("--input-csv", required=True, help="CSV with columns id,problem")
    solve.add_argument("--output-csv", default="artifacts/submission.csv")
    solve.add_argument("--debug-json", default="artifacts/debug_traces.json")
    solve.add_argument("--id-col", default="id")
    solve.add_argument("--problem-col", default="problem")
    _add_solver_args(solve)
    solve.add_argument("--quiet", action="store_true")
    solve.set_defaults(func=cmd_solve)

    ref = sub.add_parser("benchmark-reference", help="Run solver on a labeled reference CSV")
    ref.add_argument("--reference-csv", default="reference/reference_with_answers.csv")
    ref.add_argument("--output-dir", default="artifacts/reference_benchmark")
    ref.add_argument("--id-col", default="id")
    ref.add_argument("--problem-col", default="problem")
    ref.add_argument("--answer-col", default="answer")
    ref.add_argument("--limit", type=int, default=0)
    _add_solver_args(ref)
    ref.add_argument("--quiet", action="store_true")
    ref.set_defaults(func=cmd_benchmark_reference)

    kdl = sub.add_parser("kaggle-download", help="Download competition files with Kaggle API")
    kdl.add_argument(
        "--competition",
        default="ai-mathematical-olympiad-progress-prize-3",
        help="Kaggle competition slug",
    )
    kdl.add_argument("--output-dir", default="data/raw")
    kdl.add_argument("--no-unzip", action="store_true")
    kdl.set_defaults(func=cmd_kaggle_download)

    ksub = sub.add_parser("kaggle-submit", help="Submit an existing CSV to Kaggle")
    ksub.add_argument("--competition", default="ai-mathematical-olympiad-progress-prize-3")
    ksub.add_argument("--submission-csv", required=True)
    ksub.add_argument("--message", default="")
    ksub.add_argument("--wait", action="store_true")
    ksub.add_argument("--poll-seconds", type=int, default=20)
    ksub.add_argument("--timeout-minutes", type=int, default=60)
    ksub.set_defaults(func=cmd_kaggle_submit)

    kpipe = sub.add_parser(
        "kaggle-pipeline",
        help="Run solve -> submission generation -> Kaggle submit (+optional status polling)",
    )
    kpipe.add_argument("--competition", default="ai-mathematical-olympiad-progress-prize-3")
    kpipe.add_argument("--input-csv", required=True)
    kpipe.add_argument("--output-csv", default="artifacts/submission.csv")
    kpipe.add_argument("--debug-json", default="artifacts/debug_traces.json")
    kpipe.add_argument("--id-col", default="id")
    kpipe.add_argument("--problem-col", default="problem")
    _add_solver_args(kpipe)
    kpipe.add_argument("--message", default="")
    kpipe.add_argument("--wait", action="store_true")
    kpipe.add_argument("--poll-seconds", type=int, default=20)
    kpipe.add_argument("--timeout-minutes", type=int, default=60)
    kpipe.add_argument("--quiet", action="store_true")
    kpipe.set_defaults(func=cmd_kaggle_pipeline)

    return parser


def main() -> None:
    _load_dotenv_if_present()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
