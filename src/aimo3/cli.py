"""Command-line interface for AIMO3 experimentation and Kaggle automation."""

from __future__ import annotations

import argparse
import datetime as dt
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


def _build_solver_from_args(args: argparse.Namespace) -> AIMO3Solver:
    model = args.model or os.getenv("AIMO_MODEL", "openai/gpt-oss-120b")
    base_url = args.base_url or os.getenv("AIMO_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = args.api_key or os.getenv("AIMO_API_KEY")

    client = OpenAICompatChatClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_sec=args.request_timeout,
        extra_body={"top_p": args.top_p},
    )

    config = SolverConfig(
        attempts=args.attempts,
        temperatures=tuple(args.temperatures),
        max_tokens=args.max_tokens,
        min_consensus=args.min_consensus,
        early_stop_attempt=args.early_stop_attempt,
        max_code_blocks_per_attempt=args.max_code_blocks_per_attempt,
        default_answer=args.default_answer,
    )

    return AIMO3Solver(client=client, config=config)


def cmd_solve(args: argparse.Namespace) -> None:
    solver = _build_solver_from_args(args)

    problems = pd.read_csv(args.input_csv)
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
    solve.add_argument("--model", default=None)
    solve.add_argument("--base-url", default=None)
    solve.add_argument("--api-key", default=None)
    solve.add_argument("--attempts", type=int, default=8)
    solve.add_argument("--temperatures", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.45])
    solve.add_argument("--max-tokens", type=int, default=1024)
    solve.add_argument("--top-p", type=float, default=0.95)
    solve.add_argument("--min-consensus", type=int, default=3)
    solve.add_argument("--early-stop-attempt", type=int, default=4)
    solve.add_argument("--max-code-blocks-per-attempt", type=int, default=2)
    solve.add_argument("--default-answer", type=int, default=0)
    solve.add_argument("--request-timeout", type=int, default=180)
    solve.add_argument("--quiet", action="store_true")
    solve.set_defaults(func=cmd_solve)

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
    kpipe.add_argument("--model", default=None)
    kpipe.add_argument("--base-url", default=None)
    kpipe.add_argument("--api-key", default=None)
    kpipe.add_argument("--attempts", type=int, default=8)
    kpipe.add_argument("--temperatures", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.45])
    kpipe.add_argument("--max-tokens", type=int, default=1024)
    kpipe.add_argument("--top-p", type=float, default=0.95)
    kpipe.add_argument("--min-consensus", type=int, default=3)
    kpipe.add_argument("--early-stop-attempt", type=int, default=4)
    kpipe.add_argument("--max-code-blocks-per-attempt", type=int, default=2)
    kpipe.add_argument("--default-answer", type=int, default=0)
    kpipe.add_argument("--request-timeout", type=int, default=180)
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
