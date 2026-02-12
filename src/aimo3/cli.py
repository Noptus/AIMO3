"""Command-line interface for AIMO3 experimentation and Kaggle automation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
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

    parser.add_argument(
        "--profile",
        choices=["cheap", "balanced", "hard", "aimo120b", "autonomous120b"],
        default="balanced",
    )
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.45])
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None)

    parser.add_argument("--min-consensus", type=int, default=3)
    parser.add_argument("--early-stop-ratio", type=float, default=0.8)
    parser.add_argument("--early-stop-attempt", type=int, default=4)
    parser.add_argument("--max-code-blocks-per-attempt", type=int, default=2)
    parser.add_argument("--agentic-tool-rounds", type=int, default=1)
    parser.add_argument("--agentic-observation-chars", type=int, default=1200)
    parser.add_argument("--agentic-stateful-python", action="store_true", default=True)
    parser.add_argument("--no-agentic-stateful-python", action="store_false", dest="agentic_stateful_python")
    parser.add_argument("--agentic-state-chars", type=int, default=20000)
    parser.add_argument(
        "--stage-time-reserve-sec",
        type=int,
        default=0,
        help="Reserved tail budget for downstream arbitration stages; new attempts stop when reserve is reached.",
    )
    parser.add_argument(
        "--force-tool-round-for-unverified",
        action="store_true",
        default=None,
        dest="force_tool_round_for_unverified",
        help="Require at least one tool follow-up round when answer is unverified and complexity is medium/hard.",
    )
    parser.add_argument(
        "--no-force-tool-round-for-unverified",
        action="store_false",
        dest="force_tool_round_for_unverified",
        help="Disable forced tool follow-up for unverified medium/hard answers.",
    )
    parser.add_argument(
        "--parallel-attempt-workers",
        type=int,
        default=1,
        help="Parallel workers for independent attempts (LLM + tool loops).",
    )
    parser.add_argument(
        "--parallel-code-workers",
        type=int,
        default=1,
        help="Parallel workers for code block execution when stateful python is disabled.",
    )
    parser.add_argument(
        "--per-problem-time-sec",
        type=int,
        default=0,
        help="Hard wall-clock budget per problem in seconds (0 disables).",
    )
    parser.add_argument(
        "--min-time-for-attempt-sec",
        type=int,
        default=20,
        help="Required remaining seconds to launch a new main/recovery attempt.",
    )
    parser.add_argument(
        "--min-time-for-stage-sec",
        type=int,
        default=8,
        help="Required remaining seconds to run optional arbitration stages.",
    )
    parser.add_argument(
        "--force-full-problem-time",
        action="store_true",
        dest="force_full_problem_time",
        default=None,
        help="Disable early-stop and continue searching until time budget expires.",
    )
    parser.add_argument(
        "--no-force-full-problem-time",
        action="store_false",
        dest="force_full_problem_time",
        help="Allow confidence-based early stop even when per-problem time is set.",
    )
    parser.add_argument("--default-answer", type=int, default=0)

    parser.add_argument("--adaptive-complexity", action="store_true", default=True)
    parser.add_argument("--no-adaptive-complexity", action="store_false", dest="adaptive_complexity")
    parser.add_argument("--hard-mode", action="store_true")
    parser.add_argument("--hard-attempts", type=int, default=12)
    parser.add_argument("--hard-max-tokens", type=int, default=4096)

    parser.add_argument("--repair-passes", type=int, default=0)
    parser.add_argument("--final-extractor-passes", type=int, default=1)
    parser.add_argument("--final-extractor-max-tokens", type=int, default=256)
    parser.add_argument("--verification-attempts", type=int, default=0)
    parser.add_argument("--verification-top-k", type=int, default=3)
    parser.add_argument("--consistency-audit-attempts", type=int, default=0)
    parser.add_argument("--consistency-audit-top-k", type=int, default=4)
    parser.add_argument("--consistency-audit-temperature", type=float, default=0.09)
    parser.add_argument("--adversarial-probe-attempts", type=int, default=0)
    parser.add_argument("--adversarial-probe-top-k", type=int, default=4)
    parser.add_argument("--adversarial-probe-temperature", type=float, default=0.16)
    parser.add_argument("--geometry-recheck-attempts", type=int, default=0)
    parser.add_argument("--geometry-top-k", type=int, default=4)
    parser.add_argument("--geometry-recheck-temperature", type=float, default=0.08)
    parser.add_argument("--small-answer-guard-attempts", type=int, default=1)
    parser.add_argument("--small-answer-guard-top-k", type=int, default=3)
    parser.add_argument("--small-answer-guard-temperature", type=float, default=0.12)
    parser.add_argument("--fallback-guess-attempts", type=int, default=1)
    parser.add_argument("--fallback-guess-temperature", type=float, default=0.15)
    parser.add_argument("--selector-attempts", type=int, default=0)
    parser.add_argument("--selector-top-k", type=int, default=4)
    parser.add_argument("--selector-temperature", type=float, default=0.05)
    parser.add_argument("--escalation-attempts", type=int, default=0)
    parser.add_argument("--escalation-temperature", type=float, default=0.55)
    parser.add_argument("--escalation-trigger-ratio", type=float, default=0.72)
    parser.add_argument("--escalation-min-valid", type=int, default=3)

    parser.add_argument("--request-timeout", type=int, default=240)
    parser.add_argument("--client-max-retries", type=int, default=2)
    parser.add_argument("--sparse-recovery-attempts", type=int, default=2)
    parser.add_argument("--sparse-recovery-temperature", type=float, default=0.1)
    parser.add_argument(
        "--enable-code-interpreter",
        action="store_true",
        help="Enable hosted code interpreter tool when backend supports it.",
    )
    parser.add_argument(
        "--disable-code-interpreter",
        action="store_true",
        help="Force-disable hosted code interpreter tool.",
    )


def _profile_overrides(args: argparse.Namespace) -> dict[str, int | bool | float]:
    """Return profile-specific overrides for stronger/cheaper behavior."""

    if args.profile == "cheap":
        return {
            "attempts": min(args.attempts, 2),
            "max_tokens": min(args.max_tokens, 384),
            "hard_mode": False,
            "agentic_tool_rounds": min(args.agentic_tool_rounds, 1),
            "agentic_observation_chars": min(args.agentic_observation_chars, 800),
            "agentic_stateful_python": bool(args.agentic_stateful_python),
            "agentic_state_chars": min(args.agentic_state_chars, 8000),
            "stage_time_reserve_sec": 0,
            "force_tool_round_for_unverified": False,
            "parallel_attempt_workers": 1,
            "parallel_code_workers": 1,
            "repair_passes": 0,
            "final_extractor_passes": 1,
            "final_extractor_max_tokens": min(args.final_extractor_max_tokens, 128),
            "verification_attempts": 0,
            "consistency_audit_attempts": 0,
            "adversarial_probe_attempts": 0,
            "geometry_recheck_attempts": 0,
            "small_answer_guard_attempts": 0,
            "fallback_guess_attempts": 1,
            "fallback_guess_temperature": min(args.fallback_guess_temperature, 0.12),
            "selector_attempts": 0,
            "escalation_attempts": 0,
            "sparse_recovery_attempts": 1,
            "sparse_recovery_temperature": min(args.sparse_recovery_temperature, 0.1),
            "early_stop_ratio": min(max(args.early_stop_ratio, 0.6), 0.75),
            "adaptive_complexity": False,
        }

    if args.profile == "hard":
        return {
            "attempts": max(args.attempts, 12),
            "max_tokens": max(args.max_tokens, 4096),
            "max_code_blocks_per_attempt": max(args.max_code_blocks_per_attempt, 4),
            "agentic_tool_rounds": max(args.agentic_tool_rounds, 2),
            "agentic_observation_chars": max(args.agentic_observation_chars, 1200),
            "agentic_stateful_python": True,
            "agentic_state_chars": max(args.agentic_state_chars, 25000),
            "stage_time_reserve_sec": max(args.stage_time_reserve_sec, 20),
            "force_tool_round_for_unverified": (
                bool(args.force_tool_round_for_unverified)
                if args.force_tool_round_for_unverified is not None
                else True
            ),
            "parallel_attempt_workers": max(2, args.parallel_attempt_workers),
            "parallel_code_workers": max(2, args.parallel_code_workers),
            "hard_mode": True,
            "hard_attempts": max(args.hard_attempts, 12),
            "hard_max_tokens": max(args.hard_max_tokens, 4096),
            "repair_passes": max(args.repair_passes, 1),
            "final_extractor_passes": max(args.final_extractor_passes, 2),
            "final_extractor_max_tokens": max(args.final_extractor_max_tokens, 256),
            "verification_attempts": max(args.verification_attempts, 2),
            "verification_top_k": max(args.verification_top_k, 4),
            "consistency_audit_attempts": max(args.consistency_audit_attempts, 1),
            "consistency_audit_top_k": max(args.consistency_audit_top_k, 4),
            "consistency_audit_temperature": min(args.consistency_audit_temperature, 0.1),
            "adversarial_probe_attempts": max(args.adversarial_probe_attempts, 1),
            "adversarial_probe_top_k": max(args.adversarial_probe_top_k, 4),
            "adversarial_probe_temperature": min(args.adversarial_probe_temperature, 0.16),
            "geometry_recheck_attempts": max(args.geometry_recheck_attempts, 1),
            "geometry_top_k": max(args.geometry_top_k, 4),
            "geometry_recheck_temperature": min(args.geometry_recheck_temperature, 0.08),
            "small_answer_guard_attempts": max(args.small_answer_guard_attempts, 1),
            "small_answer_guard_top_k": max(args.small_answer_guard_top_k, 3),
            "small_answer_guard_temperature": min(args.small_answer_guard_temperature, 0.12),
            "fallback_guess_attempts": max(args.fallback_guess_attempts, 1),
            "fallback_guess_temperature": min(args.fallback_guess_temperature, 0.15),
            "selector_attempts": max(args.selector_attempts, 1),
            "selector_top_k": max(args.selector_top_k, 4),
            "selector_temperature": min(args.selector_temperature, 0.08),
            "escalation_attempts": max(args.escalation_attempts, 1),
            "escalation_temperature": min(args.escalation_temperature, 0.62),
            "escalation_trigger_ratio": min(max(args.escalation_trigger_ratio, 0.65), 0.8),
            "escalation_min_valid": max(args.escalation_min_valid, 2),
            "sparse_recovery_attempts": max(args.sparse_recovery_attempts, 3),
            "sparse_recovery_temperature": min(args.sparse_recovery_temperature, 0.12),
            "early_stop_ratio": max(args.early_stop_ratio, 0.85),
            "adaptive_complexity": True,
        }

    if args.profile == "aimo120b":
        return {
            "attempts": max(args.attempts, 8),
            "max_tokens": max(args.max_tokens, 4096),
            "max_code_blocks_per_attempt": max(args.max_code_blocks_per_attempt, 4),
            "agentic_tool_rounds": max(args.agentic_tool_rounds, 2),
            "agentic_observation_chars": max(args.agentic_observation_chars, 1400),
            "agentic_stateful_python": True,
            "agentic_state_chars": max(args.agentic_state_chars, 30000),
            "stage_time_reserve_sec": max(args.stage_time_reserve_sec, 30),
            "force_tool_round_for_unverified": (
                bool(args.force_tool_round_for_unverified)
                if args.force_tool_round_for_unverified is not None
                else True
            ),
            "parallel_attempt_workers": max(3, args.parallel_attempt_workers),
            "parallel_code_workers": max(3, args.parallel_code_workers),
            "hard_mode": True,
            "hard_attempts": max(args.hard_attempts, 12),
            "hard_max_tokens": max(args.hard_max_tokens, 4096),
            "repair_passes": max(args.repair_passes, 1),
            "final_extractor_passes": max(args.final_extractor_passes, 2),
            "final_extractor_max_tokens": max(args.final_extractor_max_tokens, 256),
            "verification_attempts": max(args.verification_attempts, 3),
            "verification_top_k": max(args.verification_top_k, 4),
            "consistency_audit_attempts": max(args.consistency_audit_attempts, 2),
            "consistency_audit_top_k": max(args.consistency_audit_top_k, 4),
            "consistency_audit_temperature": min(args.consistency_audit_temperature, 0.09),
            "adversarial_probe_attempts": max(args.adversarial_probe_attempts, 2),
            "adversarial_probe_top_k": max(args.adversarial_probe_top_k, 4),
            "adversarial_probe_temperature": min(args.adversarial_probe_temperature, 0.14),
            "geometry_recheck_attempts": max(args.geometry_recheck_attempts, 2),
            "geometry_top_k": max(args.geometry_top_k, 4),
            "geometry_recheck_temperature": min(args.geometry_recheck_temperature, 0.07),
            "small_answer_guard_attempts": max(args.small_answer_guard_attempts, 2),
            "small_answer_guard_top_k": max(args.small_answer_guard_top_k, 3),
            "small_answer_guard_temperature": min(args.small_answer_guard_temperature, 0.1),
            "fallback_guess_attempts": max(args.fallback_guess_attempts, 1),
            "fallback_guess_temperature": min(args.fallback_guess_temperature, 0.12),
            "selector_attempts": max(args.selector_attempts, 2),
            "selector_top_k": max(args.selector_top_k, 4),
            "selector_temperature": min(args.selector_temperature, 0.06),
            "escalation_attempts": max(args.escalation_attempts, 2),
            "escalation_temperature": min(args.escalation_temperature, 0.6),
            "escalation_trigger_ratio": min(max(args.escalation_trigger_ratio, 0.65), 0.78),
            "escalation_min_valid": max(args.escalation_min_valid, 3),
            "sparse_recovery_attempts": max(args.sparse_recovery_attempts, 4),
            "sparse_recovery_temperature": min(args.sparse_recovery_temperature, 0.12),
            "early_stop_ratio": max(args.early_stop_ratio, 0.85),
            "adaptive_complexity": True,
        }

    if args.profile == "autonomous120b":
        force_full = args.force_full_problem_time if args.force_full_problem_time is not None else True
        per_problem_time = args.per_problem_time_sec if args.per_problem_time_sec > 0 else 600
        return {
            "attempts": max(args.attempts, 16),
            "max_tokens": max(args.max_tokens, 6144),
            "max_code_blocks_per_attempt": max(args.max_code_blocks_per_attempt, 6),
            "agentic_tool_rounds": max(args.agentic_tool_rounds, 4),
            "agentic_observation_chars": max(args.agentic_observation_chars, 2200),
            "agentic_stateful_python": True,
            "agentic_state_chars": max(args.agentic_state_chars, 40000),
            "stage_time_reserve_sec": max(args.stage_time_reserve_sec, 45),
            "force_tool_round_for_unverified": (
                bool(args.force_tool_round_for_unverified)
                if args.force_tool_round_for_unverified is not None
                else True
            ),
            "parallel_attempt_workers": max(4, args.parallel_attempt_workers),
            "parallel_code_workers": max(4, args.parallel_code_workers),
            "hard_mode": True,
            "hard_attempts": max(args.hard_attempts, 16),
            "hard_max_tokens": max(args.hard_max_tokens, 6144),
            "repair_passes": max(args.repair_passes, 2),
            "final_extractor_passes": max(args.final_extractor_passes, 2),
            "final_extractor_max_tokens": max(args.final_extractor_max_tokens, 320),
            "verification_attempts": max(args.verification_attempts, 4),
            "verification_top_k": max(args.verification_top_k, 5),
            "consistency_audit_attempts": max(args.consistency_audit_attempts, 3),
            "consistency_audit_top_k": max(args.consistency_audit_top_k, 5),
            "consistency_audit_temperature": min(args.consistency_audit_temperature, 0.08),
            "adversarial_probe_attempts": max(args.adversarial_probe_attempts, 3),
            "adversarial_probe_top_k": max(args.adversarial_probe_top_k, 5),
            "adversarial_probe_temperature": min(args.adversarial_probe_temperature, 0.12),
            "geometry_recheck_attempts": max(args.geometry_recheck_attempts, 3),
            "geometry_top_k": max(args.geometry_top_k, 5),
            "geometry_recheck_temperature": min(args.geometry_recheck_temperature, 0.06),
            "small_answer_guard_attempts": max(args.small_answer_guard_attempts, 2),
            "small_answer_guard_top_k": max(args.small_answer_guard_top_k, 4),
            "small_answer_guard_temperature": min(args.small_answer_guard_temperature, 0.1),
            "fallback_guess_attempts": max(args.fallback_guess_attempts, 1),
            "fallback_guess_temperature": min(args.fallback_guess_temperature, 0.1),
            "selector_attempts": max(args.selector_attempts, 3),
            "selector_top_k": max(args.selector_top_k, 5),
            "selector_temperature": min(args.selector_temperature, 0.05),
            "escalation_attempts": max(args.escalation_attempts, 3),
            "escalation_temperature": min(args.escalation_temperature, 0.58),
            "escalation_trigger_ratio": min(max(args.escalation_trigger_ratio, 0.66), 0.76),
            "escalation_min_valid": max(args.escalation_min_valid, 3),
            "sparse_recovery_attempts": max(args.sparse_recovery_attempts, 6),
            "sparse_recovery_temperature": min(args.sparse_recovery_temperature, 0.1),
            "early_stop_ratio": max(args.early_stop_ratio, 0.9),
            "adaptive_complexity": True,
            "per_problem_time_sec": per_problem_time,
            "min_time_for_attempt_sec": max(args.min_time_for_attempt_sec, 25),
            "min_time_for_stage_sec": max(args.min_time_for_stage_sec, 10),
            "force_full_problem_time": bool(force_full),
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

    is_groq = "api.groq.com" in (base_url or "")
    is_gpt_oss = model.startswith("openai/gpt-oss-")
    auto_enable_code_tool = is_groq and is_gpt_oss
    if args.disable_code_interpreter:
        use_code_tool = False
    elif args.enable_code_interpreter:
        use_code_tool = True
    else:
        use_code_tool = auto_enable_code_tool

    if use_code_tool:
        # Groq gpt-oss models can fail with `tool_use_failed` unless a tool is provided.
        extra_body["tools"] = [{"type": "code_interpreter"}]

    client = OpenAICompatChatClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_sec=args.request_timeout,
        max_retries=args.client_max_retries,
        extra_body=extra_body,
    )

    overrides = _profile_overrides(args)

    def value(name: str):
        return overrides.get(name, getattr(args, name))

    temperatures = tuple(args.temperatures)
    if args.profile in {"aimo120b", "autonomous120b"} and temperatures == (0.15, 0.25, 0.35, 0.45):
        temperatures = (0.1, 0.2, 0.35, 0.5, 0.7, 0.85)

    config = SolverConfig(
        attempts=int(value("attempts")),
        temperatures=temperatures,
        max_tokens=int(value("max_tokens")),
        min_consensus=args.min_consensus,
        early_stop_ratio=float(value("early_stop_ratio")),
        early_stop_attempt=args.early_stop_attempt,
        max_code_blocks_per_attempt=int(value("max_code_blocks_per_attempt")),
        agentic_tool_rounds=int(value("agentic_tool_rounds")),
        agentic_observation_chars=int(value("agentic_observation_chars")),
        agentic_stateful_python=bool(value("agentic_stateful_python")),
        agentic_state_chars=int(value("agentic_state_chars")),
        stage_time_reserve_sec=max(0, int(value("stage_time_reserve_sec"))),
        force_tool_round_for_unverified=bool(value("force_tool_round_for_unverified")),
        parallel_attempt_workers=max(1, int(value("parallel_attempt_workers"))),
        parallel_code_workers=max(1, int(value("parallel_code_workers"))),
        per_problem_time_sec=int(value("per_problem_time_sec")),
        min_time_for_attempt_sec=int(value("min_time_for_attempt_sec")),
        min_time_for_stage_sec=int(value("min_time_for_stage_sec")),
        force_full_problem_time=bool(value("force_full_problem_time")),
        default_answer=args.default_answer,
        adaptive_complexity=bool(value("adaptive_complexity")),
        hard_mode=bool(value("hard_mode")),
        hard_attempts=int(value("hard_attempts")),
        hard_max_tokens=int(value("hard_max_tokens")),
        repair_passes=int(value("repair_passes")),
        final_extractor_passes=int(value("final_extractor_passes")),
        final_extractor_max_tokens=int(value("final_extractor_max_tokens")),
        verification_attempts=int(value("verification_attempts")),
        verification_top_k=int(value("verification_top_k")),
        consistency_audit_attempts=int(value("consistency_audit_attempts")),
        consistency_audit_top_k=int(value("consistency_audit_top_k")),
        consistency_audit_temperature=float(value("consistency_audit_temperature")),
        adversarial_probe_attempts=int(value("adversarial_probe_attempts")),
        adversarial_probe_top_k=int(value("adversarial_probe_top_k")),
        adversarial_probe_temperature=float(value("adversarial_probe_temperature")),
        geometry_recheck_attempts=int(value("geometry_recheck_attempts")),
        geometry_top_k=int(value("geometry_top_k")),
        geometry_recheck_temperature=float(value("geometry_recheck_temperature")),
        small_answer_guard_attempts=int(value("small_answer_guard_attempts")),
        small_answer_guard_top_k=int(value("small_answer_guard_top_k")),
        small_answer_guard_temperature=float(value("small_answer_guard_temperature")),
        fallback_guess_attempts=int(value("fallback_guess_attempts")),
        fallback_guess_temperature=float(value("fallback_guess_temperature")),
        selector_attempts=int(value("selector_attempts")),
        selector_top_k=int(value("selector_top_k")),
        selector_temperature=float(value("selector_temperature")),
        escalation_attempts=int(value("escalation_attempts")),
        escalation_temperature=float(value("escalation_temperature")),
        escalation_trigger_ratio=float(value("escalation_trigger_ratio")),
        escalation_min_valid=int(value("escalation_min_valid")),
        sparse_recovery_attempts=int(value("sparse_recovery_attempts")),
        sparse_recovery_temperature=float(value("sparse_recovery_temperature")),
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


def _build_sweep_trials(args: argparse.Namespace) -> list[dict[str, object]]:
    base_temps = list(args.temperatures)
    expanded_temps = sorted({*base_temps, 0.1, 0.35, 0.55, 0.75, 0.9})
    cool_temps = sorted({*base_temps, 0.08, 0.18, 0.28, 0.4})

    trials: list[dict[str, object]] = [
        {"name": "base", "overrides": {}},
        {
            "name": "escalation_heavy",
            "overrides": {
                "escalation_attempts": max(args.escalation_attempts, 4),
                "escalation_temperature": min(args.escalation_temperature, 0.58),
                "escalation_trigger_ratio": min(max(args.escalation_trigger_ratio, 0.66), 0.76),
                "selector_attempts": max(args.selector_attempts, 3),
            },
        },
        {
            "name": "selector_focus",
            "overrides": {
                "selector_attempts": max(args.selector_attempts, 4),
                "selector_top_k": max(args.selector_top_k, 5),
                "consistency_audit_attempts": max(args.consistency_audit_attempts, 3),
                "adversarial_probe_attempts": max(args.adversarial_probe_attempts, 3),
            },
        },
        {
            "name": "geometry_guard",
            "overrides": {
                "geometry_recheck_attempts": max(args.geometry_recheck_attempts, 4),
                "geometry_top_k": max(args.geometry_top_k, 5),
                "small_answer_guard_attempts": max(args.small_answer_guard_attempts, 3),
            },
        },
        {
            "name": "diverse_temperature",
            "overrides": {
                "temperatures": expanded_temps,
                "attempts": max(args.attempts, len(expanded_temps)),
                "parallel_attempt_workers": max(args.parallel_attempt_workers, 4),
            },
        },
        {
            "name": "cool_verifier",
            "overrides": {
                "temperatures": cool_temps,
                "attempts": max(args.attempts, len(cool_temps)),
                "verification_attempts": max(args.verification_attempts, 4),
                "selector_attempts": max(args.selector_attempts, 3),
            },
        },
        {
            "name": "anti_trivial_strict",
            "overrides": {
                "small_answer_guard_attempts": max(args.small_answer_guard_attempts, 4),
                "small_answer_guard_temperature": min(args.small_answer_guard_temperature, 0.09),
                "fallback_guess_attempts": max(args.fallback_guess_attempts, 2),
                "fallback_guess_temperature": min(args.fallback_guess_temperature, 0.1),
            },
        },
        {
            "name": "parallel_deep",
            "overrides": {
                "parallel_attempt_workers": max(args.parallel_attempt_workers, 6),
                "parallel_code_workers": max(args.parallel_code_workers, 6),
                "agentic_tool_rounds": max(args.agentic_tool_rounds, 5),
                "attempts": max(args.attempts, 18),
            },
        },
        {
            "name": "verification_heavy",
            "overrides": {
                "verification_attempts": max(args.verification_attempts, 5),
                "consistency_audit_attempts": max(args.consistency_audit_attempts, 4),
                "adversarial_probe_attempts": max(args.adversarial_probe_attempts, 4),
                "selector_attempts": max(args.selector_attempts, 4),
            },
        },
        {
            "name": "long_context_agent",
            "overrides": {
                "agentic_observation_chars": max(args.agentic_observation_chars, 2800),
                "agentic_state_chars": max(args.agentic_state_chars, 60000),
                "max_tokens": max(args.max_tokens, 7000),
                "hard_max_tokens": max(args.hard_max_tokens, 7000),
            },
        },
    ]

    if args.trial_set == "quick":
        return trials[:4]
    if args.trial_set == "max":
        return trials
    return trials[:8]


def _compute_debug_metrics(debug_rows: list[dict[str, object]], merged: pd.DataFrame) -> dict[str, float]:
    candidate_counts: list[float] = []
    consensus_ratios: list[float] = []
    verified_counts: list[float] = []
    agentic_counts: list[float] = []
    small_guess_count = 0.0

    for item in debug_rows:
        summary = item.get("summary", {})
        if not isinstance(summary, dict):
            continue
        candidate_count = float(summary.get("candidate_count", 0.0) or 0.0)
        candidate_counts.append(candidate_count)
        verified_counts.append(float(summary.get("verified_candidates", 0.0) or 0.0))
        agentic_counts.append(float(summary.get("agentic_candidates", 0.0) or 0.0))
        top_votes = summary.get("top_votes", [])
        if isinstance(top_votes, list) and top_votes:
            top = top_votes[0]
            if isinstance(top, (list, tuple)) and len(top) >= 2:
                top_count = float(top[1] or 0.0)
                denom = max(1.0, candidate_count)
                consensus_ratios.append(top_count / denom)

    if len(merged):
        small_guess_count = float((merged["answer"].isin([0, 1])).mean())

    def safe_mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    return {
        "mean_candidate_count": safe_mean(candidate_counts),
        "mean_consensus_ratio": safe_mean(consensus_ratios),
        "mean_verified_candidates": safe_mean(verified_counts),
        "mean_agentic_candidates": safe_mean(agentic_counts),
        "small_answer_rate": small_guess_count,
    }


def _render_recommended_command(
    args: argparse.Namespace,
    overrides: dict[str, object],
    *,
    input_csv: str,
    output_csv: str,
    debug_json: str,
) -> str:
    pieces = [
        "PYTHONPATH=src python -m aimo3.cli solve",
        f"--input-csv {input_csv}",
        f"--output-csv {output_csv}",
        f"--debug-json {debug_json}",
        f"--profile {args.profile}",
        f"--model {args.model or os.getenv('AIMO_MODEL') or 'openai/gpt-oss-120b'}",
    ]

    for key, value in sorted(overrides.items()):
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                pieces.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            items = " ".join(str(v) for v in value)
            pieces.append(f"{flag} {items}")
            continue
        pieces.append(f"{flag} {value}")

    return " \\\n  ".join(pieces)


def cmd_benchmark_sweep(args: argparse.Namespace) -> None:
    reference_path = _validate_input_path(args.reference_csv)
    reference_df = pd.read_csv(reference_path)
    required = {args.id_col, args.problem_col, args.answer_col}
    missing = required - set(reference_df.columns)
    if missing:
        raise ValueError(f"Reference CSV missing columns: {sorted(missing)}")

    if args.limit and args.limit > 0:
        reference_df = reference_df.head(args.limit)

    trial_specs = _build_sweep_trials(args)
    if args.max_trials and args.max_trials > 0:
        trial_specs = trial_specs[: args.max_trials]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for idx, spec in enumerate(trial_specs, start=1):
        name = str(spec["name"])
        overrides = dict(spec.get("overrides", {}))
        trial_args = argparse.Namespace(**vars(args))
        for key, value in overrides.items():
            setattr(trial_args, key, value)

        solver = _build_solver_from_args(trial_args)
        trial_dir = trials_dir / f"{idx:02d}_{name}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()

        submission_df, debug_rows = run_inference(
            solver,
            reference_df,
            id_col=args.id_col,
            problem_col=args.problem_col,
            verbose=not args.quiet,
        )
        elapsed = time.perf_counter() - started

        truth = reference_df[[args.id_col, args.answer_col]].rename(
            columns={args.id_col: "id", args.answer_col: "answer_true"}
        )
        merged = submission_df.merge(truth, on="id", how="left")
        merged["correct"] = merged["answer"] == merged["answer_true"]

        solved = int(merged["correct"].sum())
        total = int(len(merged))
        accuracy = float(merged["correct"].mean()) if total else 0.0
        debug_metrics = _compute_debug_metrics(debug_rows, merged)

        pred_path = trial_dir / "reference_predictions.csv"
        debug_path = trial_dir / "reference_debug.json"
        merged.to_csv(pred_path, index=False)
        save_debug(debug_rows, debug_path)

        row = {
            "trial_index": idx,
            "trial_name": name,
            "solved": solved,
            "total": total,
            "accuracy": accuracy,
            "runtime_sec": round(elapsed, 3),
            "profile": args.profile,
            "model": args.model or os.getenv("AIMO_MODEL") or "openai/gpt-oss-120b",
            "overrides": overrides,
            **debug_metrics,
        }
        all_rows.append(row)
        print(
            f"[trial {idx:02d}/{len(trial_specs):02d}] {name}: "
            f"{solved}/{total} ({accuracy:.1%}) runtime={elapsed:.1f}s small={debug_metrics['small_answer_rate']:.1%}"
        )

    ranked = sorted(
        all_rows,
        key=lambda r: (
            float(r["accuracy"]),
            -float(r["small_answer_rate"]),
            float(r["mean_consensus_ratio"]),
            -float(r["runtime_sec"]),
        ),
        reverse=True,
    )

    leaderboard_path = output_dir / "sweep_leaderboard.csv"
    leaderboard_json_path = output_dir / "sweep_leaderboard.json"
    best_path = output_dir / "best_trial.json"

    pd.DataFrame(ranked).to_csv(leaderboard_path, index=False)
    leaderboard_json_path.write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    best = ranked[0] if ranked else {}
    if best:
        best_overrides = best.get("overrides", {})
        if isinstance(best_overrides, dict):
            best["recommended_command"] = _render_recommended_command(
                args,
                best_overrides,
                input_csv="data/raw/test.csv",
                output_csv="artifacts/submission_best_from_sweep.csv",
                debug_json="artifacts/debug_best_from_sweep.json",
            )
    best_path.write_text(json.dumps(best, indent=2), encoding="utf-8")

    print(f"Saved sweep leaderboard: {leaderboard_path}")
    print(f"Saved sweep leaderboard JSON: {leaderboard_json_path}")
    print(f"Saved best trial: {best_path}")


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

    sweep = sub.add_parser(
        "benchmark-sweep",
        help="Run multiple high-effort trial variants on reference data and select the best configuration",
    )
    sweep.add_argument("--reference-csv", default="reference/reference_with_answers.csv")
    sweep.add_argument("--output-dir", default="artifacts/reference_sweep")
    sweep.add_argument("--id-col", default="id")
    sweep.add_argument("--problem-col", default="problem")
    sweep.add_argument("--answer-col", default="answer")
    sweep.add_argument("--limit", type=int, default=0)
    sweep.add_argument("--trial-set", choices=["quick", "standard", "max"], default="standard")
    sweep.add_argument("--max-trials", type=int, default=0)
    _add_solver_args(sweep)
    sweep.add_argument("--quiet", action="store_true")
    sweep.set_defaults(func=cmd_benchmark_sweep)

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
