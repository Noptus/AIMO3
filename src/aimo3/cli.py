"""Command-line interface for AIMO3 experimentation and Kaggle automation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

from .client import OpenAICompatChatClient
from .kaggle_api import KaggleAutomation
from .langgraph_solver import LangGraphAIMO3Solver, LangGraphUnavailableError
from .pipeline import (
    run_inference,
    sanitize_submission,
    save_debug,
    save_submission,
    validate_submission_file,
)
from .solver import AIMO3Solver, SolverConfig

FORCED_MODEL = "openai/gpt-oss-120b"


def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    load_dotenv(override=False)


def _add_solver_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=None,
        help=(f"Requested model identifier. Runtime policy currently enforces {FORCED_MODEL}."),
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--orchestrator",
        choices=["classic", "langgraph"],
        default=os.getenv("AIMO_ORCHESTRATOR", "classic"),
        help="Solver runtime: classic handcrafted pipeline or LangGraph state machine.",
    )

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
    parser.add_argument("--mandatory-code-attempts", type=int, default=0)
    parser.add_argument("--agentic-tool-rounds", type=int, default=1)
    parser.add_argument("--agentic-observation-chars", type=int, default=1200)
    parser.add_argument("--agentic-stateful-python", action="store_true", default=True)
    parser.add_argument(
        "--no-agentic-stateful-python", action="store_false", dest="agentic_stateful_python"
    )
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
    parser.add_argument("--mini-solver-enabled", action="store_true", default=True)
    parser.add_argument(
        "--no-mini-solver-enabled", action="store_false", dest="mini_solver_enabled"
    )
    parser.add_argument("--mini-solver-min-confidence", type=float, default=0.95)
    parser.add_argument("--strict-zero-one-policy", action="store_true", default=True)
    parser.add_argument(
        "--no-strict-zero-one-policy", action="store_false", dest="strict_zero_one_policy"
    )

    parser.add_argument("--adaptive-complexity", action="store_true", default=True)
    parser.add_argument(
        "--no-adaptive-complexity", action="store_false", dest="adaptive_complexity"
    )
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
            "mandatory_code_attempts": min(max(args.mandatory_code_attempts, 0), 1),
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
            "mini_solver_enabled": True,
            "mini_solver_min_confidence": max(args.mini_solver_min_confidence, 0.95),
            "strict_zero_one_policy": True,
        }

    if args.profile == "hard":
        return {
            "attempts": max(args.attempts, 12),
            "max_tokens": max(args.max_tokens, 4096),
            "max_code_blocks_per_attempt": max(args.max_code_blocks_per_attempt, 4),
            "mandatory_code_attempts": max(args.mandatory_code_attempts, 2),
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
            "mini_solver_enabled": True,
            "mini_solver_min_confidence": min(max(args.mini_solver_min_confidence, 0.93), 0.99),
            "strict_zero_one_policy": True,
        }

    if args.profile == "aimo120b":
        return {
            "attempts": max(args.attempts, 8),
            "max_tokens": max(args.max_tokens, 4096),
            "max_code_blocks_per_attempt": max(args.max_code_blocks_per_attempt, 4),
            "mandatory_code_attempts": max(args.mandatory_code_attempts, 2),
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
            "mini_solver_enabled": True,
            "mini_solver_min_confidence": min(max(args.mini_solver_min_confidence, 0.92), 0.99),
            "strict_zero_one_policy": True,
        }

    if args.profile == "autonomous120b":
        force_full = (
            args.force_full_problem_time if args.force_full_problem_time is not None else True
        )
        per_problem_time = args.per_problem_time_sec if args.per_problem_time_sec > 0 else 600
        return {
            "attempts": max(args.attempts, 16),
            "max_tokens": max(args.max_tokens, 6144),
            "max_code_blocks_per_attempt": max(args.max_code_blocks_per_attempt, 6),
            "mandatory_code_attempts": max(args.mandatory_code_attempts, 3),
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
            "mini_solver_enabled": True,
            "mini_solver_min_confidence": min(max(args.mini_solver_min_confidence, 0.9), 0.99),
            "strict_zero_one_policy": True,
        }

    return {}


def _build_solver_from_args(args: argparse.Namespace) -> AIMO3Solver:
    groq_key = os.getenv("GROQ_API_KEY")

    requested_model = args.model or os.getenv("AIMO_MODEL") or FORCED_MODEL
    model = FORCED_MODEL
    if requested_model != FORCED_MODEL:
        print(
            f"[model-policy] Overriding requested model '{requested_model}' to '{FORCED_MODEL}'.",
            flush=True,
        )

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
        mandatory_code_attempts=max(0, int(value("mandatory_code_attempts"))),
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
        mini_solver_enabled=bool(value("mini_solver_enabled")),
        mini_solver_min_confidence=float(value("mini_solver_min_confidence")),
        strict_zero_one_policy=bool(value("strict_zero_one_policy")),
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

    orchestrator = str(getattr(args, "orchestrator", "classic") or "classic").strip().lower()
    if orchestrator == "langgraph":
        try:
            return LangGraphAIMO3Solver(client=client, config=config)
        except LangGraphUnavailableError as exc:
            raise RuntimeError(
                "LangGraph orchestrator requested but dependency is missing. "
                "Install with `pip install -e '.[agentic]'` and retry."
            ) from exc

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
            f"Input CSV not found: {input_path}. Examples in this repo: {preview}"
        )
    return input_path


def _read_kernel_metadata(kernel_dir: str | Path) -> tuple[Path, dict[str, object]]:
    root = Path(kernel_dir).expanduser()
    metadata_path = root / "kernel-metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Kernel metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return root, metadata


def _resolve_kernel_ref(kernel: str | None, kernel_dir: str | Path) -> str:
    if kernel and kernel.strip():
        return kernel.strip()
    _, metadata = _read_kernel_metadata(kernel_dir)
    kernel_ref = str(metadata.get("id", "")).strip()
    if not kernel_ref:
        raise ValueError("Kernel id is missing. Set --kernel or kernel-metadata.json id.")
    return kernel_ref


def _kernel_default_username(kernel_ref: str) -> str | None:
    if "/" not in kernel_ref:
        return None
    owner, _ = kernel_ref.split("/", 1)
    owner = owner.strip()
    return owner if owner else None


def _build_submission_message(message: str, *, prefix: str) -> str:
    if message:
        return message
    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return f"{prefix} ({timestamp})"


def _assert_kernel_metadata_safe(
    *,
    metadata: dict[str, object],
    kernel_root: Path,
    competition: str,
) -> None:
    kernel_ref = str(metadata.get("id", "")).strip()
    if not kernel_ref:
        raise ValueError("kernel-metadata.json must define `id`.")
    if "/" not in kernel_ref:
        raise ValueError("kernel-metadata.json `id` must be <owner>/<slug>.")

    code_file = str(metadata.get("code_file", "")).strip()
    if not code_file:
        raise ValueError("kernel-metadata.json must define `code_file`.")
    code_path = kernel_root / code_file
    if not code_path.exists():
        raise FileNotFoundError(f"Kernel code file not found: {code_path}")

    code_text = code_path.read_text(encoding="utf-8")
    required_markers = [
        "AIMO3InferenceServer",
        "run_local_gateway",
        "KAGGLE_IS_COMPETITION_RERUN",
    ]
    missing_markers = [m for m in required_markers if m not in code_text]
    if missing_markers:
        raise ValueError(
            "Kernel code appears not wired for Kaggle inference-server submission. "
            f"Missing markers in {code_path}: {missing_markers}"
        )

    sources = metadata.get("competition_sources", [])
    if not isinstance(sources, list) or competition not in [str(x) for x in sources]:
        raise ValueError(
            f"kernel-metadata.json `competition_sources` must include `{competition}`."
        )

    enable_internet = metadata.get("enable_internet", False)
    if str(enable_internet).strip().lower() == "true":
        raise ValueError(
            "kernel-metadata.json has `enable_internet=true`. "
            "This competition requires internet disabled."
        )

    model_sources_raw = metadata.get("model_sources", [])
    if not isinstance(model_sources_raw, list):
        raise ValueError("kernel-metadata.json `model_sources` must be a list.")
    model_sources = [str(x).strip() for x in model_sources_raw if str(x).strip()]
    if not model_sources:
        raise ValueError(
            "kernel-metadata.json `model_sources` is empty. "
            "Attach a strong offline model for no-internet competition inference."
        )

    model_blob = " ".join(model_sources).lower()
    required_hints = [
        token.strip().lower()
        for token in os.getenv(
            "AIMO_REQUIRED_MODEL_HINTS",
            "gpt-oss-120b,gpt-oss-20b,qwen3-32b,qwen3-30b-a3b",
        ).split(",")
        if token.strip()
    ]
    weak_hints = [
        token.strip().lower()
        for token in os.getenv(
            "AIMO_WEAK_MODEL_HINTS",
            "deepseek-math-7b,7b-instruct",
        ).split(",")
        if token.strip()
    ]

    has_required = any(hint in model_blob for hint in required_hints)
    has_weak = any(hint in model_blob for hint in weak_hints)
    if not has_required:
        raise ValueError(
            "kernel-metadata.json `model_sources` does not include a required strong model hint. "
            f"Configured={model_sources} required_hints={required_hints}"
        )
    if has_weak and not has_required:
        raise ValueError(
            "kernel-metadata.json appears to use a weak model source "
            f"({model_sources}). This commonly leads to zero-score submissions."
        )


def _check_daily_submission_quota(
    api: KaggleAutomation,
    *,
    max_submissions_per_day: int,
    scan_limit: int,
    skip_if_daily_limit_reached: bool,
) -> bool:
    if max_submissions_per_day <= 0:
        return True

    today_utc = dt.datetime.utcnow().date()
    used_today = api.count_submissions_on_utc_date(today_utc, limit=scan_limit)
    remaining = max_submissions_per_day - used_today
    print(
        f"Submission quota check (UTC {today_utc}): "
        f"used={used_today} max={max_submissions_per_day} remaining={max(0, remaining)}"
    )
    if used_today < max_submissions_per_day:
        return True

    message = (
        "Daily submission allowance already reached. "
        "Skipping submission to avoid a guaranteed Kaggle API error."
    )
    if skip_if_daily_limit_reached:
        print(message)
        return False
    raise RuntimeError(message)


def _validate_kernel_runtime_health(
    *,
    output_dir: str | Path,
    strict: bool,
) -> None:
    out = Path(output_dir).expanduser()
    debug_path = out / "submission_debug_sources.csv"
    runtime_health_path = out / "runtime_health.json"

    if not debug_path.exists():
        message = (
            "Kernel output is missing submission_debug_sources.csv; "
            "cannot verify runtime model health."
        )
        if strict:
            raise FileNotFoundError(message)
        print(f"Warning: {message}")
        return

    debug_df = pd.read_csv(debug_path)
    if debug_df.empty:
        message = "submission_debug_sources.csv is empty; runtime health cannot be verified."
        if strict:
            raise ValueError(message)
        print(f"Warning: {message}")
        return

    required_columns = [
        "id",
        "answer",
        "source",
        "model_status",
        "time_left_s",
        "tool_calls",
        "tool_errors",
        "candidate_count",
        "vote_margin",
    ]
    missing_columns = [col for col in required_columns if col not in debug_df.columns]
    if missing_columns:
        message = (
            "submission_debug_sources.csv is missing required columns: "
            f"{missing_columns}"
        )
        if strict:
            raise ValueError(message)
        print(f"Warning: {message}")
        return

    # Enforce numeric and range integrity for strict runtime diagnostics.
    answer_values = pd.to_numeric(debug_df["answer"], errors="coerce")
    if bool(answer_values.isna().any()):
        raise RuntimeError("Kernel runtime health check failed: non-numeric debug answers")
    if bool(((answer_values < 0) | (answer_values > 99_999)).any()):
        raise RuntimeError("Kernel runtime health check failed: debug answers out of range [0,99999]")

    for numeric_column in ["time_left_s", "tool_calls", "tool_errors", "candidate_count", "vote_margin"]:
        parsed = pd.to_numeric(debug_df[numeric_column], errors="coerce")
        if bool(parsed.isna().any()):
            raise RuntimeError(
                "Kernel runtime health check failed: "
                f"non-numeric debug field `{numeric_column}`"
            )

    if "source" in debug_df.columns:
        source_counts = debug_df["source"].astype(str).value_counts().to_dict()
    else:
        source_counts = {}

    runtime_health: dict[str, object] = {}
    if runtime_health_path.exists():
        try:
            loaded = json.loads(runtime_health_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                runtime_health = loaded
        except Exception:
            runtime_health = {}

    if source_counts and set(source_counts.keys()) == {"sample_validation_passthrough"}:
        if strict:
            if not runtime_health:
                raise RuntimeError(
                    "Runtime health validation failed: sample-only output requires runtime_health.json "
                    "with solver warmup diagnostics."
                )
            if not bool(runtime_health.get("solver_warmup_ok", False)):
                raise RuntimeError(
                    "Runtime health validation failed: solver warmup was not healthy "
                    f"(runtime_health={runtime_health})"
                )
        print(
            "Runtime health validation: sample validation passthrough detected; "
            "skipping offline-model health checks."
        )
        return

    issues: list[str] = []

    if strict and not runtime_health:
        issues.append("missing_runtime_health_json")

    if runtime_health:
        if not bool(runtime_health.get("solver_warmup_ok", False)):
            issues.append(f"runtime_health_solver_warmup_failed={runtime_health}")

        selected_family = str(runtime_health.get("selected_model_family", "")).strip().lower()
        if strict and selected_family and selected_family not in {"gpt_oss", "deepseek"}:
            issues.append(f"invalid_selected_model_family={selected_family}")

        backend = str(runtime_health.get("backend", "")).strip().lower()
        if strict and backend in {"", "none", "disabled", "safe_mode"}:
            issues.append(f"runtime_health_backend_inactive={backend or 'empty'}")

        try:
            gpu_sm = int(runtime_health.get("gpu_sm", 0) or 0)
        except Exception:
            gpu_sm = 0
        if strict and gpu_sm <= 0:
            issues.append(f"invalid_gpu_sm={runtime_health.get('gpu_sm')}")

        warmup_reason = str(runtime_health.get("reason", "")).lower()
        fatal_reason_markers = [
            "quantization unsupported",
            "gpt_oss_quantization_incompatible",
            "server died",
            "model load failed",
            "model_load_failed",
        ]
        detected_reason_markers = [m for m in fatal_reason_markers if m in warmup_reason]
        if detected_reason_markers:
            issues.append(f"runtime_health_reason_markers={detected_reason_markers}")

    if "model_status" in debug_df.columns:
        statuses = debug_df["model_status"].astype(str)
        bad_mask = statuses.str.startswith("disabled:")
        bad_mask = bad_mask | statuses.str.contains("safe_mode", case=False, regex=False)
        if bool(bad_mask.any()):
            bad_values = sorted(set(statuses[bad_mask].tolist()))
            issues.append(f"disabled_model_status={bad_values}")
    else:
        issues.append("missing_model_status_column")

    if not source_counts:
        issues.append("missing_source_column")
    else:
        safe_mode_sources = [
            s for s in source_counts.keys() if "safe_mode" in str(s).lower()
        ]
        if safe_mode_sources:
            issues.append(f"safe_mode_sources={sorted(safe_mode_sources)}")

    log_files = sorted(out.glob("*.log"))
    if log_files:
        log_text = "\n".join(
            p.read_text(encoding="utf-8", errors="ignore").lower() for p in log_files
        )
        fatal_markers = [
            "cuda out of memory",
            "model_load_failed",
            "model_path_not_found",
            "quantization unsupported",
            "server died",
            "model load failed",
        ]
        detected = [m for m in fatal_markers if m in log_text]
        if detected:
            issues.append(f"log_markers={detected}")

    if issues:
        raise RuntimeError(
            "Kernel runtime health check failed: "
            + "; ".join(issues)
            + f"; source_counts={source_counts}"
        )

    print(
        "Runtime health validation: "
        f"rows={len(debug_df)} sources={source_counts}"
    )


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _write_preflight_status(path: str | Path, payload: dict[str, object]) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _load_preflight_status(path: str | Path) -> dict[str, object]:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(
            f"Preflight status file not found: {source}. "
            "Run `kaggle-kernel-preflight-44 --stage all` first."
        )
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse preflight status file: {source}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Invalid preflight status payload in: {source}")
    return data


def _run_local_preflight_44(
    *,
    modules: list[str],
) -> dict[str, object]:
    if not modules:
        return {
            "ok": True,
            "modules": [],
            "details": "no_local_modules_configured",
        }

    cmd = [os.environ.get("PYTHON", os.sys.executable), "-m", "unittest", *modules]
    print("Running local preflight tests:", " ".join(cmd))
    started = time.perf_counter()
    env = dict(os.environ)
    cwd = Path.cwd()
    src_path = str((cwd / "src").resolve())
    existing_pp = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = f"{src_path}:{existing_pp}" if existing_pp else src_path

    completed = subprocess.run(cmd, check=False, env=env)
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        raise RuntimeError(
            "Local preflight tests failed: "
            f"returncode={completed.returncode} modules={modules}"
        )

    return {
        "ok": True,
        "modules": modules,
        "elapsed_sec": round(elapsed, 3),
    }


def _assert_preflight_ready_for_submit(
    *,
    status: dict[str, object],
    notebook_dir: str,
    competition: str,
    max_age_minutes: int,
) -> None:
    if not bool(status.get("ok", False)):
        raise RuntimeError(
            "Preflight status indicates failure. "
            "Run `kaggle-kernel-preflight-44 --stage all` and resolve failures."
        )

    stage = str(status.get("stage", "")).strip().lower()
    if stage not in {"all", "kaggle"}:
        raise RuntimeError(
            "Preflight status does not include Kaggle validation stage. "
            "Run `kaggle-kernel-preflight-44 --stage all`."
        )

    recorded_dir = str(status.get("notebook_dir", "")).strip()
    if recorded_dir and recorded_dir != notebook_dir:
        raise RuntimeError(
            "Preflight notebook dir mismatch. "
            f"status={recorded_dir} requested={notebook_dir}"
        )

    recorded_comp = str(status.get("competition", "")).strip()
    if recorded_comp and recorded_comp != competition:
        raise RuntimeError(
            "Preflight competition mismatch. "
            f"status={recorded_comp} requested={competition}"
        )

    finished_at = str(status.get("finished_at", "")).strip()
    if not finished_at:
        raise RuntimeError("Preflight status is missing `finished_at` timestamp.")

    try:
        parsed = dt.datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
    except Exception as exc:
        raise RuntimeError(f"Invalid preflight timestamp: {finished_at}") from exc

    age_minutes = (dt.datetime.now(dt.timezone.utc) - parsed).total_seconds() / 60.0
    if max_age_minutes > 0 and age_minutes > max_age_minutes:
        raise RuntimeError(
            "Preflight status is stale. "
            f"age_minutes={age_minutes:.1f} max={max_age_minutes}. "
            "Re-run `kaggle-kernel-preflight-44 --stage all`."
        )


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

    input_ids = [str(x) for x in problems[args.id_col].tolist()]
    submission_df, validation = sanitize_submission(
        submission_df,
        input_ids=input_ids,
        default_answer=int(args.default_answer),
    )
    out = save_submission(submission_df, args.output_csv)
    print(f"Saved submission file: {out}")
    print(
        "Submission validation: "
        f"rows={validation.rows_after} "
        f"duplicates={validation.duplicate_ids} "
        f"invalid_fixed={validation.invalid_answers_fixed} "
        f"missing_filled={validation.missing_ids_filled} "
        f"range_normalized={validation.out_of_range_normalized}"
    )

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
        "model": FORCED_MODEL,
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


def _compute_debug_metrics(
    debug_rows: list[dict[str, object]], merged: pd.DataFrame
) -> dict[str, float]:
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
        f"--model {FORCED_MODEL}",
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
            "model": FORCED_MODEL,
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
        print(
            f"Downloaded and extracted {len(extracted)} files into {Path(args.output_dir).resolve()}"
        )
    else:
        print(f"Downloaded archive into {Path(args.output_dir).resolve()}")


def cmd_kaggle_submit(args: argparse.Namespace) -> None:
    api = KaggleAutomation(args.competition)

    submission_path = Path(args.submission_csv).expanduser()
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    raw_submission = pd.read_csv(submission_path)
    sanitized_df, validation = sanitize_submission(
        raw_submission,
        default_answer=int(args.default_answer),
    )
    safe_submission_path = submission_path
    if validation.changed:
        safe_submission_path = submission_path.with_name(f"{submission_path.stem}.validated.csv")
        save_submission(sanitized_df, safe_submission_path)
        print(
            "Sanitized submission before upload: "
            f"{safe_submission_path} "
            f"(duplicates={validation.duplicate_ids}, "
            f"invalid_fixed={validation.invalid_answers_fixed}, "
            f"range_normalized={validation.out_of_range_normalized})"
        )

    message = _build_submission_message(args.message, prefix="AIMO3 automated CSV submission")

    response = api.submit(str(safe_submission_path), message=message)
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


def cmd_kaggle_submit_code(args: argparse.Namespace) -> None:
    kernel_ref = _resolve_kernel_ref(args.kernel, args.kernel_dir)
    api = KaggleAutomation(args.competition, default_username=_kernel_default_username(kernel_ref))

    if not _check_daily_submission_quota(
        api,
        max_submissions_per_day=args.max_submissions_per_day,
        scan_limit=args.daily_limit_scan,
        skip_if_daily_limit_reached=args.skip_if_daily_limit_reached,
    ):
        return

    kernel_version = (
        int(args.kernel_version) if args.kernel_version and args.kernel_version > 0 else None
    )
    message = _build_submission_message(args.message, prefix="AIMO3 automated code submission")
    response = api.submit_code(
        kernel=kernel_ref,
        output_file_name=args.output_file_name,
        message=message,
        kernel_version=kernel_version,
    )
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


def cmd_kaggle_kernel_pipeline(args: argparse.Namespace) -> None:
    kernel_root, metadata = _read_kernel_metadata(args.kernel_dir)
    kernel_ref = _resolve_kernel_ref(args.kernel, args.kernel_dir)
    if args.strict_kernel_metadata:
        _assert_kernel_metadata_safe(
            metadata=metadata,
            kernel_root=kernel_root,
            competition=args.competition,
        )

    api = KaggleAutomation(args.competition, default_username=_kernel_default_username(kernel_ref))

    pushed = api.push_kernel(
        kernel_root,
        timeout_sec=int(args.push_timeout_sec) if args.push_timeout_sec > 0 else None,
    )
    effective_kernel = pushed.kernel_ref or kernel_ref
    print(
        "Kernel pushed: "
        f"kernel={effective_kernel} version={pushed.version_number} url={pushed.url or 'n/a'}"
    )

    kernel_state = api.wait_for_kernel(
        effective_kernel,
        poll_seconds=args.kernel_poll_seconds,
        timeout_minutes=args.kernel_timeout_minutes,
    )
    if kernel_state.status != "complete":
        try:
            failure_outputs = api.download_kernel_output(
                effective_kernel,
                args.output_dir,
                overwrite_dir=args.overwrite_output_dir,
            )
            print(
                "Kernel failed; downloaded failure artifacts: "
                f"{len(failure_outputs)} files at {Path(args.output_dir).resolve()}"
            )
        except Exception as exc:
            print(f"Kernel failed and failure-artifact download also failed: {exc}")
        raise RuntimeError(
            "Kernel run did not complete successfully: "
            f"status={kernel_state.status} failure={kernel_state.failure_message}"
        )

    output_files = api.download_kernel_output(
        effective_kernel,
        args.output_dir,
        overwrite_dir=args.overwrite_output_dir,
    )
    print(f"Downloaded {len(output_files)} output files to {Path(args.output_dir).resolve()}")

    required_path = Path(args.output_dir).expanduser() / args.required_output_file
    if not required_path.exists():
        raise FileNotFoundError(
            "Required output file missing after kernel run: "
            f"{required_path}. The notebook version is not submittable."
        )

    _, validation = validate_submission_file(
        required_path,
        default_answer=int(args.default_answer),
        strict=args.strict_output_validation,
    )
    print(
        "Output validation: "
        f"format={validation.file_format} rows={validation.rows} "
        f"columns_ok={validation.columns_ok} changed={validation.changed_by_sanitization}"
    )
    if validation.rows <= 0:
        raise ValueError("Required output file is empty.")

    _validate_kernel_runtime_health(
        output_dir=args.output_dir,
        strict=bool(args.strict_output_validation),
    )

    result_payload: dict[str, object] = {
        "kernel": effective_kernel,
        "kernel_version": pushed.version_number,
        "output_dir": str(Path(args.output_dir).expanduser()),
        "required_output_file": args.required_output_file,
        "rows": int(validation.rows),
        "submitted": False,
    }

    if args.no_submit:
        print("Skipping competition submit (--no-submit).")
        return result_payload

    if not _check_daily_submission_quota(
        api,
        max_submissions_per_day=args.max_submissions_per_day,
        scan_limit=args.daily_limit_scan,
        skip_if_daily_limit_reached=args.skip_if_daily_limit_reached,
    ):
        return

    kernel_version = pushed.version_number
    if kernel_version is None and args.kernel_version and args.kernel_version > 0:
        kernel_version = int(args.kernel_version)

    message = _build_submission_message(args.message, prefix="AIMO3 kernel pipeline submission")
    response = api.submit_code(
        kernel=effective_kernel,
        output_file_name=args.required_output_file,
        message=message,
        kernel_version=kernel_version,
    )
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

    result_payload["submitted"] = True
    result_payload["kernel_version"] = kernel_version
    return result_payload


def cmd_kaggle_kernel_preflight_44(args: argparse.Namespace) -> None:
    stage = str(args.stage).strip().lower()
    status_payload: dict[str, object] = {
        "command": "kaggle-kernel-preflight-44",
        "ok": False,
        "stage": stage,
        "competition": args.competition,
        "notebook_dir": args.notebook_dir,
        "required_output_file": args.required_output_file,
        "started_at": _utc_now_iso(),
        "local": {"ok": False, "skipped": True},
        "kaggle": {"ok": False, "skipped": True},
    }

    try:
        if stage in {"local", "all"}:
            local_result = _run_local_preflight_44(modules=list(args.local_test_modules))
            status_payload["local"] = {"ok": True, "skipped": False, **local_result}

        if stage in {"kaggle", "all"}:
            kernel_args = argparse.Namespace(
                competition=args.competition,
                kernel_dir=args.notebook_dir,
                kernel=args.kernel,
                strict_kernel_metadata=True,
                push_timeout_sec=args.push_timeout_sec,
                kernel_poll_seconds=args.kernel_poll_seconds,
                kernel_timeout_minutes=args.kernel_timeout_minutes,
                output_dir=args.output_dir,
                overwrite_output_dir=True,
                required_output_file=args.required_output_file,
                strict_output_validation=args.strict_runtime_health,
                default_answer=0,
                kernel_version=0,
                no_submit=True,
                message=args.message,
                max_submissions_per_day=1,
                daily_limit_scan=100,
                skip_if_daily_limit_reached=True,
                wait=False,
                poll_seconds=20,
                timeout_minutes=90,
            )
            kaggle_result = cmd_kaggle_kernel_pipeline(kernel_args) or {}
            status_payload["kaggle"] = {"ok": True, "skipped": False, **kaggle_result}

        status_payload["ok"] = True
        status_payload["finished_at"] = _utc_now_iso()
        status_path = _write_preflight_status(args.status_file, status_payload)
        print(f"Preflight succeeded. Status written to: {status_path}")

    except Exception as exc:
        status_payload["ok"] = False
        status_payload["error"] = f"{type(exc).__name__}: {exc}"
        status_payload["finished_at"] = _utc_now_iso()
        status_path = _write_preflight_status(args.status_file, status_payload)
        print(f"Preflight failed. Status written to: {status_path}")
        raise


def cmd_kaggle_submit_44(args: argparse.Namespace) -> None:
    status = _load_preflight_status(args.status_file)
    _assert_preflight_ready_for_submit(
        status=status,
        notebook_dir=args.notebook_dir,
        competition=args.competition,
        max_age_minutes=args.max_preflight_age_minutes,
    )

    kernel_version_from_status = 0
    kaggle_stage = status.get("kaggle", {})
    if isinstance(kaggle_stage, dict):
        try:
            kernel_version_from_status = int(kaggle_stage.get("kernel_version", 0) or 0)
        except Exception:
            kernel_version_from_status = 0

    submit_args = argparse.Namespace(
        competition=args.competition,
        kernel=args.kernel,
        kernel_dir=args.notebook_dir,
        kernel_version=kernel_version_from_status,
        output_file_name=args.output_file_name,
        message=args.message,
        max_submissions_per_day=args.max_submissions_per_day,
        daily_limit_scan=args.daily_limit_scan,
        skip_if_daily_limit_reached=args.skip_if_daily_limit_reached,
        wait=args.wait,
        poll_seconds=args.poll_seconds,
        timeout_minutes=args.timeout_minutes,
    )
    cmd_kaggle_submit_code(submit_args)


def cmd_kaggle_pipeline(args: argparse.Namespace) -> None:
    solve_args = argparse.Namespace(**vars(args))
    cmd_solve(solve_args)

    submit_args = argparse.Namespace(
        competition=args.competition,
        submission_csv=args.output_csv,
        default_answer=args.default_answer,
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
    ksub.add_argument("--default-answer", type=int, default=0)
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

    kcode = sub.add_parser(
        "kaggle-submit-code",
        help="Submit a notebook/kernel version output file (code competition flow)",
    )
    kcode.add_argument("--competition", default="ai-mathematical-olympiad-progress-prize-3")
    kcode.add_argument(
        "--kernel",
        default="",
        help="Kernel ref <owner>/<slug>. If omitted, use kernel-metadata.json id.",
    )
    kcode.add_argument(
        "--kernel-dir",
        default="kaggle_kernel_submission",
        help="Directory containing kernel-metadata.json for fallback kernel id.",
    )
    kcode.add_argument("--kernel-version", type=int, default=0)
    kcode.add_argument("--output-file-name", default="submission.parquet")
    kcode.add_argument("--message", default="")
    kcode.add_argument("--max-submissions-per-day", type=int, default=1)
    kcode.add_argument("--daily-limit-scan", type=int, default=100)
    kcode.add_argument("--skip-if-daily-limit-reached", action="store_true", default=True)
    kcode.add_argument(
        "--no-skip-if-daily-limit-reached",
        action="store_false",
        dest="skip_if_daily_limit_reached",
    )
    kcode.add_argument("--wait", action="store_true")
    kcode.add_argument("--poll-seconds", type=int, default=20)
    kcode.add_argument("--timeout-minutes", type=int, default=60)
    kcode.set_defaults(func=cmd_kaggle_submit_code)

    kkernel = sub.add_parser(
        "kaggle-kernel-pipeline",
        help="Hardened pipeline: push kernel -> wait run -> validate submission.parquet -> submit",
    )
    kkernel.add_argument("--competition", default="ai-mathematical-olympiad-progress-prize-3")
    kkernel.add_argument("--kernel-dir", default="kaggle_kernel_submission")
    kkernel.add_argument(
        "--kernel",
        default="",
        help="Kernel ref <owner>/<slug>. If omitted, use kernel-metadata.json id.",
    )
    kkernel.add_argument(
        "--strict-kernel-metadata",
        action="store_true",
        default=True,
        help="Fail if kernel metadata is unsafe for this competition.",
    )
    kkernel.add_argument(
        "--no-strict-kernel-metadata",
        action="store_false",
        dest="strict_kernel_metadata",
    )
    kkernel.add_argument("--push-timeout-sec", type=int, default=0)
    kkernel.add_argument("--kernel-poll-seconds", type=int, default=20)
    kkernel.add_argument("--kernel-timeout-minutes", type=int, default=180)
    kkernel.add_argument("--output-dir", default="artifacts/kaggle_kernel_output_latest")
    kkernel.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        default=True,
        help="Delete output-dir before downloading kernel outputs.",
    )
    kkernel.add_argument(
        "--no-overwrite-output-dir",
        action="store_false",
        dest="overwrite_output_dir",
    )
    kkernel.add_argument("--required-output-file", default="submission.parquet")
    kkernel.add_argument("--strict-output-validation", action="store_true", default=True)
    kkernel.add_argument(
        "--no-strict-output-validation",
        action="store_false",
        dest="strict_output_validation",
    )
    kkernel.add_argument("--default-answer", type=int, default=0)
    kkernel.add_argument("--kernel-version", type=int, default=0)
    kkernel.add_argument("--no-submit", action="store_true")
    kkernel.add_argument("--message", default="")
    kkernel.add_argument("--max-submissions-per-day", type=int, default=1)
    kkernel.add_argument("--daily-limit-scan", type=int, default=100)
    kkernel.add_argument("--skip-if-daily-limit-reached", action="store_true", default=True)
    kkernel.add_argument(
        "--no-skip-if-daily-limit-reached",
        action="store_false",
        dest="skip_if_daily_limit_reached",
    )
    kkernel.add_argument("--wait", action="store_true")
    kkernel.add_argument("--poll-seconds", type=int, default=20)
    kkernel.add_argument("--timeout-minutes", type=int, default=90)
    kkernel.set_defaults(func=cmd_kaggle_kernel_pipeline)

    kpre44 = sub.add_parser(
        "kaggle-kernel-preflight-44",
        help="Staged preflight for the hardened 44/50 notebook (local checks + Kaggle no-submit validation).",
    )
    kpre44.add_argument("--competition", default="ai-mathematical-olympiad-progress-prize-3")
    kpre44.add_argument(
        "--notebook-dir",
        default="kaggle_kernel_submission_44_50",
        help="Notebook directory containing kernel-metadata.json and .ipynb for the 44/50 kernel.",
    )
    kpre44.add_argument(
        "--kernel",
        default="",
        help="Kernel ref <owner>/<slug>. If omitted, use notebook-dir/kernel-metadata.json id.",
    )
    kpre44.add_argument("--stage", choices=["local", "kaggle", "all"], default="all")
    kpre44.add_argument("--push-timeout-sec", type=int, default=0)
    kpre44.add_argument("--kernel-poll-seconds", type=int, default=20)
    kpre44.add_argument("--kernel-timeout-minutes", type=int, default=180)
    kpre44.add_argument("--output-dir", default="artifacts/kaggle_kernel_output_44_latest")
    kpre44.add_argument("--required-output-file", default="submission.parquet")
    kpre44.add_argument("--strict-runtime-health", action="store_true", default=True)
    kpre44.add_argument(
        "--no-strict-runtime-health",
        action="store_false",
        dest="strict_runtime_health",
    )
    kpre44.add_argument("--message", default="")
    kpre44.add_argument("--status-file", default="artifacts/preflight_44_status.json")
    kpre44.add_argument(
        "--local-test-modules",
        nargs="+",
        default=[
            "tests.test_notebook_44_50_contract",
            "tests.test_notebook_44_50_local_gate",
            "tests.test_pipeline",
            "tests.test_cli",
        ],
        help="Python unittest modules executed in local stage.",
    )
    kpre44.set_defaults(func=cmd_kaggle_kernel_preflight_44)

    ksubmit44 = sub.add_parser(
        "kaggle-submit-44",
        help="Submit 44/50 notebook output only if staged preflight status is present and fresh.",
    )
    ksubmit44.add_argument("--competition", default="ai-mathematical-olympiad-progress-prize-3")
    ksubmit44.add_argument(
        "--notebook-dir",
        default="kaggle_kernel_submission_44_50",
        help="Notebook directory used by preflight.",
    )
    ksubmit44.add_argument(
        "--kernel",
        default="",
        help="Kernel ref <owner>/<slug>. If omitted, use notebook-dir/kernel-metadata.json id.",
    )
    ksubmit44.add_argument("--output-file-name", default="submission.parquet")
    ksubmit44.add_argument("--message", default="")
    ksubmit44.add_argument("--status-file", default="artifacts/preflight_44_status.json")
    ksubmit44.add_argument("--max-preflight-age-minutes", type=int, default=240)
    ksubmit44.add_argument("--max-submissions-per-day", type=int, default=1)
    ksubmit44.add_argument("--daily-limit-scan", type=int, default=100)
    ksubmit44.add_argument("--skip-if-daily-limit-reached", action="store_true", default=True)
    ksubmit44.add_argument(
        "--no-skip-if-daily-limit-reached",
        action="store_false",
        dest="skip_if_daily_limit_reached",
    )
    ksubmit44.add_argument("--wait", action="store_true")
    ksubmit44.add_argument("--poll-seconds", type=int, default=20)
    ksubmit44.add_argument("--timeout-minutes", type=int, default=60)
    ksubmit44.set_defaults(func=cmd_kaggle_submit_44)

    return parser


def main() -> None:
    _load_dotenv_if_present()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
