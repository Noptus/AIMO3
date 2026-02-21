"""Composable AIMO3 solver with tool-integrated reasoning."""

from __future__ import annotations

import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from .client import ChatClient
from .mini_solvers import run_mini_solvers
from .parsing import (
    extract_python_blocks,
    parse_answer_from_text,
    parse_integer_from_stdout,
    parse_modulus,
    parse_structured_result,
)
from .prompts import (
    ProblemProfile,
    build_adversarial_probe_prompt,
    build_agent_followup_prompt,
    build_consistency_audit_prompt,
    build_fallback_guess_prompt,
    build_final_extractor_prompt,
    build_forced_code_check_prompt,
    build_geometry_recheck_prompt,
    build_prompt,
    build_repair_prompt,
    build_selector_prompt,
    build_small_answer_guard_prompt,
    build_verifier_prompt,
    estimate_problem_profile,
)
from .sandbox import SandboxPolicy, execute_python


@dataclass(frozen=True)
class SolverConfig:
    attempts: int = 8
    temperatures: tuple[float, ...] = (0.15, 0.25, 0.35, 0.45)
    max_tokens: int = 1024
    min_consensus: int = 3
    early_stop_ratio: float = 0.8
    early_stop_attempt: int = 4
    max_code_blocks_per_attempt: int = 2
    agentic_tool_rounds: int = 1
    agentic_observation_chars: int = 1200
    agentic_stateful_python: bool = True
    agentic_state_chars: int = 20_000
    parallel_attempt_workers: int = 1
    parallel_code_workers: int = 1
    stage_time_reserve_sec: int = 0
    force_tool_round_for_unverified: bool = False
    per_problem_time_sec: int = 0
    min_time_for_attempt_sec: int = 20
    min_time_for_stage_sec: int = 8
    force_full_problem_time: bool = False
    default_answer: int = 0

    # Hard-problem behavior.
    adaptive_complexity: bool = True
    hard_mode: bool = False
    hard_attempts: int = 12
    hard_max_tokens: int = 4096

    # Second-pass correction and arbitration.
    repair_passes: int = 0
    final_extractor_passes: int = 1
    final_extractor_max_tokens: int = 256
    verification_attempts: int = 0
    verification_top_k: int = 3
    verification_temperature: float = 0.1

    # Contradiction-focused arbitration.
    consistency_audit_attempts: int = 0
    consistency_audit_top_k: int = 4
    consistency_audit_temperature: float = 0.09

    # Adversarial contradiction probe to challenge fragile consensus.
    adversarial_probe_attempts: int = 0
    adversarial_probe_top_k: int = 4
    adversarial_probe_temperature: float = 0.16

    # Geometry-focused recheck pass.
    geometry_recheck_attempts: int = 0
    geometry_top_k: int = 4
    geometry_recheck_temperature: float = 0.08

    # Guard stage to reduce unsupported trivial (0/1) collapse.
    small_answer_guard_attempts: int = 1
    small_answer_guard_top_k: int = 3
    small_answer_guard_temperature: float = 0.12

    # Final fallback stage if all extraction paths failed.
    fallback_guess_attempts: int = 1
    fallback_guess_temperature: float = 0.15

    # GenSelect-style selector pass.
    selector_attempts: int = 0
    selector_top_k: int = 4
    selector_temperature: float = 0.05

    # If too few valid answers were produced, run extra rescue attempts.
    sparse_recovery_attempts: int = 2
    sparse_recovery_temperature: float = 0.1

    # Uncertainty-triggered escalation before arbitration stages.
    escalation_attempts: int = 0
    escalation_temperature: float = 0.55
    escalation_trigger_ratio: float = 0.72
    escalation_min_valid: int = 3

    # Structured output + mandatory executable validation.
    mandatory_code_attempts: int = 0

    # Deterministic mini-solvers for trivial/high-confidence sub-cases.
    mini_solver_enabled: bool = True
    mini_solver_min_confidence: float = 0.95

    # Additional strictness for 0/1 collapse.
    strict_zero_one_policy: bool = True


@dataclass(frozen=True)
class SolveBudget:
    attempts: int
    max_tokens: int
    allow_early_stop: bool


@dataclass
class Candidate:
    attempt: int
    temperature: float
    response_text: str
    answer: int | None
    answer_source: str
    modulus: int | None
    category: str
    archetype: str
    complexity: str
    stage: str = "initial"
    code_blocks: int = 0
    code_verified: bool = False
    agent_rounds: int = 0
    repair_used: bool = False
    extractor_used: bool = False
    code_answers: list[int] = field(default_factory=list)
    sandbox_errors: list[str] = field(default_factory=list)
    generation_error: str | None = None
    answer_in_problem: bool = False
    method: str = ""
    independent_check_passed: bool = False
    missing_forced_code_check: bool = False

    @property
    def score(self) -> float:
        if self.generation_error:
            return 0.01

        base = 0.2
        if self.answer is not None:
            base += 1.0

        if self.answer_source in {"final_answer_tag", "boxed"}:
            base += 0.44
        elif self.answer_source == "structured_json":
            base += 0.78
        elif self.answer_source == "verifier":
            base += 0.8
        elif self.answer_source == "consistency_audit":
            base += 0.88
        elif self.answer_source == "adversarial_probe":
            base += 0.76
        elif self.answer_source == "geometry_recheck":
            base += 0.9
        elif self.answer_source == "small_guard":
            base += 0.85
        elif self.answer_source == "fallback_guess":
            base += 0.45
        elif self.answer_source == "fallback_guess_heuristic":
            base += 0.18
        elif self.answer_source == "selector":
            base += 1.0
        elif self.answer_source == "mini_solver":
            base += 1.25

        if self.code_verified:
            base += 2.2
        elif self.code_answers:
            base += 0.5
        if self.independent_check_passed:
            base += 0.55

        if self.repair_used:
            base += 0.2

        if self.extractor_used:
            base += 0.05

        if self.agent_rounds:
            base += 0.08 * min(3, self.agent_rounds)

        if self.complexity == "hard":
            base += 0.1

        if self.answer is not None and 0 <= self.answer <= 9:
            base -= 0.18

        if self.answer_in_problem:
            base -= 0.12

        if self.missing_forced_code_check:
            base -= 0.85

        if (
            self.answer_source in {"final_answer_tag", "boxed", "answer_line"}
            and not self.code_verified
            and not self.independent_check_passed
        ):
            base -= 0.35

        base -= min(0.5, 0.2 * len(self.sandbox_errors))
        return max(base, 0.01)


@dataclass
class SolveResult:
    problem_id: Any
    predicted_answer: int
    modulus: int | None
    profile: ProblemProfile
    candidates: list[Candidate]

    @property
    def debug_summary(self) -> dict[str, Any]:
        votes = Counter(c.answer for c in self.candidates if c.answer is not None)
        return {
            "predicted_answer": self.predicted_answer,
            "modulus": self.modulus,
            "category": self.profile.category,
            "archetype": self.profile.archetype,
            "complexity": self.profile.complexity,
            "complexity_score": self.profile.score,
            "candidate_count": len(self.candidates),
            "top_votes": votes.most_common(5),
            "verified_candidates": sum(1 for c in self.candidates if c.code_verified),
            "independent_check_candidates": sum(
                1 for c in self.candidates if c.independent_check_passed
            ),
            "agentic_candidates": sum(1 for c in self.candidates if c.agent_rounds > 0),
            "max_agent_rounds": max((c.agent_rounds for c in self.candidates), default=0),
            "repair_candidates": sum(1 for c in self.candidates if c.repair_used),
            "extractor_candidates": sum(1 for c in self.candidates if c.extractor_used),
            "mini_solver_candidates": sum(1 for c in self.candidates if c.stage == "mini_solver"),
            "forced_code_missing_candidates": sum(
                1 for c in self.candidates if c.missing_forced_code_check
            ),
            "verifier_candidates": sum(1 for c in self.candidates if c.stage == "verifier"),
            "consistency_audit_candidates": sum(
                1 for c in self.candidates if c.stage == "consistency_audit"
            ),
            "adversarial_probe_candidates": sum(
                1 for c in self.candidates if c.stage == "adversarial_probe"
            ),
            "geometry_recheck_candidates": sum(
                1 for c in self.candidates if c.stage == "geometry_recheck"
            ),
            "small_guard_candidates": sum(1 for c in self.candidates if c.stage == "small_guard"),
            "fallback_guess_candidates": sum(
                1 for c in self.candidates if c.stage == "fallback_guess"
            ),
            "selector_candidates": sum(1 for c in self.candidates if c.stage == "selector"),
            "escalation_candidates": sum(1 for c in self.candidates if c.stage == "escalation"),
            "problem_number_hits": sum(1 for c in self.candidates if c.answer_in_problem),
        }


class AIMO3Solver:
    """LLM orchestrator for the AIMO3 competition."""

    def __init__(
        self,
        client: ChatClient,
        *,
        config: SolverConfig | None = None,
        sandbox_policy: SandboxPolicy | None = None,
    ) -> None:
        self.client = client
        self.config = config or SolverConfig()
        self.sandbox_policy = sandbox_policy or SandboxPolicy()
        self._sandbox_cache: dict[str, Any] = {}

    def solve(self, problem_text: str, problem_id: Any = None) -> SolveResult:
        # Per-problem sandbox cache for repeated code blocks across attempts/stages.
        self._sandbox_cache = {}
        modulus = parse_modulus(problem_text)
        profile = estimate_problem_profile(problem_text)
        budget = self._build_budget(profile)
        problem_numbers = self._extract_problem_numbers(problem_text)
        deadline = self._problem_deadline()

        candidates: list[Candidate] = []
        if self.config.mini_solver_enabled:
            candidates.extend(
                self._run_mini_solver_candidates(
                    problem_text=problem_text,
                    modulus=modulus,
                    profile=profile,
                    problem_numbers=problem_numbers,
                )
            )

        candidates.extend(
            self._run_initial_attempts(
                problem_text=problem_text,
                modulus=modulus,
                profile=profile,
                budget=budget,
                problem_numbers=problem_numbers,
                deadline=deadline,
            )
        )

        if self._has_time(deadline, self.config.min_time_for_attempt_sec):
            candidates.extend(
                self._run_escalation_attempts(
                    problem_text=problem_text,
                    modulus=modulus,
                    profile=profile,
                    candidates=candidates,
                    max_tokens=min(self.config.hard_max_tokens, budget.max_tokens + 512),
                    problem_numbers=problem_numbers,
                    deadline=deadline,
                )
            )

        if self._has_time(deadline, self.config.min_time_for_stage_sec):
            candidates.extend(
                self._run_verification(
                    problem_text=problem_text,
                    modulus=modulus,
                    profile=profile,
                    candidates=candidates,
                    max_tokens=max(256, min(640, budget.max_tokens // 2)),
                    problem_numbers=problem_numbers,
                    deadline=deadline,
                )
            )

        if self._has_time(deadline, self.config.min_time_for_stage_sec):
            candidates.extend(
                self._run_sparse_recovery(
                    problem_text=problem_text,
                    modulus=modulus,
                    profile=profile,
                    candidates=candidates,
                    max_tokens=budget.max_tokens,
                    problem_numbers=problem_numbers,
                    deadline=deadline,
                )
            )

        if self._has_time(deadline, self.config.min_time_for_stage_sec):
            candidates.extend(
                self._run_consistency_audit(
                    problem_text=problem_text,
                    modulus=modulus,
                    profile=profile,
                    candidates=candidates,
                    max_tokens=max(320, min(896, budget.max_tokens // 2)),
                    problem_numbers=problem_numbers,
                    deadline=deadline,
                )
            )

        if self._has_time(deadline, self.config.min_time_for_stage_sec):
            candidates.extend(
                self._run_adversarial_probe(
                    problem_text=problem_text,
                    modulus=modulus,
                    profile=profile,
                    candidates=candidates,
                    max_tokens=max(320, min(896, budget.max_tokens // 2)),
                    problem_numbers=problem_numbers,
                    deadline=deadline,
                )
            )

        if self._has_time(deadline, self.config.min_time_for_stage_sec):
            candidates.extend(
                self._run_geometry_recheck(
                    problem_text=problem_text,
                    modulus=modulus,
                    profile=profile,
                    candidates=candidates,
                    max_tokens=max(320, min(896, budget.max_tokens // 2)),
                    problem_numbers=problem_numbers,
                    deadline=deadline,
                )
            )

        if self._has_time(deadline, self.config.min_time_for_stage_sec):
            candidates.extend(
                self._run_small_answer_guard(
                    problem_text=problem_text,
                    modulus=modulus,
                    profile=profile,
                    candidates=candidates,
                    max_tokens=max(320, min(896, budget.max_tokens // 2)),
                    problem_numbers=problem_numbers,
                    deadline=deadline,
                )
            )

        if self._has_time(deadline, self.config.min_time_for_stage_sec):
            candidates.extend(
                self._run_selector(
                    problem_text=problem_text,
                    modulus=modulus,
                    profile=profile,
                    candidates=candidates,
                    max_tokens=max(384, min(1024, budget.max_tokens // 2)),
                    problem_numbers=problem_numbers,
                    deadline=deadline,
                )
            )

        if self._has_time(deadline, 2):
            candidates.extend(
                self._run_fallback_guess(
                    problem_text=problem_text,
                    modulus=modulus,
                    profile=profile,
                    candidates=candidates,
                    max_tokens=max(192, min(512, budget.max_tokens // 3)),
                    problem_numbers=problem_numbers,
                    deadline=deadline,
                )
            )

        predicted = self._aggregate(candidates, modulus, problem_numbers)

        return SolveResult(
            problem_id=problem_id,
            predicted_answer=predicted,
            modulus=modulus,
            profile=profile,
            candidates=candidates,
        )

    def _run_mini_solver_candidates(
        self,
        *,
        problem_text: str,
        modulus: int | None,
        profile: ProblemProfile,
        problem_numbers: set[int],
    ) -> list[Candidate]:
        staged: list[Candidate] = []
        for idx, solved in enumerate(run_mini_solvers(problem_text, modulus=modulus)):
            if solved.confidence < float(self.config.mini_solver_min_confidence):
                continue
            staged.append(
                Candidate(
                    attempt=-10_000 - idx,
                    temperature=0.0,
                    response_text=solved.reason,
                    answer=solved.answer,
                    answer_source="mini_solver",
                    modulus=modulus,
                    category=profile.category,
                    archetype=profile.archetype,
                    complexity=profile.complexity,
                    stage="mini_solver",
                    code_blocks=1,
                    code_verified=True,
                    method=solved.method,
                    independent_check_passed=bool(solved.independent_check_passed),
                    answer_in_problem=bool(solved.answer in problem_numbers),
                )
            )
        return staged

    def _build_budget(self, profile: ProblemProfile) -> SolveBudget:
        attempts = self.config.attempts
        max_tokens = self.config.max_tokens
        allow_early_stop = not self.config.force_full_problem_time

        is_hard = self.config.hard_mode or (
            self.config.adaptive_complexity and profile.complexity == "hard"
        )

        if is_hard:
            attempts = max(attempts, self.config.hard_attempts)
            max_tokens = max(max_tokens, self.config.hard_max_tokens)
            # For hard problems, avoid stopping too early on weak consensus.
            allow_early_stop = False
        elif self.config.adaptive_complexity and profile.complexity == "medium":
            attempts = max(attempts, min(self.config.hard_attempts, self.config.attempts + 2))
            max_tokens = max(
                max_tokens, min(self.config.hard_max_tokens, self.config.max_tokens + 256)
            )

        if self.config.adaptive_complexity and profile.category == "geometry":
            attempts = max(attempts, min(self.config.hard_attempts, self.config.attempts + 2))
            max_tokens = max(
                max_tokens, min(self.config.hard_max_tokens, self.config.max_tokens + 512)
            )
            if profile.complexity != "easy":
                allow_early_stop = False

        return SolveBudget(
            attempts=attempts,
            max_tokens=max_tokens,
            allow_early_stop=allow_early_stop,
        )

    def _problem_deadline(self) -> float | None:
        if self.config.per_problem_time_sec <= 0:
            return None
        return time.monotonic() + float(self.config.per_problem_time_sec)

    def _remaining_time(self, deadline: float | None) -> float:
        if deadline is None:
            return float("inf")
        return max(0.0, deadline - time.monotonic())

    def _has_time(self, deadline: float | None, required_sec: float = 0.0) -> bool:
        return self._remaining_time(deadline) > max(0.0, required_sec)

    def _downstream_stages_enabled(self) -> bool:
        return any(
            [
                self.config.verification_attempts > 0,
                self.config.sparse_recovery_attempts > 0,
                self.config.consistency_audit_attempts > 0,
                self.config.adversarial_probe_attempts > 0,
                self.config.geometry_recheck_attempts > 0,
                self.config.small_answer_guard_attempts > 0,
                self.config.selector_attempts > 0,
                self.config.fallback_guess_attempts > 0,
                self.config.escalation_attempts > 0,
            ]
        )

    def _active_downstream_stage_count(self) -> int:
        count = 0
        if self.config.verification_attempts > 0:
            count += 1
        if self.config.sparse_recovery_attempts > 0:
            count += 1
        if self.config.consistency_audit_attempts > 0:
            count += 1
        if self.config.adversarial_probe_attempts > 0:
            count += 1
        if self.config.geometry_recheck_attempts > 0:
            count += 1
        if self.config.small_answer_guard_attempts > 0:
            count += 1
        if self.config.selector_attempts > 0:
            count += 1
        if self.config.fallback_guess_attempts > 0:
            count += 1
        if self.config.escalation_attempts > 0:
            count += 1
        return count

    def _should_hold_stage_reserve(
        self,
        deadline: float | None,
        *,
        launching_new_attempt: bool = False,
    ) -> bool:
        if deadline is None:
            return False
        if self.config.stage_time_reserve_sec <= 0:
            return False
        if not self._downstream_stages_enabled():
            return False
        reserve = float(self.config.stage_time_reserve_sec)
        # When launching another expensive attempt, hold extra cushion so downstream
        # arbitration stages still get wall-clock time.
        if launching_new_attempt:
            reserve += float(max(0, self.config.min_time_for_attempt_sec))
            # Also preserve a minimum downstream-stage floor for at least a few
            # arbitration stages (verifier/selector/guards) instead of starving them.
            stage_floor = min(2, self._active_downstream_stage_count())
            reserve += float(max(0, self.config.min_time_for_stage_sec)) * float(stage_floor)

        # Avoid pathological over-reserving when per-problem budget is modest.
        if self.config.per_problem_time_sec >= 30:
            reserve_cap = max(0.0, float(self.config.per_problem_time_sec) * 0.58)
            reserve = min(reserve, reserve_cap)
        return self._remaining_time(deadline) <= reserve

    def _needs_escalation(self, candidates: list[Candidate]) -> bool:
        if self.config.escalation_attempts <= 0:
            return False

        valid = [c for c in candidates if c.answer is not None]
        if len(valid) < self.config.escalation_min_valid:
            return True

        answers = [c.answer for c in valid if c.answer is not None]
        if not answers:
            return True

        _, top_count = Counter(answers).most_common(1)[0]
        top_ratio = top_count / max(1, len(answers))
        if top_ratio < self.config.escalation_trigger_ratio:
            return True

        top_answer = Counter(answers).most_common(1)[0][0]
        strong_top_evidence = any(
            c.answer == top_answer
            and (
                c.code_verified
                or c.stage
                in {
                    "verifier",
                    "consistency_audit",
                    "adversarial_probe",
                    "geometry_recheck",
                    "selector",
                }
            )
            for c in valid
        )
        if top_answer in {0, 1} and not strong_top_evidence:
            return True

        return False

    def _run_initial_attempts(
        self,
        *,
        problem_text: str,
        modulus: int | None,
        profile: ProblemProfile,
        budget: SolveBudget,
        problem_numbers: set[int],
        deadline: float | None,
    ) -> list[Candidate]:
        def should_hold_after_minimum(candidates_so_far: list[Candidate]) -> bool:
            # On realistic competition budgets, require at least two initial attempts
            # before stage-reserve throttling kicks in.
            if self.config.per_problem_time_sec >= 30:
                initial_so_far = sum(1 for c in candidates_so_far if c.stage == "initial")
                if initial_so_far < 2:
                    return False
            return True

        workers = max(1, int(self.config.parallel_attempt_workers))
        if workers <= 1:
            candidates: list[Candidate] = []
            for attempt in range(budget.attempts):
                if not self._has_time(deadline, self.config.min_time_for_attempt_sec):
                    break
                if (
                    candidates
                    and self._should_hold_stage_reserve(
                        deadline,
                        launching_new_attempt=True,
                    )
                    and should_hold_after_minimum(candidates)
                ):
                    break
                temp = self.config.temperatures[attempt % len(self.config.temperatures)]
                force_code_first = attempt < max(0, int(self.config.mandatory_code_attempts))
                candidate = self._run_attempt(
                    problem_text=problem_text,
                    modulus=modulus,
                    profile=profile,
                    attempt=attempt,
                    temperature=temp,
                    max_tokens=budget.max_tokens,
                    problem_numbers=problem_numbers,
                    force_code_first=force_code_first,
                    deadline=deadline,
                )
                candidates.append(candidate)
                if (
                    budget.allow_early_stop
                    and not self.config.force_full_problem_time
                    and self._should_early_stop(candidates)
                ):
                    break
            return candidates

        candidates: list[Candidate] = []
        next_attempt = 0
        while next_attempt < budget.attempts:
            if not self._has_time(deadline, self.config.min_time_for_attempt_sec):
                break
            if (
                candidates
                and self._should_hold_stage_reserve(
                    deadline,
                    launching_new_attempt=True,
                )
                and should_hold_after_minimum(candidates)
            ):
                break

            batch_size = min(workers, budget.attempts - next_attempt)
            if (
                deadline is not None
                and not candidates
                and self.config.stage_time_reserve_sec > 0
                and self._downstream_stages_enabled()
            ):
                # Under strict per-problem deadlines, run the first attempt alone so
                # we do not consume stage budget with an initial parallel burst.
                batch_size = 1
            futures = {}
            with ThreadPoolExecutor(max_workers=batch_size) as pool:
                for _ in range(batch_size):
                    if not self._has_time(deadline, self.config.min_time_for_attempt_sec):
                        break
                    if (
                        candidates
                        and self._should_hold_stage_reserve(
                            deadline,
                            launching_new_attempt=True,
                        )
                        and should_hold_after_minimum(candidates)
                    ):
                        break
                    attempt = next_attempt
                    next_attempt += 1
                    temp = self.config.temperatures[attempt % len(self.config.temperatures)]
                    force_code_first = attempt < max(0, int(self.config.mandatory_code_attempts))
                    future = pool.submit(
                        self._run_attempt,
                        problem_text=problem_text,
                        modulus=modulus,
                        profile=profile,
                        attempt=attempt,
                        temperature=temp,
                        max_tokens=budget.max_tokens,
                        problem_numbers=problem_numbers,
                        force_code_first=force_code_first,
                        deadline=deadline,
                    )
                    futures[future] = (attempt, temp)

                for future in as_completed(futures):
                    attempt, temp = futures[future]
                    try:
                        candidate = future.result()
                    except Exception as exc:
                        candidate = Candidate(
                            attempt=attempt,
                            temperature=temp,
                            response_text="",
                            answer=None,
                            answer_source="none",
                            modulus=modulus,
                            category=profile.category,
                            archetype=profile.archetype,
                            complexity=profile.complexity,
                            generation_error=f"parallel_attempt_error: {exc}",
                        )
                    candidates.append(candidate)

            candidates.sort(key=lambda c: c.attempt)
            if (
                budget.allow_early_stop
                and not self.config.force_full_problem_time
                and self._should_early_stop(candidates)
            ):
                break

        return candidates

    def _run_escalation_attempts(
        self,
        *,
        problem_text: str,
        modulus: int | None,
        profile: ProblemProfile,
        candidates: list[Candidate],
        max_tokens: int,
        problem_numbers: set[int],
        deadline: float | None,
    ) -> list[Candidate]:
        if not self._needs_escalation(candidates):
            return []

        staged: list[Candidate] = []
        base_attempt = 30_000 + len(candidates)
        for idx in range(self.config.escalation_attempts):
            if not self._has_time(deadline, self.config.min_time_for_attempt_sec):
                break

            # Temperature ladder broadens exploration while staying bounded.
            temp = min(0.95, self.config.escalation_temperature + 0.08 * idx)
            attempt_candidate = self._run_attempt(
                problem_text=problem_text,
                modulus=modulus,
                profile=profile,
                attempt=base_attempt + idx,
                temperature=temp,
                max_tokens=max_tokens,
                problem_numbers=problem_numbers,
                deadline=deadline,
            )
            attempt_candidate.stage = "escalation"
            staged.append(attempt_candidate)

        return staged

    def _run_attempt(
        self,
        *,
        problem_text: str,
        modulus: int | None,
        profile: ProblemProfile,
        attempt: int,
        temperature: float,
        max_tokens: int,
        problem_numbers: set[int],
        force_code_first: bool = False,
        deadline: float | None = None,
    ) -> Candidate:
        if not self._has_time(deadline, self.config.min_time_for_attempt_sec):
            return Candidate(
                attempt=attempt,
                temperature=temperature,
                response_text="",
                answer=None,
                answer_source="none",
                modulus=modulus,
                category=profile.category,
                archetype=profile.archetype,
                complexity=profile.complexity,
                generation_error="problem_time_budget_exhausted_before_attempt",
            )

        prompt = build_prompt(
            problem_text,
            attempt_index=attempt,
            modulus=modulus,
            profile=profile,
            hard_mode=self.config.hard_mode,
            force_code_first=force_code_first,
        )

        try:
            response_text = self.client.generate(
                system_prompt=prompt.system,
                user_prompt=prompt.user,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            return Candidate(
                attempt=attempt,
                temperature=temperature,
                response_text="",
                answer=None,
                answer_source="none",
                modulus=modulus,
                category=profile.category,
                archetype=profile.archetype,
                complexity=profile.complexity,
                generation_error=str(exc),
            )

        code_answers: list[int] = []
        sandbox_errors: list[str] = []
        code_blocks_count = 0
        python_state = ""
        answer: int | None = None
        answer_source = "none"
        parsed = parse_answer_from_text(response_text, modulus=modulus)
        structured = parse_structured_result(response_text, modulus=modulus)
        method = structured.method
        independent_check_passed = bool(structured.independent_check_passed)
        agent_rounds_used = 0

        def refresh_from_response(text: str) -> tuple[list[int], list[str], int]:
            nonlocal answer, answer_source, code_blocks_count, parsed, python_state
            nonlocal method, independent_check_passed
            parsed = parse_answer_from_text(text, modulus=modulus)
            structured_local = parse_structured_result(text, modulus=modulus)
            if structured_local.method:
                method = structured_local.method
            if structured_local.independent_check_passed:
                independent_check_passed = True
            phase_answer = parsed.answer
            phase_answers, phase_errors, phase_blocks, python_state = self._evaluate_code_blocks(
                text,
                modulus,
                stateful_prefix=python_state,
            )
            code_answers.extend(phase_answers)
            sandbox_errors.extend(phase_errors)
            code_blocks_count += phase_blocks
            if phase_answers:
                independent_check_passed = True

            if phase_answer is not None:
                answer = phase_answer
                answer_source = parsed.source
            elif answer is None and code_answers:
                answer = Counter(code_answers).most_common(1)[0][0]
                answer_source = "code_majority"

            return phase_answers, phase_errors, phase_blocks

        phase_answers, phase_errors, phase_blocks = refresh_from_response(response_text)

        if (
            force_code_first
            and not code_answers
            and self._has_time(deadline, self.config.min_time_for_stage_sec)
        ):
            forced_check_prompt = build_forced_code_check_prompt(
                problem_text,
                previous_response=self._trim_for_prompt(response_text),
                modulus=modulus,
                profile=profile,
            )
            try:
                response_text = self.client.generate(
                    system_prompt=forced_check_prompt.system,
                    user_prompt=forced_check_prompt.user,
                    temperature=min(max(temperature, 0.08), 0.2),
                    max_tokens=max(320, max_tokens // 2),
                )
                agent_rounds_used += 1
                phase_answers, phase_errors, phase_blocks = refresh_from_response(response_text)
            except Exception as exc:
                sandbox_errors.append(f"forced_code_check_generation_error: {exc}")

        for round_idx in range(self.config.agentic_tool_rounds):
            if not self._has_time(deadline, self.config.min_time_for_stage_sec):
                break
            if not self._should_continue_agent_loop(
                answer=answer,
                answer_source=answer_source,
                code_blocks_in_response=phase_blocks,
                has_code_evidence=bool(code_answers),
                complexity=profile.complexity,
            ):
                break

            observation = self._build_agent_observation(
                round_index=round_idx,
                previous_response=response_text,
                code_answers=phase_answers,
                sandbox_errors=phase_errors,
            )
            followup_prompt = build_agent_followup_prompt(
                problem_text,
                previous_response=self._trim_for_prompt(response_text),
                tool_observation=observation,
                modulus=modulus,
                profile=profile,
                round_index=round_idx,
            )

            try:
                response_text = self.client.generate(
                    system_prompt=followup_prompt.system,
                    user_prompt=followup_prompt.user,
                    temperature=min(max(temperature, 0.1), 0.3),
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                sandbox_errors.append(f"agent_generation_error_round_{round_idx + 1}: {exc}")
                break

            agent_rounds_used += 1
            phase_answers, phase_errors, phase_blocks = refresh_from_response(response_text)

        repair_used = False
        extractor_used = False
        for _ in range(self.config.repair_passes):
            if not self._has_time(deadline, self.config.min_time_for_stage_sec):
                break
            if not self._needs_repair(answer, code_answers):
                break

            repair_used = True
            tool_feedback = self._format_tool_feedback(answer, code_answers, sandbox_errors)
            repair_prompt = build_repair_prompt(
                problem_text,
                previous_response=self._trim_for_prompt(response_text),
                tool_feedback=tool_feedback,
                modulus=modulus,
                profile=profile,
            )
            try:
                response_text = self.client.generate(
                    system_prompt=repair_prompt.system,
                    user_prompt=repair_prompt.user,
                    temperature=min(temperature, 0.2),
                    max_tokens=max(256, max_tokens // 2),
                )
            except Exception as exc:
                sandbox_errors.append(f"repair_generation_error: {exc}")
                break

            parsed = parse_answer_from_text(response_text, modulus=modulus)
            answer = parsed.answer
            answer_source = parsed.source
            extra_answers, extra_errors, extra_code_blocks, python_state = (
                self._evaluate_code_blocks(
                    response_text,
                    modulus,
                    stateful_prefix=python_state,
                )
            )
            code_answers.extend(extra_answers)
            sandbox_errors.extend(extra_errors)
            code_blocks_count += extra_code_blocks

            if answer is None and code_answers:
                answer = Counter(code_answers).most_common(1)[0][0]
                answer_source = "code_majority"

        for _ in range(self.config.final_extractor_passes):
            if not self._has_time(deadline, 2):
                break
            if answer is not None:
                break
            extractor_used = True
            extractor_prompt = build_final_extractor_prompt(
                problem_text,
                previous_response=self._trim_for_prompt(response_text),
                modulus=modulus,
                profile=profile,
            )
            try:
                response_text = self.client.generate(
                    system_prompt=extractor_prompt.system,
                    user_prompt=extractor_prompt.user,
                    temperature=min(temperature, 0.15),
                    max_tokens=self.config.final_extractor_max_tokens,
                )
            except Exception as exc:
                sandbox_errors.append(f"extractor_generation_error: {exc}")
                break

            parsed = parse_answer_from_text(response_text, modulus=modulus)
            answer = parsed.answer
            answer_source = parsed.source

        code_verified = bool(answer is not None and code_answers and answer in code_answers)
        missing_forced_code_check = bool(force_code_first and not code_answers)

        return Candidate(
            attempt=attempt,
            temperature=temperature,
            response_text=response_text,
            answer=answer,
            answer_source=answer_source,
            modulus=modulus,
            category=profile.category,
            archetype=profile.archetype,
            complexity=profile.complexity,
            stage="initial",
            code_blocks=code_blocks_count,
            code_verified=code_verified,
            agent_rounds=agent_rounds_used,
            repair_used=repair_used,
            extractor_used=extractor_used,
            code_answers=code_answers,
            sandbox_errors=sandbox_errors,
            answer_in_problem=bool(answer is not None and answer in problem_numbers),
            method=method,
            independent_check_passed=independent_check_passed or code_verified,
            missing_forced_code_check=missing_forced_code_check,
        )

    def _evaluate_code_blocks(
        self,
        response_text: str,
        modulus: int | None,
        *,
        stateful_prefix: str = "",
    ) -> tuple[list[int], list[str], int, str]:
        code_answers: list[int] = []
        sandbox_errors: list[str] = []
        code_blocks = extract_python_blocks(response_text)
        selected_blocks = code_blocks[: self.config.max_code_blocks_per_attempt]
        prefix = stateful_prefix if self.config.agentic_stateful_python else ""

        can_parallel_stateless = (
            not self.config.agentic_stateful_python
            and len(selected_blocks) > 1
            and int(self.config.parallel_code_workers) > 1
        )
        if can_parallel_stateless:
            workers = min(len(selected_blocks), max(1, int(self.config.parallel_code_workers)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(self._execute_python_cached, block) for block in selected_blocks
                ]
                for future in futures:
                    try:
                        run = future.result()
                    except Exception as exc:
                        sandbox_errors.append(f"parallel_code_execution_error: {exc}")
                        continue
                    if run.success:
                        code_answer = parse_integer_from_stdout(run.stdout, modulus=modulus)
                        if code_answer is not None:
                            code_answers.append(code_answer)
                    elif run.error:
                        sandbox_errors.append(run.error)
            return code_answers, sandbox_errors, len(code_blocks), prefix

        for block in selected_blocks:
            executable = block
            if prefix:
                executable = prefix + "\n\n" + block
            run = self._execute_python_cached(executable)
            if run.success:
                code_answer = parse_integer_from_stdout(run.stdout, modulus=modulus)
                if code_answer is not None:
                    code_answers.append(code_answer)
                if self.config.agentic_stateful_python:
                    prefix = (prefix + "\n\n" + block).strip() if prefix else block
                    if len(prefix) > self.config.agentic_state_chars:
                        prefix = prefix[-self.config.agentic_state_chars :]
            elif run.error:
                sandbox_errors.append(run.error)

        return code_answers, sandbox_errors, len(code_blocks), prefix

    def _execute_python_cached(self, executable: str):
        cached = self._sandbox_cache.get(executable)
        if cached is not None:
            return cached
        run = execute_python(executable, policy=self.sandbox_policy)
        if len(self._sandbox_cache) >= 512:
            # Cheap FIFO-ish eviction based on insertion order.
            try:
                first_key = next(iter(self._sandbox_cache))
                self._sandbox_cache.pop(first_key, None)
            except Exception:
                self._sandbox_cache = {}
        self._sandbox_cache[executable] = run
        return run

    def _needs_repair(self, answer: int | None, code_answers: list[int]) -> bool:
        if answer is None:
            return True
        if code_answers and answer not in code_answers:
            return True
        return False

    def _format_tool_feedback(
        self,
        answer: int | None,
        code_answers: list[int],
        sandbox_errors: list[str],
    ) -> str:
        pieces = [f"Parsed answer: {answer}"]
        if code_answers:
            pieces.append(f"Code-derived answers: {code_answers}")
        if sandbox_errors:
            pieces.append(f"Sandbox errors: {sandbox_errors[:3]}")
        if not code_answers and not sandbox_errors:
            pieces.append("No executable code evidence was found.")
        return "\n".join(pieces)

    def _should_continue_agent_loop(
        self,
        *,
        answer: int | None,
        answer_source: str,
        code_blocks_in_response: int,
        has_code_evidence: bool,
        complexity: str,
    ) -> bool:
        if self.config.agentic_tool_rounds <= 0:
            return False
        if answer is None:
            return True
        if (
            self.config.force_tool_round_for_unverified
            and not has_code_evidence
            and complexity in {"medium", "hard"}
        ):
            return True
        if code_blocks_in_response <= 0 and answer_source in {"final_answer_tag", "boxed"}:
            return False
        # Continue when answer confidence is weak and code context exists.
        return answer_source not in {"final_answer_tag", "boxed", "verifier", "selector"}

    def _build_agent_observation(
        self,
        *,
        round_index: int,
        previous_response: str,
        code_answers: list[int],
        sandbox_errors: list[str],
    ) -> str:
        response_tail = self._trim_for_prompt(
            previous_response,
            max_chars=max(300, self.config.agentic_observation_chars),
        )
        lines = [f"Round {round_index + 1} tool summary:"]
        if code_answers:
            lines.append(f"- Parsed integers from python stdout: {code_answers[:6]}")
        else:
            lines.append("- No integer parsed from python stdout.")
        if sandbox_errors:
            lines.append(f"- Sandbox errors: {sandbox_errors[:3]}")
        lines.append("Previous response tail:")
        lines.append(response_tail)
        return "\n".join(lines)

    def _parse_tail_answer_integer(self, text: str, *, modulus: int | None) -> int | None:
        tail = "\n".join(text.splitlines()[-6:])
        if not re.search(r"(?:final|answer|therefore|thus|hence)", tail, flags=re.IGNORECASE):
            return None
        return parse_integer_from_stdout(tail, modulus=modulus)

    def _trim_for_prompt(self, text: str, *, max_chars: int = 4500) -> str:
        if len(text) <= max_chars:
            return text
        head = max_chars // 3
        tail = max_chars - head - 32
        return text[:head] + "\n...[truncated]...\n" + text[-tail:]

    def _run_verification(
        self,
        *,
        problem_text: str,
        modulus: int | None,
        profile: ProblemProfile,
        candidates: list[Candidate],
        max_tokens: int,
        problem_numbers: set[int],
        deadline: float | None = None,
    ) -> list[Candidate]:
        if self.config.verification_attempts <= 0:
            return []

        distinct = self._top_distinct_candidates(candidates, self.config.verification_top_k)
        if len(distinct) < 1:
            return []

        candidate_answers = [c.answer for c in distinct if c.answer is not None]
        candidate_summaries = [
            f"answer={c.answer} source={c.answer_source} code_verified={c.code_verified} score={c.score:.2f}"
            for c in distinct
        ]

        verifier_candidates: list[Candidate] = []
        allowed = set(candidate_answers)

        for verify_idx in range(self.config.verification_attempts):
            if not self._has_time(deadline, self.config.min_time_for_stage_sec):
                break
            prompt = build_verifier_prompt(
                problem_text,
                candidate_answers=candidate_answers,
                candidate_summaries=candidate_summaries,
                modulus=modulus,
                profile=profile,
            )
            try:
                response = self.client.generate(
                    system_prompt=prompt.system,
                    user_prompt=prompt.user,
                    temperature=self.config.verification_temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                verifier_candidates.append(
                    Candidate(
                        attempt=10_000 + verify_idx,
                        temperature=self.config.verification_temperature,
                        response_text="",
                        answer=None,
                        answer_source="none",
                        modulus=modulus,
                        category=profile.category,
                        archetype=profile.archetype,
                        complexity=profile.complexity,
                        stage="verifier",
                        generation_error=str(exc),
                    )
                )
                continue

            parsed = parse_answer_from_text(response, modulus=modulus)
            answer = parsed.answer
            if answer not in allowed:
                answer = None

            verifier_candidates.append(
                Candidate(
                    attempt=10_000 + verify_idx,
                    temperature=self.config.verification_temperature,
                    response_text=response,
                    answer=answer,
                    answer_source="verifier" if answer is not None else "none",
                    modulus=modulus,
                    category=profile.category,
                    archetype=profile.archetype,
                    complexity=profile.complexity,
                    stage="verifier",
                    code_blocks=0,
                    code_verified=False,
                    independent_check_passed=bool(answer is not None),
                    answer_in_problem=bool(answer is not None and answer in problem_numbers),
                )
            )

        return verifier_candidates

    def _run_consistency_audit(
        self,
        *,
        problem_text: str,
        modulus: int | None,
        profile: ProblemProfile,
        candidates: list[Candidate],
        max_tokens: int,
        problem_numbers: set[int],
        deadline: float | None = None,
    ) -> list[Candidate]:
        if self.config.consistency_audit_attempts <= 0:
            return []

        distinct = self._top_distinct_candidates(candidates, self.config.consistency_audit_top_k)
        if len(distinct) < 2:
            return []

        candidate_answers = [c.answer for c in distinct if c.answer is not None]
        candidate_summaries = [
            (
                f"answer={c.answer} stage={c.stage} source={c.answer_source} "
                f"code_verified={c.code_verified} score={c.score:.2f}"
            )
            for c in distinct
        ]
        allowed = set(candidate_answers)
        audit_candidates: list[Candidate] = []

        for idx in range(self.config.consistency_audit_attempts):
            if not self._has_time(deadline, self.config.min_time_for_stage_sec):
                break
            prompt = build_consistency_audit_prompt(
                problem_text,
                candidate_answers=candidate_answers,
                candidate_summaries=candidate_summaries,
                modulus=modulus,
                profile=profile,
            )
            try:
                response = self.client.generate(
                    system_prompt=prompt.system,
                    user_prompt=prompt.user,
                    temperature=self.config.consistency_audit_temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                audit_candidates.append(
                    Candidate(
                        attempt=11_000 + idx,
                        temperature=self.config.consistency_audit_temperature,
                        response_text="",
                        answer=None,
                        answer_source="none",
                        modulus=modulus,
                        category=profile.category,
                        archetype=profile.archetype,
                        complexity=profile.complexity,
                        stage="consistency_audit",
                        generation_error=str(exc),
                    )
                )
                continue

            parsed = parse_answer_from_text(response, modulus=modulus)
            answer = parsed.answer
            if answer not in allowed:
                answer = None

            audit_candidates.append(
                Candidate(
                    attempt=11_000 + idx,
                    temperature=self.config.consistency_audit_temperature,
                    response_text=response,
                    answer=answer,
                    answer_source="consistency_audit" if answer is not None else "none",
                    modulus=modulus,
                    category=profile.category,
                    archetype=profile.archetype,
                    complexity=profile.complexity,
                    stage="consistency_audit",
                    code_blocks=0,
                    code_verified=False,
                    independent_check_passed=bool(answer is not None),
                    answer_in_problem=bool(answer is not None and answer in problem_numbers),
                )
            )

        return audit_candidates

    def _run_adversarial_probe(
        self,
        *,
        problem_text: str,
        modulus: int | None,
        profile: ProblemProfile,
        candidates: list[Candidate],
        max_tokens: int,
        problem_numbers: set[int],
        deadline: float | None = None,
    ) -> list[Candidate]:
        if self.config.adversarial_probe_attempts <= 0:
            return []
        if not self._should_run_adversarial_probe(candidates):
            return []

        distinct = self._top_distinct_candidates(candidates, self.config.adversarial_probe_top_k)
        candidate_answers = [c.answer for c in distinct if c.answer is not None]
        candidate_summaries = [
            (
                f"answer={c.answer} stage={c.stage} source={c.answer_source} "
                f"code_verified={c.code_verified} score={c.score:.2f}"
            )
            for c in distinct
        ]

        probe_candidates: list[Candidate] = []
        for idx in range(self.config.adversarial_probe_attempts):
            if not self._has_time(deadline, self.config.min_time_for_stage_sec):
                break
            prompt = build_adversarial_probe_prompt(
                problem_text,
                candidate_answers=candidate_answers,
                candidate_summaries=candidate_summaries,
                modulus=modulus,
                profile=profile,
            )
            try:
                response = self.client.generate(
                    system_prompt=prompt.system,
                    user_prompt=prompt.user,
                    temperature=self.config.adversarial_probe_temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                probe_candidates.append(
                    Candidate(
                        attempt=11_500 + idx,
                        temperature=self.config.adversarial_probe_temperature,
                        response_text="",
                        answer=None,
                        answer_source="none",
                        modulus=modulus,
                        category=profile.category,
                        archetype=profile.archetype,
                        complexity=profile.complexity,
                        stage="adversarial_probe",
                        generation_error=str(exc),
                    )
                )
                continue

            parsed = parse_answer_from_text(response, modulus=modulus)
            answer = parsed.answer

            probe_candidates.append(
                Candidate(
                    attempt=11_500 + idx,
                    temperature=self.config.adversarial_probe_temperature,
                    response_text=response,
                    answer=answer,
                    answer_source="adversarial_probe" if answer is not None else "none",
                    modulus=modulus,
                    category=profile.category,
                    archetype=profile.archetype,
                    complexity=profile.complexity,
                    stage="adversarial_probe",
                    code_blocks=0,
                    code_verified=False,
                    independent_check_passed=bool(answer is not None),
                    answer_in_problem=bool(answer is not None and answer in problem_numbers),
                )
            )

        return probe_candidates

    def _should_run_adversarial_probe(self, candidates: list[Candidate]) -> bool:
        valid = [c for c in candidates if c.answer is not None]
        if not valid:
            return False

        answers = [c.answer for c in valid if c.answer is not None]
        if not answers:
            return False

        top_answer, top_count = Counter(answers).most_common(1)[0]
        top_ratio = top_count / max(1, len(answers))
        distinct = len(set(answers))

        strong_support = any(
            c.answer == top_answer
            and (
                c.code_verified
                or c.stage in {"verifier", "consistency_audit", "geometry_recheck", "selector"}
            )
            for c in valid
        )

        if distinct >= 2 and top_ratio < 0.78:
            return True

        if len(valid) == 1 and not strong_support:
            return True

        if top_answer in {0, 1} and not strong_support:
            return True

        return False

    def _run_geometry_recheck(
        self,
        *,
        problem_text: str,
        modulus: int | None,
        profile: ProblemProfile,
        candidates: list[Candidate],
        max_tokens: int,
        problem_numbers: set[int],
        deadline: float | None = None,
    ) -> list[Candidate]:
        if self.config.geometry_recheck_attempts <= 0:
            return []
        if profile.category != "geometry":
            return []

        distinct = self._top_distinct_candidates(candidates, self.config.geometry_top_k)
        if len(distinct) < 2:
            return []

        candidate_answers = [c.answer for c in distinct if c.answer is not None]
        candidate_summaries = [
            (
                f"answer={c.answer} stage={c.stage} source={c.answer_source} "
                f"code_verified={c.code_verified} score={c.score:.2f}"
            )
            for c in distinct
        ]
        allowed = set(candidate_answers)
        geometry_candidates: list[Candidate] = []

        for idx in range(self.config.geometry_recheck_attempts):
            if not self._has_time(deadline, self.config.min_time_for_stage_sec):
                break
            prompt = build_geometry_recheck_prompt(
                problem_text,
                candidate_answers=candidate_answers,
                candidate_summaries=candidate_summaries,
                modulus=modulus,
                profile=profile,
            )
            try:
                response = self.client.generate(
                    system_prompt=prompt.system,
                    user_prompt=prompt.user,
                    temperature=self.config.geometry_recheck_temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                geometry_candidates.append(
                    Candidate(
                        attempt=12_000 + idx,
                        temperature=self.config.geometry_recheck_temperature,
                        response_text="",
                        answer=None,
                        answer_source="none",
                        modulus=modulus,
                        category=profile.category,
                        archetype=profile.archetype,
                        complexity=profile.complexity,
                        stage="geometry_recheck",
                        generation_error=str(exc),
                    )
                )
                continue

            parsed = parse_answer_from_text(response, modulus=modulus)
            answer = parsed.answer
            if answer not in allowed:
                answer = None

            geometry_candidates.append(
                Candidate(
                    attempt=12_000 + idx,
                    temperature=self.config.geometry_recheck_temperature,
                    response_text=response,
                    answer=answer,
                    answer_source="geometry_recheck" if answer is not None else "none",
                    modulus=modulus,
                    category=profile.category,
                    archetype=profile.archetype,
                    complexity=profile.complexity,
                    stage="geometry_recheck",
                    code_blocks=0,
                    code_verified=False,
                    independent_check_passed=bool(answer is not None),
                    answer_in_problem=bool(answer is not None and answer in problem_numbers),
                )
            )

        return geometry_candidates

    def _run_small_answer_guard(
        self,
        *,
        problem_text: str,
        modulus: int | None,
        profile: ProblemProfile,
        candidates: list[Candidate],
        max_tokens: int,
        problem_numbers: set[int],
        deadline: float | None = None,
    ) -> list[Candidate]:
        if self.config.small_answer_guard_attempts <= 0:
            return []
        if not self._should_run_small_answer_guard(candidates):
            return []

        distinct = self._top_distinct_candidates(candidates, self.config.small_answer_guard_top_k)
        candidate_answers = [c.answer for c in distinct if c.answer is not None]
        candidate_summaries = [
            (
                f"answer={c.answer} stage={c.stage} source={c.answer_source} "
                f"code_verified={c.code_verified} score={c.score:.2f}"
            )
            for c in distinct
        ]

        guard_candidates: list[Candidate] = []
        for idx in range(self.config.small_answer_guard_attempts):
            if not self._has_time(deadline, self.config.min_time_for_stage_sec):
                break
            prompt = build_small_answer_guard_prompt(
                problem_text,
                candidate_answers=candidate_answers,
                candidate_summaries=candidate_summaries,
                modulus=modulus,
                profile=profile,
            )
            try:
                response = self.client.generate(
                    system_prompt=prompt.system,
                    user_prompt=prompt.user,
                    temperature=self.config.small_answer_guard_temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                guard_candidates.append(
                    Candidate(
                        attempt=14_000 + idx,
                        temperature=self.config.small_answer_guard_temperature,
                        response_text="",
                        answer=None,
                        answer_source="none",
                        modulus=modulus,
                        category=profile.category,
                        archetype=profile.archetype,
                        complexity=profile.complexity,
                        stage="small_guard",
                        generation_error=str(exc),
                    )
                )
                continue

            parsed = parse_answer_from_text(response, modulus=modulus)
            answer = parsed.answer
            guard_candidates.append(
                Candidate(
                    attempt=14_000 + idx,
                    temperature=self.config.small_answer_guard_temperature,
                    response_text=response,
                    answer=answer,
                    answer_source="small_guard" if answer is not None else "none",
                    modulus=modulus,
                    category=profile.category,
                    archetype=profile.archetype,
                    complexity=profile.complexity,
                    stage="small_guard",
                    code_blocks=0,
                    code_verified=False,
                    independent_check_passed=bool(answer is not None and answer not in {0, 1}),
                    answer_in_problem=bool(answer is not None and answer in problem_numbers),
                )
            )

        return guard_candidates

    def _should_run_small_answer_guard(self, candidates: list[Candidate]) -> bool:
        valid = [c for c in candidates if c.answer is not None]
        if len(valid) < 1:
            # All extraction paths failed: run one guarded salvage attempt instead of defaulting to 0.
            return True

        answers = [c.answer for c in valid if c.answer is not None]
        if not answers:
            return False

        top_answer, top_count = Counter(answers).most_common(1)[0]
        top_ratio = top_count / max(1, len(answers))
        small_count = sum(1 for answer in answers if answer in {0, 1})
        small_ratio = small_count / max(1, len(answers))

        top_has_strong_evidence = any(
            c.answer == top_answer
            and (
                c.code_verified
                or c.stage
                in {
                    "verifier",
                    "consistency_audit",
                    "adversarial_probe",
                    "geometry_recheck",
                    "selector",
                }
            )
            for c in valid
        )

        if len(valid) == 1 and top_answer in {0, 1} and not top_has_strong_evidence:
            return True

        if top_answer in {0, 1} and not top_has_strong_evidence and top_ratio < 0.95:
            return True

        if (
            self.config.strict_zero_one_policy
            and top_answer in {0, 1}
            and not top_has_strong_evidence
        ):
            return True

        return small_ratio >= 0.6

    def _run_fallback_guess(
        self,
        *,
        problem_text: str,
        modulus: int | None,
        profile: ProblemProfile,
        candidates: list[Candidate],
        max_tokens: int,
        problem_numbers: set[int],
        deadline: float | None = None,
    ) -> list[Candidate]:
        if self.config.fallback_guess_attempts <= 0:
            return []
        if not self._should_run_fallback_guess(candidates):
            return []

        fallback_candidates: list[Candidate] = []
        for idx in range(self.config.fallback_guess_attempts):
            if not self._has_time(deadline, 2):
                break
            prompt = build_fallback_guess_prompt(
                problem_text,
                modulus=modulus,
                profile=profile,
            )
            try:
                response = self.client.generate(
                    system_prompt=prompt.system,
                    user_prompt=prompt.user,
                    temperature=self.config.fallback_guess_temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                fallback_candidates.append(
                    Candidate(
                        attempt=16_000 + idx,
                        temperature=self.config.fallback_guess_temperature,
                        response_text="",
                        answer=None,
                        answer_source="none",
                        modulus=modulus,
                        category=profile.category,
                        archetype=profile.archetype,
                        complexity=profile.complexity,
                        stage="fallback_guess",
                        generation_error=str(exc),
                    )
                )
                continue

            parsed = parse_answer_from_text(response, modulus=modulus)
            answer = parsed.answer
            answer_source = "fallback_guess" if answer is not None else "none"
            if answer is None:
                heuristic = self._parse_tail_answer_integer(response, modulus=modulus)
                if heuristic is not None:
                    answer = heuristic
                    answer_source = "fallback_guess_heuristic"

            fallback_candidates.append(
                Candidate(
                    attempt=16_000 + idx,
                    temperature=self.config.fallback_guess_temperature,
                    response_text=response,
                    answer=answer,
                    answer_source=answer_source,
                    modulus=modulus,
                    category=profile.category,
                    archetype=profile.archetype,
                    complexity=profile.complexity,
                    stage="fallback_guess",
                    code_blocks=0,
                    code_verified=False,
                    independent_check_passed=False,
                    answer_in_problem=bool(answer is not None and answer in problem_numbers),
                )
            )

        return fallback_candidates

    def _should_run_fallback_guess(self, candidates: list[Candidate]) -> bool:
        valid = [c for c in candidates if c.answer is not None]
        if not valid:
            return True

        answers = [c.answer for c in valid if c.answer is not None]
        if not answers:
            return True

        unique = set(answers)
        if not unique.issubset({0, 1}):
            return False

        strong_evidence = any(
            c.answer in {0, 1}
            and (
                c.code_verified
                or c.stage
                in {
                    "verifier",
                    "consistency_audit",
                    "adversarial_probe",
                    "geometry_recheck",
                    "selector",
                }
            )
            for c in valid
        )
        if not strong_evidence:
            return True

        _, count = Counter(answers).most_common(1)[0]
        ratio = count / max(1, len(answers))
        return ratio < 0.9

    def _run_selector(
        self,
        *,
        problem_text: str,
        modulus: int | None,
        profile: ProblemProfile,
        candidates: list[Candidate],
        max_tokens: int,
        problem_numbers: set[int],
        deadline: float | None = None,
    ) -> list[Candidate]:
        if self.config.selector_attempts <= 0:
            return []

        distinct = self._top_distinct_candidates(candidates, self.config.selector_top_k)
        if len(distinct) < 2:
            return []

        options = [c.answer for c in distinct if c.answer is not None]
        if len(options) < 2:
            return []

        evidence = [self._build_option_evidence(answer, candidates) for answer in options]
        allowed = set(options)
        selector_candidates: list[Candidate] = []

        for idx in range(self.config.selector_attempts):
            if not self._has_time(deadline, self.config.min_time_for_stage_sec):
                break
            prompt = build_selector_prompt(
                problem_text,
                answer_options=options,
                option_evidence=evidence,
                modulus=modulus,
                profile=profile,
            )
            try:
                response = self.client.generate(
                    system_prompt=prompt.system,
                    user_prompt=prompt.user,
                    temperature=self.config.selector_temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                selector_candidates.append(
                    Candidate(
                        attempt=15_000 + idx,
                        temperature=self.config.selector_temperature,
                        response_text="",
                        answer=None,
                        answer_source="none",
                        modulus=modulus,
                        category=profile.category,
                        archetype=profile.archetype,
                        complexity=profile.complexity,
                        stage="selector",
                        generation_error=str(exc),
                    )
                )
                continue

            parsed = parse_answer_from_text(response, modulus=modulus)
            answer = parsed.answer
            if answer not in allowed:
                answer = None

            selector_candidates.append(
                Candidate(
                    attempt=15_000 + idx,
                    temperature=self.config.selector_temperature,
                    response_text=response,
                    answer=answer,
                    answer_source="selector" if answer is not None else "none",
                    modulus=modulus,
                    category=profile.category,
                    archetype=profile.archetype,
                    complexity=profile.complexity,
                    stage="selector",
                    code_blocks=0,
                    code_verified=False,
                    independent_check_passed=bool(answer is not None),
                    answer_in_problem=bool(answer is not None and answer in problem_numbers),
                )
            )

        return selector_candidates

    def _run_sparse_recovery(
        self,
        *,
        problem_text: str,
        modulus: int | None,
        profile: ProblemProfile,
        candidates: list[Candidate],
        max_tokens: int,
        problem_numbers: set[int],
        deadline: float | None = None,
    ) -> list[Candidate]:
        if self.config.sparse_recovery_attempts <= 0:
            return []

        valid_answers = [c for c in candidates if c.answer is not None]
        if len(valid_answers) >= 2:
            return []

        rescue: list[Candidate] = []
        base_idx = 20_000 + len(candidates)
        for i in range(self.config.sparse_recovery_attempts):
            if not self._has_time(deadline, self.config.min_time_for_attempt_sec):
                break
            rescue_candidate = self._run_attempt(
                problem_text=problem_text,
                modulus=modulus,
                profile=profile,
                attempt=base_idx + i,
                temperature=self.config.sparse_recovery_temperature,
                max_tokens=max_tokens,
                problem_numbers=problem_numbers,
                deadline=deadline,
            )
            rescue_candidate.stage = "recovery"
            rescue.append(rescue_candidate)

        return rescue

    def _top_distinct_candidates(self, candidates: list[Candidate], limit: int) -> list[Candidate]:
        ranked = sorted(
            [c for c in candidates if c.answer is not None],
            key=lambda c: c.score,
            reverse=True,
        )
        distinct: list[Candidate] = []
        seen: set[int] = set()
        for candidate in ranked:
            if candidate.answer is None or candidate.answer in seen:
                continue
            distinct.append(candidate)
            seen.add(candidate.answer)
            if len(distinct) >= max(1, limit):
                break
        return distinct

    def _build_option_evidence(self, answer: int, candidates: list[Candidate]) -> str:
        matching = [c for c in candidates if c.answer == answer]
        if not matching:
            return "No direct evidence."

        support = len(matching)
        verified = sum(1 for c in matching if c.code_verified)
        independent = sum(1 for c in matching if c.independent_check_passed)
        agent_rounds = sum(c.agent_rounds for c in matching)
        verifier_votes = sum(1 for c in matching if c.stage == "verifier")
        audit_votes = sum(1 for c in matching if c.stage == "consistency_audit")
        probe_votes = sum(1 for c in matching if c.stage == "adversarial_probe")
        selector_votes = sum(1 for c in matching if c.stage == "selector")
        max_score = max(c.score for c in matching)

        top = sorted(matching, key=lambda c: c.score, reverse=True)[:2]
        snippets: list[str] = []
        for idx, candidate in enumerate(top, start=1):
            normalized = " ".join(candidate.response_text.split())
            snippet = normalized[:300]
            if len(normalized) > 300:
                snippet += "..."
            snippets.append(
                f"trace{idx}: stage={candidate.stage} source={candidate.answer_source} "
                f"independent={candidate.independent_check_passed} method={candidate.method or 'n/a'} "
                f"score={candidate.score:.2f} snippet={snippet}"
            )

        stats = (
            f"support={support} code_verified={verified} "
            f"independent_checks={independent} "
            f"verifier_votes={verifier_votes} audit_votes={audit_votes} "
            f"probe_votes={probe_votes} selector_votes={selector_votes} "
            f"agent_rounds={agent_rounds} max_score={max_score:.2f}"
        )
        return stats + "\n" + "\n".join(snippets)

    def _extract_problem_numbers(self, problem_text: str) -> set[int]:
        numbers: set[int] = set()
        for match in re.findall(r"(?<!\d)\d{1,6}(?!\d)", problem_text):
            try:
                numbers.add(int(match))
            except Exception:
                continue
        return numbers

    def _should_early_stop(self, candidates: list[Candidate]) -> bool:
        if len(candidates) < self.config.early_stop_attempt:
            return False

        answers = [c.answer for c in candidates if c.answer is not None]
        if not answers:
            return False

        top_answer, count = Counter(answers).most_common(1)[0]
        ratio = count / max(1, len(answers))
        top_has_strong_evidence = any(
            c.answer == top_answer
            and (
                c.code_verified
                or c.independent_check_passed
                or c.stage
                in {
                    "verifier",
                    "consistency_audit",
                    "adversarial_probe",
                    "geometry_recheck",
                    "selector",
                }
            )
            for c in candidates
        )

        if count >= self.config.min_consensus and ratio >= self.config.early_stop_ratio:
            if (
                self.config.strict_zero_one_policy
                and top_answer in {0, 1}
                and not top_has_strong_evidence
            ):
                return False
            return True

        verified_count = sum(
            1
            for c in candidates
            if (c.code_verified or c.independent_check_passed) and c.answer == top_answer
        )
        return verified_count >= 2 and ratio >= 0.6

    def _aggregate(
        self, candidates: list[Candidate], modulus: int | None, problem_numbers: set[int]
    ) -> int:
        stats: dict[int, dict[str, Any]] = {}
        for candidate in candidates:
            if candidate.answer is None:
                continue
            row = stats.setdefault(
                candidate.answer,
                {
                    "score": 0.0,
                    "count": 0.0,
                    "verified": 0.0,
                    "verifier_votes": 0.0,
                    "audit_votes": 0.0,
                    "probe_votes": 0.0,
                    "geometry_votes": 0.0,
                    "guard_votes": 0.0,
                    "selector_votes": 0.0,
                    "independent": 0.0,
                    "forced_missing": 0.0,
                    "text_only_unverified": 0.0,
                    "stages": set(),
                    "sources": set(),
                },
            )
            row["score"] += candidate.score
            row["count"] += 1.0
            row["stages"].add(candidate.stage)
            row["sources"].add(candidate.answer_source)
            if candidate.code_verified:
                row["verified"] += 1.0
            if candidate.independent_check_passed:
                row["independent"] += 1.0
            if candidate.missing_forced_code_check:
                row["forced_missing"] += 1.0
            if (
                candidate.answer_source
                in {"final_answer_tag", "boxed", "answer_line", "plain_integer"}
                and not candidate.code_verified
                and not candidate.independent_check_passed
            ):
                row["text_only_unverified"] += 1.0
            if candidate.stage == "verifier":
                row["verifier_votes"] += 1.0
            if candidate.stage == "consistency_audit":
                row["audit_votes"] += 1.0
            if candidate.stage == "adversarial_probe":
                row["probe_votes"] += 1.0
            if candidate.stage == "geometry_recheck":
                row["geometry_votes"] += 1.0
            if candidate.stage == "small_guard":
                row["guard_votes"] += 1.0
            if candidate.stage == "selector":
                row["selector_votes"] += 1.0

        if not stats:
            return self.config.default_answer

        ranked_answers: dict[int, tuple[Any, ...]] = {}
        best_answer = self.config.default_answer
        best_key: tuple[Any, ...] | None = None

        for answer, row in stats.items():
            total_score = row["score"]
            total_score += 0.25 * min(row["verified"], 4.0)
            total_score += 0.35 * row["verifier_votes"]
            total_score += 0.38 * row["audit_votes"]
            total_score += 0.3 * row["probe_votes"]
            total_score += 0.4 * row["geometry_votes"]
            total_score += 0.45 * row["guard_votes"]
            total_score += 0.5 * row["selector_votes"]
            total_score += 0.55 * min(row["independent"], 4.0)
            stage_diversity = max(0.0, float(len(row["stages"]) - 1))
            total_score += 0.12 * min(4.0, stage_diversity)
            source_diversity = max(0.0, float(len(row["sources"]) - 1))
            total_score += 0.08 * min(4.0, source_diversity)
            total_score -= 0.28 * min(4.0, row["text_only_unverified"])
            total_score -= 0.35 * min(4.0, row["forced_missing"])

            if 0 <= answer <= 9:
                total_score -= 0.25
            if (
                answer in {0, 1}
                and (
                    row["verified"]
                    + row["independent"]
                    + row["verifier_votes"]
                    + row["audit_votes"]
                    + row["probe_votes"]
                    + row["geometry_votes"]
                    + row["selector_votes"]
                )
                <= 0
            ):
                total_score -= 1.4
            if row["stages"] == {"fallback_guess"}:
                total_score -= 0.2
            if answer in problem_numbers:
                total_score -= 0.15

            rank_key = (
                total_score,
                row["independent"],
                row["verified"],
                row["selector_votes"],
                row["guard_votes"],
                row["audit_votes"],
                row["probe_votes"],
                row["geometry_votes"],
                row["count"],
                answer,
            )
            ranked_answers[answer] = rank_key
            if best_key is None or rank_key > best_key:
                best_key = rank_key
                best_answer = answer

        if self.config.strict_zero_one_policy and best_answer in {0, 1}:
            row = stats.get(best_answer, {})
            weak_small = (
                row.get("independent", 0.0) <= 0
                and row.get("verified", 0.0) <= 0
                and row.get("verifier_votes", 0.0) <= 0
                and row.get("audit_votes", 0.0) <= 0
                and row.get("probe_votes", 0.0) <= 0
                and row.get("geometry_votes", 0.0) <= 0
                and row.get("selector_votes", 0.0) <= 0
            )
            if weak_small:
                alternatives = [
                    (ans, key) for ans, key in ranked_answers.items() if ans not in {0, 1}
                ]
                if alternatives:
                    alternatives.sort(key=lambda item: item[1], reverse=True)
                    best_answer = alternatives[0][0]

        if modulus:
            return best_answer % modulus

        return best_answer % 100_000
