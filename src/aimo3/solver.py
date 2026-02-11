"""Composable AIMO3 solver with tool-integrated reasoning."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .client import ChatClient
from .parsing import (
    extract_python_blocks,
    parse_answer_from_text,
    parse_integer_from_stdout,
    parse_modulus,
    select_weighted_mode,
)
from .prompts import (
    ProblemProfile,
    build_prompt,
    build_repair_prompt,
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
    early_stop_attempt: int = 4
    max_code_blocks_per_attempt: int = 2
    default_answer: int = 0

    # Hard-problem behavior.
    adaptive_complexity: bool = True
    hard_mode: bool = False
    hard_attempts: int = 12
    hard_max_tokens: int = 1536

    # Second-pass correction and arbitration.
    repair_passes: int = 0
    verification_attempts: int = 0
    verification_top_k: int = 3
    verification_temperature: float = 0.1


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
    complexity: str
    stage: str = "initial"
    code_blocks: int = 0
    code_verified: bool = False
    repair_used: bool = False
    code_answers: list[int] = field(default_factory=list)
    sandbox_errors: list[str] = field(default_factory=list)
    generation_error: str | None = None

    @property
    def score(self) -> float:
        if self.generation_error:
            return 0.01

        base = 0.2
        if self.answer is not None:
            base += 1.0

        if self.answer_source in {"final_answer_tag", "boxed"}:
            base += 0.6
        elif self.answer_source == "verifier":
            base += 0.8

        if self.code_verified:
            base += 2.2
        elif self.code_answers:
            base += 0.5

        if self.repair_used:
            base += 0.2

        if self.complexity == "hard":
            base += 0.1

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
            "complexity": self.profile.complexity,
            "complexity_score": self.profile.score,
            "candidate_count": len(self.candidates),
            "top_votes": votes.most_common(5),
            "verified_candidates": sum(1 for c in self.candidates if c.code_verified),
            "repair_candidates": sum(1 for c in self.candidates if c.repair_used),
            "verifier_candidates": sum(1 for c in self.candidates if c.stage == "verifier"),
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

    def solve(self, problem_text: str, problem_id: Any = None) -> SolveResult:
        modulus = parse_modulus(problem_text)
        profile = estimate_problem_profile(problem_text)
        budget = self._build_budget(profile)

        candidates: list[Candidate] = []

        for attempt in range(budget.attempts):
            temp = self.config.temperatures[attempt % len(self.config.temperatures)]
            candidate = self._run_attempt(
                problem_text=problem_text,
                modulus=modulus,
                profile=profile,
                attempt=attempt,
                temperature=temp,
                max_tokens=budget.max_tokens,
            )
            candidates.append(candidate)

            if budget.allow_early_stop and self._should_early_stop(candidates):
                break

        candidates.extend(
            self._run_verification(
                problem_text=problem_text,
                modulus=modulus,
                profile=profile,
                candidates=candidates,
                max_tokens=max(256, min(640, budget.max_tokens // 2)),
            )
        )

        predicted = self._aggregate(candidates, modulus)

        return SolveResult(
            problem_id=problem_id,
            predicted_answer=predicted,
            modulus=modulus,
            profile=profile,
            candidates=candidates,
        )

    def _build_budget(self, profile: ProblemProfile) -> SolveBudget:
        attempts = self.config.attempts
        max_tokens = self.config.max_tokens
        allow_early_stop = True

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
            max_tokens = max(max_tokens, min(self.config.hard_max_tokens, self.config.max_tokens + 256))

        return SolveBudget(
            attempts=attempts,
            max_tokens=max_tokens,
            allow_early_stop=allow_early_stop,
        )

    def _run_attempt(
        self,
        *,
        problem_text: str,
        modulus: int | None,
        profile: ProblemProfile,
        attempt: int,
        temperature: float,
        max_tokens: int,
    ) -> Candidate:
        prompt = build_prompt(
            problem_text,
            attempt_index=attempt,
            modulus=modulus,
            profile=profile,
            hard_mode=self.config.hard_mode,
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
                complexity=profile.complexity,
                generation_error=str(exc),
            )

        parsed = parse_answer_from_text(response_text, modulus=modulus)
        answer = parsed.answer
        code_answers, sandbox_errors, code_blocks_count = self._evaluate_code_blocks(response_text, modulus)

        if answer is None and code_answers:
            answer = Counter(code_answers).most_common(1)[0][0]

        repair_used = False
        for _ in range(self.config.repair_passes):
            if not self._needs_repair(answer, code_answers):
                break

            repair_used = True
            tool_feedback = self._format_tool_feedback(answer, code_answers, sandbox_errors)
            repair_prompt = build_repair_prompt(
                problem_text,
                previous_response=response_text,
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
            extra_answers, extra_errors, extra_code_blocks = self._evaluate_code_blocks(response_text, modulus)
            code_answers.extend(extra_answers)
            sandbox_errors.extend(extra_errors)
            code_blocks_count += extra_code_blocks

            if answer is None and code_answers:
                answer = Counter(code_answers).most_common(1)[0][0]

        code_verified = bool(answer is not None and code_answers and answer in code_answers)

        return Candidate(
            attempt=attempt,
            temperature=temperature,
            response_text=response_text,
            answer=answer,
            answer_source=parsed.source,
            modulus=modulus,
            category=profile.category,
            complexity=profile.complexity,
            stage="initial",
            code_blocks=code_blocks_count,
            code_verified=code_verified,
            repair_used=repair_used,
            code_answers=code_answers,
            sandbox_errors=sandbox_errors,
        )

    def _evaluate_code_blocks(
        self,
        response_text: str,
        modulus: int | None,
    ) -> tuple[list[int], list[str], int]:
        code_answers: list[int] = []
        sandbox_errors: list[str] = []
        code_blocks = extract_python_blocks(response_text)

        for block in code_blocks[: self.config.max_code_blocks_per_attempt]:
            run = execute_python(block, policy=self.sandbox_policy)
            if run.success:
                code_answer = parse_integer_from_stdout(run.stdout, modulus=modulus)
                if code_answer is not None:
                    code_answers.append(code_answer)
            elif run.error:
                sandbox_errors.append(run.error)

        return code_answers, sandbox_errors, len(code_blocks)

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

    def _run_verification(
        self,
        *,
        problem_text: str,
        modulus: int | None,
        profile: ProblemProfile,
        candidates: list[Candidate],
        max_tokens: int,
    ) -> list[Candidate]:
        if self.config.verification_attempts <= 0:
            return []

        # Keep top distinct answers only.
        ranked = sorted(
            [c for c in candidates if c.answer is not None],
            key=lambda c: c.score,
            reverse=True,
        )
        if len(ranked) < 2:
            return []

        distinct: list[Candidate] = []
        seen: set[int] = set()
        for candidate in ranked:
            if candidate.answer is None:
                continue
            if candidate.answer in seen:
                continue
            distinct.append(candidate)
            seen.add(candidate.answer)
            if len(distinct) >= self.config.verification_top_k:
                break

        if len(distinct) < 2:
            return []

        candidate_answers = [c.answer for c in distinct if c.answer is not None]
        candidate_summaries = [
            f"answer={c.answer} source={c.answer_source} code_verified={c.code_verified} score={c.score:.2f}"
            for c in distinct
        ]

        verifier_candidates: list[Candidate] = []
        allowed = set(candidate_answers)

        for verify_idx in range(self.config.verification_attempts):
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
                    complexity=profile.complexity,
                    stage="verifier",
                    code_blocks=0,
                    code_verified=False,
                )
            )

        return verifier_candidates

    def _should_early_stop(self, candidates: list[Candidate]) -> bool:
        if len(candidates) < self.config.early_stop_attempt:
            return False

        answers = [c.answer for c in candidates if c.answer is not None]
        if not answers:
            return False

        top_answer, count = Counter(answers).most_common(1)[0]
        if count >= self.config.min_consensus:
            return True

        verified_count = sum(1 for c in candidates if c.code_verified and c.answer == top_answer)
        return verified_count >= 2

    def _aggregate(self, candidates: list[Candidate], modulus: int | None) -> int:
        answers = [c.answer for c in candidates if c.answer is not None]
        if not answers:
            return self.config.default_answer

        weights = [c.score for c in candidates if c.answer is not None]
        weighted = select_weighted_mode(answers, weights)

        if weighted is None:
            weighted = Counter(answers).most_common(1)[0][0]

        if modulus:
            return weighted % modulus

        return weighted % 100_000
