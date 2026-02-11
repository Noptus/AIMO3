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
from .prompts import build_prompt
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


@dataclass
class Candidate:
    attempt: int
    temperature: float
    response_text: str
    answer: int | None
    answer_source: str
    modulus: int | None
    code_blocks: int
    code_verified: bool
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
        if self.code_verified:
            base += 2.2
        elif self.code_answers:
            base += 0.5

        base -= min(0.5, 0.2 * len(self.sandbox_errors))
        return max(base, 0.01)


@dataclass
class SolveResult:
    problem_id: Any
    predicted_answer: int
    modulus: int | None
    candidates: list[Candidate]

    @property
    def debug_summary(self) -> dict[str, Any]:
        votes = Counter(c.answer for c in self.candidates if c.answer is not None)
        return {
            "predicted_answer": self.predicted_answer,
            "modulus": self.modulus,
            "candidate_count": len(self.candidates),
            "top_votes": votes.most_common(5),
            "verified_candidates": sum(1 for c in self.candidates if c.code_verified),
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
        candidates: list[Candidate] = []

        for attempt in range(self.config.attempts):
            temp = self.config.temperatures[attempt % len(self.config.temperatures)]
            candidate = self._run_attempt(
                problem_text=problem_text,
                modulus=modulus,
                attempt=attempt,
                temperature=temp,
            )
            candidates.append(candidate)

            if self._should_early_stop(candidates):
                break

        predicted = self._aggregate(candidates, modulus)

        return SolveResult(
            problem_id=problem_id,
            predicted_answer=predicted,
            modulus=modulus,
            candidates=candidates,
        )

    def _run_attempt(
        self,
        *,
        problem_text: str,
        modulus: int | None,
        attempt: int,
        temperature: float,
    ) -> Candidate:
        prompt = build_prompt(problem_text, attempt_index=attempt, modulus=modulus)

        try:
            response_text = self.client.generate(
                system_prompt=prompt.system,
                user_prompt=prompt.user,
                temperature=temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:
            return Candidate(
                attempt=attempt,
                temperature=temperature,
                response_text="",
                answer=None,
                answer_source="none",
                modulus=modulus,
                code_blocks=0,
                code_verified=False,
                generation_error=str(exc),
            )

        parsed = parse_answer_from_text(response_text, modulus=modulus)
        answer = parsed.answer
        code_answers: list[int] = []
        sandbox_errors: list[str] = []

        code_blocks = extract_python_blocks(response_text)
        for block in code_blocks[: self.config.max_code_blocks_per_attempt]:
            run = execute_python(block, policy=self.sandbox_policy)
            if run.success:
                code_answer = parse_integer_from_stdout(run.stdout, modulus=modulus)
                if code_answer is not None:
                    code_answers.append(code_answer)
            else:
                if run.error:
                    sandbox_errors.append(run.error)

        if answer is None and code_answers:
            answer = Counter(code_answers).most_common(1)[0][0]

        code_verified = False
        if answer is not None and code_answers:
            code_verified = answer in code_answers

        return Candidate(
            attempt=attempt,
            temperature=temperature,
            response_text=response_text,
            answer=answer,
            answer_source=parsed.source,
            modulus=modulus,
            code_blocks=len(code_blocks),
            code_verified=code_verified,
            code_answers=code_answers,
            sandbox_errors=sandbox_errors,
        )

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
