"""LangGraph-based orchestration for AIMO3 solving."""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except Exception as exc:  # pragma: no cover - exercised in environments without langgraph
    END = START = StateGraph = None
    _LANGGRAPH_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - import has no behavior to test directly
    _LANGGRAPH_IMPORT_ERROR = None

from .parsing import (
    INTEGER_RE,
    extract_python_blocks,
    normalize_answer,
    parse_answer_from_text,
    parse_integer_from_stdout,
    parse_modulus,
    select_weighted_mode,
)
from .prompts import (
    ProblemProfile,
    build_agent_followup_prompt,
    build_prompt,
    estimate_problem_profile,
)
from .sandbox import SandboxPolicy, execute_python
from .solver import AIMO3Solver, Candidate, SolverConfig, SolveResult


class LangGraphUnavailableError(RuntimeError):
    """Raised when the user selects LangGraph orchestration but dependency is missing."""


def is_langgraph_available() -> bool:
    """Return whether LangGraph runtime is importable."""

    return _LANGGRAPH_IMPORT_ERROR is None


class _GraphState(TypedDict, total=False):
    problem_id: Any
    problem_text: str
    modulus: int | None
    profile: ProblemProfile
    attempts_limit: int
    tokens_limit: int
    deadline_ts: float | None

    attempt_index: int
    round_index: int
    current_temperature: float

    current_response: str
    current_generation_error: str | None
    current_answer: int | None
    current_answer_source: str
    current_code_answers: list[int]
    current_sandbox_errors: list[str]
    current_code_blocks: int
    current_tool_observation: str
    stateful_prefix: str

    weighted_answers: list[int]
    weighted_weights: list[float]
    candidates: list[Candidate]
    done: bool


@dataclass(frozen=True)
class LangGraphConfig:
    """Operational controls specific to graph orchestration."""

    max_followup_rounds: int = 2
    followup_answer_sources: tuple[str, ...] = ("final_answer_tag", "boxed")


class LangGraphAIMO3Solver(AIMO3Solver):
    """AIMO3 solver implemented as a LangGraph state machine."""

    def __init__(
        self,
        client,
        *,
        config: SolverConfig | None = None,
        sandbox_policy: SandboxPolicy | None = None,
        langgraph_config: LangGraphConfig | None = None,
    ) -> None:
        if not is_langgraph_available():
            raise LangGraphUnavailableError(
                "LangGraph is not installed. Install with `pip install 'aimo3[agentic]'`."
            ) from _LANGGRAPH_IMPORT_ERROR

        super().__init__(client, config=config, sandbox_policy=sandbox_policy)
        self.langgraph_config = langgraph_config or LangGraphConfig(
            max_followup_rounds=max(0, int(self.config.agentic_tool_rounds))
        )
        self._graph = self._build_graph()

    def solve(self, problem_text: str, problem_id: Any = None) -> SolveResult:
        final_state = self._graph.invoke(
            {
                "problem_id": problem_id,
                "problem_text": problem_text,
            }
        )

        profile = final_state["profile"]
        modulus = final_state["modulus"]
        candidates = list(final_state.get("candidates") or [])

        predicted = self._select_final_answer(final_state, problem_text=problem_text, modulus=modulus)
        return SolveResult(
            problem_id=problem_id,
            predicted_answer=int(predicted),
            modulus=modulus,
            profile=profile,
            candidates=candidates,
        )

    def _build_graph(self):
        builder = StateGraph(dict)
        builder.add_node("bootstrap", self._node_bootstrap)
        builder.add_node("draft", self._node_draft)
        builder.add_node("tools", self._node_tools)
        builder.add_node("followup", self._node_followup)
        builder.add_node("commit", self._node_commit)

        builder.add_edge(START, "bootstrap")
        builder.add_edge("bootstrap", "draft")
        builder.add_edge("draft", "tools")
        builder.add_conditional_edges(
            "tools",
            self._route_after_tools,
            {
                "followup": "followup",
                "commit": "commit",
            },
        )
        builder.add_edge("followup", "draft")
        builder.add_conditional_edges(
            "commit",
            self._route_after_commit,
            {
                "draft": "draft",
                "end": END,
            },
        )
        return builder.compile()

    def _node_bootstrap(self, state: _GraphState) -> _GraphState:
        problem_text = str(state.get("problem_text") or "")
        profile = estimate_problem_profile(problem_text)
        modulus = parse_modulus(problem_text)

        attempts_limit, tokens_limit = self._attempt_and_token_limits(profile)
        deadline_ts = None
        if self.config.per_problem_time_sec > 0:
            deadline_ts = time.time() + float(self.config.per_problem_time_sec)

        return {
            "problem_id": state.get("problem_id"),
            "problem_text": problem_text,
            "profile": profile,
            "modulus": modulus,
            "attempts_limit": attempts_limit,
            "tokens_limit": tokens_limit,
            "deadline_ts": deadline_ts,
            "attempt_index": 0,
            "round_index": 0,
            "current_temperature": 0.0,
            "current_response": "",
            "current_generation_error": None,
            "current_answer": None,
            "current_answer_source": "none",
            "current_code_answers": [],
            "current_sandbox_errors": [],
            "current_code_blocks": 0,
            "current_tool_observation": "",
            "stateful_prefix": "",
            "weighted_answers": [],
            "weighted_weights": [],
            "candidates": [],
            "done": False,
        }

    def _node_draft(self, state: _GraphState) -> _GraphState:
        attempt_index = int(state["attempt_index"])
        round_index = int(state["round_index"])
        profile = state["profile"]
        modulus = state["modulus"]
        problem_text = state["problem_text"]

        temperature = self._temperature_for(attempt_index=attempt_index, round_index=round_index)
        max_tokens = self._max_tokens_for(round_index=round_index, tokens_limit=int(state["tokens_limit"]))

        if round_index <= 0:
            prompt = build_prompt(
                problem_text,
                attempt_index=attempt_index + 1,
                modulus=modulus,
                profile=profile,
                hard_mode=bool(self.config.hard_mode),
            )
        else:
            prompt = build_agent_followup_prompt(
                problem_text,
                previous_response=state.get("current_response") or "",
                tool_observation=state.get("current_tool_observation") or "No tool output available.",
                modulus=modulus,
                profile=profile,
                round_index=round_index - 1,
            )

        generation_error = None
        try:
            response_text = self.client.generate(
                system_prompt=prompt.system,
                user_prompt=prompt.user,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            parsed = parse_answer_from_text(response_text, modulus=modulus)
        except Exception as exc:  # pragma: no cover - exercised in integration runs
            response_text = ""
            parsed = parse_answer_from_text("", modulus=modulus)
            generation_error = str(exc)[:400]

        next_state = dict(state)
        next_state.update(
            {
                "current_temperature": temperature,
                "current_response": response_text,
                "current_generation_error": generation_error,
                "current_answer": parsed.answer,
                "current_answer_source": parsed.source,
            "current_code_answers": [],
            "current_sandbox_errors": [],
            "current_code_blocks": 0,
            "current_tool_observation": "",
            }
        )
        return next_state

    def _node_tools(self, state: _GraphState) -> _GraphState:
        response_text = state.get("current_response") or ""
        modulus = state.get("modulus")
        generation_error = state.get("current_generation_error")

        if generation_error:
            next_state = dict(state)
            next_state.update(
                {
                    "current_code_answers": [],
                    "current_sandbox_errors": [f"generation_error: {generation_error}"],
                    "current_code_blocks": 0,
                    "current_tool_observation": "Generation failed; skipping tool execution.",
                }
            )
            return next_state

        blocks = extract_python_blocks(response_text)
        selected_blocks = blocks[: max(1, int(self.config.max_code_blocks_per_attempt))]
        code_answers, sandbox_errors, prefix, observations = self._execute_code_blocks(
            selected_blocks,
            modulus=modulus,
            stateful_prefix=state.get("stateful_prefix") or "",
        )

        observation_text = self._tool_observation_text(
            selected_blocks=selected_blocks,
            code_answers=code_answers,
            sandbox_errors=sandbox_errors,
            observations=observations,
        )

        next_state = dict(state)
        next_state.update(
            {
            "current_code_answers": code_answers,
            "current_sandbox_errors": sandbox_errors,
            "current_code_blocks": len(selected_blocks),
            "stateful_prefix": prefix,
            "current_tool_observation": observation_text,
            }
        )
        return next_state

    def _node_followup(self, state: _GraphState) -> _GraphState:
        next_state = dict(state)
        next_state["round_index"] = int(state["round_index"]) + 1
        return next_state

    def _node_commit(self, state: _GraphState) -> _GraphState:
        attempt_index = int(state["attempt_index"])
        round_index = int(state["round_index"])

        response_text = state.get("current_response") or ""
        parsed_answer = state.get("current_answer")
        parsed_source = str(state.get("current_answer_source") or "none")

        code_answers = [int(v) for v in (state.get("current_code_answers") or [])]
        sandbox_errors = [str(x) for x in (state.get("current_sandbox_errors") or [])]
        code_blocks = int(state.get("current_code_blocks") or 0)

        code_vote = Counter(code_answers).most_common(1)[0][0] if code_answers else None
        answer = parsed_answer
        answer_source = parsed_source
        code_verified = False
        generation_error = state.get("current_generation_error")

        if code_vote is not None and parsed_answer is not None and int(code_vote) == int(parsed_answer):
            answer = int(code_vote)
            answer_source = "code_verified"
            code_verified = True
        elif code_vote is not None:
            answer = int(code_vote)
            answer_source = "code_majority"

        candidate = Candidate(
            attempt=attempt_index + 1,
            temperature=float(state.get("current_temperature") or 0.0),
            response_text=response_text,
            answer=answer,
            answer_source=answer_source,
            modulus=state.get("modulus"),
            category=state["profile"].category,
            archetype=state["profile"].archetype,
            complexity=state["profile"].complexity,
            stage="agent" if round_index > 0 else "initial",
            code_blocks=code_blocks,
            code_verified=code_verified,
            agent_rounds=round_index,
            code_answers=code_answers,
            sandbox_errors=sandbox_errors,
            generation_error=str(generation_error) if generation_error else None,
            answer_in_problem=self._answer_is_in_problem(answer, state["problem_text"]),
        )

        candidates = list(state.get("candidates") or [])
        candidates.append(candidate)

        weighted_answers = list(state.get("weighted_answers") or [])
        weighted_weights = list(state.get("weighted_weights") or [])

        self._accumulate_weighted(
            weighted_answers=weighted_answers,
            weighted_weights=weighted_weights,
            answer=answer,
            answer_source=answer_source,
            profile=state["profile"],
        )
        for code_answer in code_answers:
            weighted_answers.append(int(code_answer))
            weighted_weights.append(1.35)

        next_attempt = attempt_index + 1
        done = self._should_stop_after_commit(
            attempt_index=next_attempt,
            attempts_limit=int(state["attempts_limit"]),
            weighted_answers=weighted_answers,
            weighted_weights=weighted_weights,
            deadline_ts=state.get("deadline_ts"),
        )

        next_state = dict(state)
        next_state.update(
            {
            "attempt_index": next_attempt,
            "round_index": 0,
            "current_response": "",
            "current_generation_error": None,
            "current_answer": None,
            "current_answer_source": "none",
            "current_code_answers": [],
            "current_sandbox_errors": [],
            "current_code_blocks": 0,
            "current_tool_observation": "",
            "candidates": candidates,
            "weighted_answers": weighted_answers,
            "weighted_weights": weighted_weights,
            "done": done,
            }
        )
        return next_state

    def _route_after_tools(self, state: _GraphState) -> str:
        if self._should_followup(state):
            return "followup"
        return "commit"

    def _route_after_commit(self, state: _GraphState) -> str:
        return "end" if bool(state.get("done")) else "draft"

    def _attempt_and_token_limits(self, profile: ProblemProfile) -> tuple[int, int]:
        attempts = max(1, int(self.config.attempts))
        max_tokens = max(128, int(self.config.max_tokens))

        hard_problem = profile.complexity == "hard"
        if self.config.hard_mode or (self.config.adaptive_complexity and hard_problem):
            attempts = max(attempts, int(self.config.hard_attempts))
            max_tokens = max(max_tokens, int(self.config.hard_max_tokens))

        return attempts, max_tokens

    def _temperature_for(self, *, attempt_index: int, round_index: int) -> float:
        temperatures = self.config.temperatures or (0.2,)
        idx = (attempt_index + round_index) % len(temperatures)
        base = float(temperatures[idx])
        return min(0.95, max(0.01, base + 0.03 * round_index))

    def _max_tokens_for(self, *, round_index: int, tokens_limit: int) -> int:
        if round_index <= 0:
            return tokens_limit
        reduced = int(tokens_limit * 0.7)
        return max(196, reduced)

    def _should_followup(self, state: _GraphState) -> bool:
        if self.langgraph_config.max_followup_rounds <= 0:
            return False

        if self._is_deadline_near(state.get("deadline_ts")):
            return False

        round_index = int(state.get("round_index") or 0)
        if round_index >= self.langgraph_config.max_followup_rounds:
            return False

        answer = state.get("current_answer")
        answer_source = str(state.get("current_answer_source") or "none")
        code_answers = state.get("current_code_answers") or []
        sandbox_errors = state.get("current_sandbox_errors") or []
        if state.get("current_generation_error"):
            return False

        if answer is None:
            return True

        if (
            self.config.force_tool_round_for_unverified
            and state["profile"].complexity in {"medium", "hard"}
            and round_index < 1
            and answer_source not in self.langgraph_config.followup_answer_sources
        ):
            return True

        if code_answers:
            return True

        if sandbox_errors and answer_source not in self.langgraph_config.followup_answer_sources:
            return True

        return False

    def _execute_code_blocks(
        self,
        blocks: list[str],
        *,
        modulus: int | None,
        stateful_prefix: str,
    ) -> tuple[list[int], list[str], str, list[str]]:
        if not blocks:
            return [], [], stateful_prefix, []

        answers: list[int] = []
        errors: list[str] = []
        observations: list[str] = []

        if not self.config.agentic_stateful_python and self.config.parallel_code_workers > 1:
            worker_count = max(1, min(int(self.config.parallel_code_workers), len(blocks)))
            runner = partial(execute_python, policy=self.sandbox_policy)
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                runs = list(pool.map(runner, blocks))
            for idx, run in enumerate(runs):
                if run.success:
                    if run.stdout.strip():
                        observations.append(f"block#{idx + 1}: {run.stdout.strip()[:260]}")
                    parsed = parse_integer_from_stdout(run.stdout, modulus=modulus)
                    if parsed is not None:
                        answers.append(int(parsed))
                else:
                    errors.append(run.error or "code execution failed")
            return answers, errors, stateful_prefix, observations

        prefix = stateful_prefix
        for idx, block in enumerate(blocks):
            executable = block
            if self.config.agentic_stateful_python and prefix:
                executable = f"{prefix}\n\n{block}".strip()

            run = execute_python(executable, policy=self.sandbox_policy)
            if run.success:
                if run.stdout.strip():
                    observations.append(f"block#{idx + 1}: {run.stdout.strip()[:260]}")
                parsed = parse_integer_from_stdout(run.stdout, modulus=modulus)
                if parsed is not None:
                    answers.append(int(parsed))
                if self.config.agentic_stateful_python:
                    prefix = f"{prefix}\n\n{block}".strip()
                    limit = max(4_000, int(self.config.agentic_state_chars))
                    if len(prefix) > limit:
                        prefix = prefix[-limit:]
            else:
                errors.append(run.error or "code execution failed")

        return answers, errors, prefix, observations

    def _tool_observation_text(
        self,
        *,
        selected_blocks: list[str],
        code_answers: list[int],
        sandbox_errors: list[str],
        observations: list[str],
    ) -> str:
        max_chars = max(220, int(self.config.agentic_observation_chars))
        parts = [f"Code blocks executed: {len(selected_blocks)}"]

        if code_answers:
            parts.append(f"Extracted numeric outputs: {code_answers[:6]}")
        if observations:
            parts.append("Raw snippets: " + " | ".join(observations[:4]))
        if sandbox_errors:
            parts.append(f"Sandbox errors: {sandbox_errors[:3]}")
        if not code_answers and not sandbox_errors:
            parts.append("No decisive numeric signal from tool execution.")

        text = "\n".join(parts)
        return text[:max_chars]

    def _answer_is_in_problem(self, answer: int | None, problem_text: str) -> bool:
        if answer is None:
            return False
        values = {int(v) for v in INTEGER_RE.findall(problem_text)}
        return int(answer) in values

    def _accumulate_weighted(
        self,
        *,
        weighted_answers: list[int],
        weighted_weights: list[float],
        answer: int | None,
        answer_source: str,
        profile: ProblemProfile,
    ) -> None:
        if answer is None:
            return

        weights = {
            "code_verified": 2.3,
            "code_majority": 1.85,
            "final_answer_tag": 1.5,
            "boxed": 1.35,
            "answer_line": 0.95,
            "plain_integer": 0.75,
            "none": 0.5,
        }
        weight = weights.get(answer_source, 0.8)

        if profile.complexity == "hard" and answer_source in {"code_verified", "code_majority"}:
            weight += 0.2

        weighted_answers.append(int(answer))
        weighted_weights.append(float(weight))

    def _should_stop_after_commit(
        self,
        *,
        attempt_index: int,
        attempts_limit: int,
        weighted_answers: list[int],
        weighted_weights: list[float],
        deadline_ts: float | None,
    ) -> bool:
        if attempt_index >= attempts_limit:
            return True

        if self._is_deadline_near(deadline_ts):
            return True

        if not weighted_answers:
            return False

        if self.config.force_full_problem_time and self.config.per_problem_time_sec > 0:
            return False

        counts = Counter(weighted_answers)
        top_answer, top_count = counts.most_common(1)[0]
        total = max(1, len(weighted_answers))
        support_ratio = top_count / total

        if attempt_index < max(1, int(self.config.early_stop_attempt)):
            return False

        min_consensus = max(1, int(self.config.min_consensus))
        if top_count < min_consensus:
            return False

        if support_ratio >= float(self.config.early_stop_ratio):
            return True

        # High weighted confidence may still justify early stop.
        selected = select_weighted_mode(weighted_answers, weighted_weights)
        if selected is None:
            return False

        selected_weight = sum(
            w for a, w in zip(weighted_answers, weighted_weights) if int(a) == int(selected)
        )
        total_weight = max(1e-6, float(sum(weighted_weights)))
        return (selected_weight / total_weight) >= 0.7

    def _is_deadline_near(self, deadline_ts: float | None) -> bool:
        if deadline_ts is None:
            return False
        reserve = max(0, int(self.config.min_time_for_stage_sec))
        return (deadline_ts - time.time()) <= reserve

    def _select_final_answer(
        self,
        state: _GraphState,
        *,
        problem_text: str,
        modulus: int | None,
    ) -> int:
        weighted_answers = [int(v) for v in (state.get("weighted_answers") or [])]
        weighted_weights = [float(v) for v in (state.get("weighted_weights") or [])]

        selected = select_weighted_mode(weighted_answers, weighted_weights)
        if selected is not None:
            normalized = normalize_answer(int(selected), modulus=modulus)
            if normalized is not None:
                return int(normalized)

        candidates = list(state.get("candidates") or [])
        valid_candidates = [c for c in candidates if c.answer is not None]
        if valid_candidates:
            best = max(valid_candidates, key=lambda c: c.score)
            return int(best.answer)  # type: ignore[arg-type]

        return self._deterministic_fallback(problem_text=problem_text, modulus=modulus)

    def _deterministic_fallback(self, *, problem_text: str, modulus: int | None) -> int:
        values = [int(v) for v in INTEGER_RE.findall(problem_text)]
        base = sum((i + 1) * val for i, val in enumerate(values[:40]))
        text_hash = sum((i + 7) * ord(ch) for i, ch in enumerate(problem_text[:1200]))

        mod = int(modulus) if modulus is not None else 100_000
        if mod <= 0:
            mod = 100_000

        raw = (3 * base + 11 * text_hash + int(self.config.default_answer) + 7919) % 100_000
        answer = raw % mod
        if mod > 3 and answer in {0, 1}:
            answer = (answer + 2) % mod
        return int(answer)
