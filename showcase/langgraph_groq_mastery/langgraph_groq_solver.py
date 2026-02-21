"""LangGraph Groq-first showcase solver for AIMO-style problems.

This module is intentionally separate from the production harness.
It demonstrates a stronger LangGraph workflow with:
- typed state and explicit nodes,
- deterministic pattern tools,
- parallel candidate generation,
- python sandbox verification,
- verifier arbitration,
- refinement loops based on confidence.
"""
# ruff: noqa: UP045

from __future__ import annotations

import json
import math
import operator
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal, Optional, TypedDict

import requests
from langgraph.graph import END, START, StateGraph

from aimo3.parsing import (
    extract_python_blocks,
    normalize_answer,
    parse_modulus,
    parse_structured_result,
)
from aimo3.sandbox import SandboxPolicy, execute_python

Backend = Literal["groq", "litellm"]

_FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER\s*:\s*([-+]?\d+)", flags=re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_INT_LINE_RE = re.compile(r"^\s*([-+]?\d+)\s*$")
_RESULT_JSON_RE = re.compile(r"RESULT_JSON\s*:\s*(\{[\s\S]{0,500}\})", flags=re.IGNORECASE)
_FINAL_LINE_RE = re.compile(r"^(?:final|answer)\s*[:=]\s*([-+]?\d+)\s*$", flags=re.IGNORECASE)

_POWER_MIX_RE = re.compile(
    (
        r"Let\s+S\s*=\s*([0-9]+)\^([0-9]+)\s*"
        r"\+\s*([0-9]+)\^([0-9]+)\s*"
        r"-\s*([0-9]+)\^([0-9]+)\s*"
        r"\+\s*([0-9]+)\^([0-9]+)"
        r"[\s\S]{0,120}?modulo\s+([0-9]+)"
    ),
    flags=re.IGNORECASE,
)

_CRT_CONGRUENCE_RE = re.compile(
    r"x\s*(?:\u2261|\\equiv)\s*([0-9]+)\s*\(\s*mod(?:ulo)?\s*([0-9]+)\s*\)",
    flags=re.IGNORECASE,
)

_CRT_QUADRATIC_RE = re.compile(
    (
        r"Compute\s+x\^2\s*([+-])\s*([0-9]+)\s*x\s*([+-])\s*([0-9]+)"
        r"\s+modulo\s+([0-9]+)"
    ),
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class AttemptSpec:
    index: int
    strategy: str
    temperature: float
    max_tokens: int
    code_first: bool
    followup_rounds: int


@dataclass
class AttemptResult:
    attempt_index: int
    strategy: str
    temperature: float
    response_text: str
    answer: Optional[int]
    answer_source: str
    tool_answers: list[int]
    tool_errors: list[str]
    code_blocks: int
    independent_check_passed: bool
    code_verified: bool
    method: str
    generation_error: Optional[str]
    stage: str = "attempt"

    @property
    def score(self) -> float:
        if self.generation_error:
            return 0.01

        base = 0.2
        if self.answer is not None:
            base += 1.0

        if self.answer_source == "deterministic_pattern":
            base += 2.8
        elif self.answer_source == "program_synthesis":
            base += 2.2
        elif self.answer_source == "structured_json":
            base += 0.85
        elif self.answer_source in {"final_answer_tag", "boxed"}:
            base += 0.55
        elif self.answer_source == "tool_majority":
            base += 0.3

        if self.code_verified:
            base += 2.1
        elif self.tool_answers:
            base += 0.55
        if self.independent_check_passed:
            base += 0.5

        if self.answer in {0, 1} and not (self.code_verified or self.independent_check_passed):
            base -= 1.2

        if "code-first" in self.strategy and self.code_blocks == 0:
            base -= 0.35

        if self.answer_source == "none":
            base -= 0.8

        base -= min(0.6, 0.12 * len(self.tool_errors))
        return max(base, 0.01)


@dataclass
class SolveOutcome:
    problem_id: str
    answer: int
    modulus: Optional[int]
    candidates: list[dict[str, Any]]
    verifier_votes: list[int]
    final_reason: str
    elapsed_sec: float


class ShowcaseState(TypedDict, total=False):
    problem_id: str
    problem_text: str
    model: str
    backend: Backend
    max_attempt_workers: int
    time_budget_sec: int
    started_at: float
    modulus: Optional[int]
    category: str
    complexity: str
    attempt_plan: list[dict[str, Any]]
    verify_pool: list[int]
    evidence_lines: list[str]
    candidates: Annotated[list[dict[str, Any]], operator.add]
    verifier_votes: Annotated[list[int], operator.add]
    logs: Annotated[list[str], operator.add]
    refine_round: int
    final_answer: Optional[int]
    final_reason: str
    top_margin: float
    top_answer_support: int
    done: bool


def _categorize(problem_text: str) -> tuple[str, str]:
    text = problem_text.lower()
    category = "general"
    if re.search(r"triangle|circle|angle|circum|incircle|perpendicular|parallel", text):
        category = "geometry"
    elif re.search(r"prime|divisor|gcd|lcm|mod|remainder|coprime", text):
        category = "number_theory"
    elif re.search(
        r"arrange|count|ways|permutation|combination|partition|tournament|rectangle", text
    ):
        category = "combinatorics"
    elif re.search(r"function|equation|polynomial|recurrence|root", text):
        category = "algebra"

    complexity = "easy"
    score = 0
    if len(problem_text) > 500:
        score += 1
    if len(problem_text) > 900:
        score += 1
    if re.search(r"\d+!|largest possible|for all|across all|\\lfloor|floor\(", text):
        score += 1
    if re.search(r"\d{4,}", text):
        score += 1

    if score >= 3:
        complexity = "hard"
    elif score >= 2:
        complexity = "medium"
    return category, complexity


def _safe_json_loads(chunk: str) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(chunk)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _crt_smallest_nonnegative(residues: list[int], moduli: list[int]) -> Optional[int]:
    if not residues or len(residues) != len(moduli):
        return None

    x = 0
    current_mod = 1
    for ai, mi in zip(residues, moduli):
        if mi <= 0:
            return None
        if math.gcd(current_mod, mi) != 1:
            return None
        try:
            inv = pow(current_mod, -1, mi)
        except ValueError:
            return None

        t = ((ai - x) % mi) * inv % mi
        x = x + current_mod * t
        current_mod *= mi

    return x % current_mod


def _parse_strict_answer(
    text: str, *, modulus: Optional[int]
) -> tuple[Optional[int], str, str, bool]:
    structured = parse_structured_result(text, modulus=modulus)
    if structured.answer is not None:
        return (
            int(structured.answer),
            "structured_json",
            structured.method,
            bool(structured.independent_check_passed),
        )

    final_match = _FINAL_ANSWER_RE.search(text)
    if final_match:
        value = normalize_answer(final_match.group(1), modulus=modulus)
        if value is not None:
            return int(value), "final_answer_tag", "", False

    boxed = _BOXED_RE.findall(text)
    if boxed:
        value = normalize_answer(boxed[-1], modulus=modulus)
        if value is not None:
            return int(value), "boxed", "", False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and _INT_LINE_RE.match(lines[-1]):
        value = normalize_answer(lines[-1], modulus=modulus)
        if value is not None:
            return int(value), "plain_last_line", "", False

    return None, "none", "", False


def solve_deterministic_pattern(
    problem_text: str, *, modulus: Optional[int] = None
) -> tuple[Optional[int], str]:
    compact = " ".join(problem_text.split())

    # Pattern 1: direct power mix (used in several AIMO-like synthetic tasks).
    power_match = _POWER_MIX_RE.search(compact)
    if power_match:
        a, e1, b, e2, c, e3, d, e4, mod = [int(power_match.group(i)) for i in range(1, 10)]
        if mod > 0:
            answer = (pow(a, e1, mod) + pow(b, e2, mod) - pow(c, e3, mod) + pow(d, e4, mod)) % mod
            normalized = normalize_answer(answer, modulus=modulus or mod)
            if normalized is not None:
                return int(normalized), "power_mix_closed_form"

    # Pattern 2: CRT with 3 congruences + quadratic expression in x.
    congruences = _CRT_CONGRUENCE_RE.findall(problem_text)
    quadratic = _CRT_QUADRATIC_RE.search(compact)
    if len(congruences) >= 3 and quadratic:
        residues = [int(congruences[i][0]) for i in range(3)]
        moduli = [int(congruences[i][1]) for i in range(3)]
        x = _crt_smallest_nonnegative(residues, moduli)
        if x is not None:
            sign_b = -1 if quadratic.group(1) == "-" else 1
            b = sign_b * int(quadratic.group(2))
            sign_c = -1 if quadratic.group(3) == "-" else 1
            c = sign_c * int(quadratic.group(4))
            out_mod = int(quadratic.group(5))
            if out_mod > 0:
                answer = (x * x + b * x + c) % out_mod
                normalized = normalize_answer(answer, modulus=modulus or out_mod)
                if normalized is not None:
                    return int(normalized), "crt_quadratic"

    return None, ""


class GroqChatClient:
    """Groq OpenAI-compatible chat client with retries."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_sec: int = 180,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

    def complete(self, *, system: str, user: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "reasoning_effort": "medium",
            "top_p": 0.9,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for retry in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_sec,
                )
            except requests.RequestException as exc:
                last_error = exc
                if retry < self.max_retries:
                    time.sleep(0.6 * (retry + 1))
                    continue
                raise RuntimeError(f"Groq request failed: {exc}") from exc

            if response.status_code in {408, 409, 429, 500, 502, 503, 504, 520, 522, 523, 524}:
                last_error = RuntimeError(
                    f"Transient HTTP {response.status_code}: {response.text[:260]}"
                )
                if retry < self.max_retries:
                    time.sleep(0.8 * (retry + 1))
                    continue

            if response.status_code >= 400:
                message = response.text[:320]
                code = ""
                try:
                    payload_err = response.json().get("error", {})
                    message = str(payload_err.get("message") or message)
                    code = str(payload_err.get("code") or "")
                except Exception:
                    pass

                # gpt-oss sometimes tries implicit tool calls; force plain-text retry.
                if code.lower() == "tool_use_failed" and retry < self.max_retries:
                    payload["reasoning_effort"] = "low"
                    payload["max_tokens"] = max(256, min(1200, int(payload["max_tokens"])))
                    payload["messages"][0]["content"] = (
                        system
                        + "\n\nHard constraint: never call tools/functions. Plain text only."
                    )
                    time.sleep(0.7 * (retry + 1))
                    continue

                raise RuntimeError(f"Groq error {response.status_code}: {message[:320]}")

            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                last_error = RuntimeError("Groq response had no choices")
                if retry < self.max_retries:
                    time.sleep(0.5 * (retry + 1))
                    continue
                break

            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        txt = item.get("text")
                        if isinstance(txt, str):
                            text_parts.append(txt)
                joined = "\n".join(text_parts).strip()
                if joined:
                    return joined

            reasoning = message.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning

            last_error = RuntimeError("Groq completion content was empty")
            if retry < self.max_retries:
                time.sleep(0.5 * (retry + 1))
                continue

        raise RuntimeError(str(last_error or "Groq completion failed"))


class LiteLLMChatClient:
    """Optional LiteLLM backend for model routing."""

    def __init__(
        self, *, model: str, api_key: str, timeout_sec: int = 180, max_retries: int = 2
    ) -> None:
        try:
            from litellm import completion
        except Exception as exc:
            raise RuntimeError(
                "LiteLLM backend requested but litellm is not installed. "
                "Install with: pip install litellm"
            ) from exc
        self._completion = completion
        self.model = model
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

    def complete(self, *, system: str, user: str, temperature: float, max_tokens: int) -> str:
        last_error: Optional[Exception] = None
        for retry in range(self.max_retries + 1):
            try:
                resp = self._completion(
                    model=self.model,
                    api_key=self.api_key,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout_sec,
                )
                content = resp.choices[0].message.content
                if isinstance(content, str) and content.strip():
                    return content
                raise RuntimeError("LiteLLM content was empty")
            except Exception as exc:
                last_error = exc
                if retry < self.max_retries:
                    time.sleep(0.6 * (retry + 1))
                    continue
        raise RuntimeError(f"LiteLLM completion failed: {last_error}")


class LangGraphGroqShowcaseSolver:
    """LangGraph showcase solver optimized for Groq-first execution."""

    def __init__(
        self,
        *,
        model: str = "openai/gpt-oss-120b",
        backend: Backend = "groq",
        api_key: Optional[str] = None,
        max_attempt_workers: int = 4,
        time_budget_sec: int = 180,
        sandbox_timeout_sec: float = 8.0,
    ) -> None:
        if backend not in {"groq", "litellm"}:
            raise ValueError(f"Unsupported backend: {backend}")

        resolved_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("AIMO_API_KEY")
        if not resolved_key:
            raise RuntimeError("Missing API key. Set GROQ_API_KEY (or pass api_key).")

        self.model = model
        self.backend = backend
        self.max_attempt_workers = max(1, int(max_attempt_workers))
        self.time_budget_sec = max(30, int(time_budget_sec))
        self.sandbox_policy = SandboxPolicy(timeout_sec=float(max(2.0, sandbox_timeout_sec)))

        if backend == "groq":
            self.client = GroqChatClient(model=model, api_key=resolved_key)
        else:
            self.client = LiteLLMChatClient(model=model, api_key=resolved_key)

        self.graph = self._build_graph()

    def solve(self, problem_id: str, problem_text: str) -> SolveOutcome:
        started = time.perf_counter()
        initial: ShowcaseState = {
            "problem_id": str(problem_id),
            "problem_text": problem_text,
            "model": self.model,
            "backend": self.backend,
            "max_attempt_workers": self.max_attempt_workers,
            "time_budget_sec": self.time_budget_sec,
            "started_at": time.time(),
            "refine_round": 0,
            "done": False,
            "candidates": [],
            "verifier_votes": [],
            "logs": [],
        }

        final_state = self.graph.invoke(initial)
        candidates_raw = final_state.get("candidates", [])
        votes = final_state.get("verifier_votes", [])
        answer = int(final_state.get("final_answer", 210) or 210) % 100000
        modulus = final_state.get("modulus")
        elapsed = time.perf_counter() - started

        return SolveOutcome(
            problem_id=str(problem_id),
            answer=answer,
            modulus=modulus,
            candidates=candidates_raw,
            verifier_votes=votes,
            final_reason=str(final_state.get("final_reason") or "aggregate"),
            elapsed_sec=elapsed,
        )

    def _build_graph(self):
        builder = StateGraph(ShowcaseState)

        builder.add_node("bootstrap", self._node_bootstrap)
        builder.add_node("pattern_solver", self._node_pattern_solver)
        builder.add_node("generate_candidates", self._node_generate_candidates)
        builder.add_node("prepare_verifier", self._node_prepare_verifier)
        builder.add_node("run_verifier", self._node_run_verifier)
        builder.add_node("aggregate", self._node_aggregate)
        builder.add_node("refine", self._node_refine)
        builder.add_node("finalize", self._node_finalize)

        builder.add_edge(START, "bootstrap")
        builder.add_edge("bootstrap", "pattern_solver")
        builder.add_edge("pattern_solver", "generate_candidates")
        builder.add_edge("generate_candidates", "prepare_verifier")
        builder.add_conditional_edges(
            "prepare_verifier",
            self._route_after_prepare_verifier,
            {"run_verifier": "run_verifier", "aggregate": "aggregate"},
        )
        builder.add_edge("run_verifier", "aggregate")
        builder.add_conditional_edges(
            "aggregate",
            self._route_after_aggregate,
            {"refine": "refine", "finalize": "finalize"},
        )
        builder.add_edge("refine", "generate_candidates")
        builder.add_edge("finalize", END)

        return builder.compile()

    def _node_bootstrap(self, state: ShowcaseState) -> ShowcaseState:
        problem_text = str(state["problem_text"])
        category, complexity = _categorize(problem_text)
        modulus = parse_modulus(problem_text)
        plan = self._attempt_plan(category=category, complexity=complexity, refined=False)
        return {
            "category": category,
            "complexity": complexity,
            "modulus": modulus,
            "attempt_plan": [asdict(spec) for spec in plan],
            "logs": [
                f"bootstrap category={category} complexity={complexity} modulus={modulus} attempts={len(plan)}"
            ],
        }

    def _node_pattern_solver(self, state: ShowcaseState) -> ShowcaseState:
        problem_text = str(state["problem_text"])
        modulus = state.get("modulus")
        answer, method = solve_deterministic_pattern(problem_text, modulus=modulus)

        if answer is None:
            return {"logs": ["pattern_solver none"]}

        result = AttemptResult(
            attempt_index=-1,
            strategy="deterministic-pattern",
            temperature=0.0,
            response_text="",
            answer=int(answer),
            answer_source="deterministic_pattern",
            tool_answers=[int(answer)],
            tool_errors=[],
            code_blocks=0,
            independent_check_passed=True,
            code_verified=True,
            method=method,
            generation_error=None,
            stage="pattern",
        )
        return {
            "candidates": [self._result_to_dict(result)],
            "logs": [f"pattern_solver hit method={method} answer={answer}"],
        }

    def _node_generate_candidates(self, state: ShowcaseState) -> ShowcaseState:
        plan_raw = state.get("attempt_plan", [])
        if not plan_raw:
            return {"logs": ["generate_candidates skipped: empty plan"]}

        attempts = [AttemptSpec(**item) for item in plan_raw]
        problem_text = state["problem_text"]
        modulus = state.get("modulus")
        deadline = float(state["started_at"]) + float(state["time_budget_sec"])

        results: list[AttemptResult] = []
        logs: list[str] = []

        workers = min(len(attempts), int(state.get("max_attempt_workers", 4)))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(
                    self._run_single_attempt,
                    problem_text=problem_text,
                    modulus=modulus,
                    spec=spec,
                    deadline=deadline,
                    category=str(state.get("category") or "general"),
                    complexity=str(state.get("complexity") or "medium"),
                ): spec
                for spec in attempts
            }

            for future in as_completed(futures):
                spec = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = AttemptResult(
                        attempt_index=spec.index,
                        strategy=spec.strategy,
                        temperature=spec.temperature,
                        response_text="",
                        answer=None,
                        answer_source="none",
                        tool_answers=[],
                        tool_errors=[f"attempt_error: {str(exc)[:220]}"],
                        code_blocks=0,
                        independent_check_passed=False,
                        code_verified=False,
                        method="",
                        generation_error=str(exc)[:320],
                    )

                results.append(result)
                logs.append(
                    "candidate"
                    f" attempt={result.attempt_index} answer={result.answer}"
                    f" source={result.answer_source} score={result.score:.2f}"
                )

        results.sort(key=lambda item: item.attempt_index)
        return {"candidates": [self._result_to_dict(item) for item in results], "logs": logs}

    def _node_prepare_verifier(self, state: ShowcaseState) -> ShowcaseState:
        candidates = [c for c in state.get("candidates", []) if c.get("answer") is not None]
        if not candidates:
            return {
                "verify_pool": [],
                "evidence_lines": [],
                "logs": ["prepare_verifier no candidates"],
            }

        by_answer: dict[int, list[dict[str, Any]]] = {}
        for item in candidates:
            answer = int(item["answer"])
            by_answer.setdefault(answer, []).append(item)

        ranked: list[tuple[int, float]] = []
        for answer, rows in by_answer.items():
            score = sum(self._candidate_score_dict(row) for row in rows)
            score += 0.7 * sum(1 for row in rows if row.get("code_verified"))
            score += 0.4 * sum(1 for row in rows if row.get("independent_check_passed"))
            ranked.append((answer, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        verify_pool = [answer for answer, _ in ranked[:5]]

        evidence_lines: list[str] = []
        for answer in verify_pool:
            rows = by_answer[answer]
            evidence_lines.append(
                f"answer={answer} support={len(rows)} "
                f"verified={sum(1 for row in rows if row.get('code_verified'))} "
                f"independent={sum(1 for row in rows if row.get('independent_check_passed'))} "
                f"sources={[row.get('answer_source') for row in rows[:4]]}"
            )

        return {
            "verify_pool": verify_pool,
            "evidence_lines": evidence_lines,
        }

    def _route_after_prepare_verifier(self, state: ShowcaseState) -> str:
        pool = state.get("verify_pool", [])
        if len(pool) >= 1:
            return "run_verifier"
        return "aggregate"

    def _node_run_verifier(self, state: ShowcaseState) -> ShowcaseState:
        pool = [int(item) for item in state.get("verify_pool", [])]
        if not pool:
            return {"verifier_votes": [], "logs": ["run_verifier skipped: empty pool"]}

        system = (
            "You are a strict olympiad verifier. "
            "Choose exactly one answer from the provided candidates. "
            "Do not use tools or function calls. "
            "Output RESULT_JSON then FINAL_ANSWER only."
        )

        options = "\n".join(f"- {answer}" for answer in pool)
        evidence = "\n".join(state.get("evidence_lines", []))
        votes: list[int] = []
        logs: list[str] = []

        for pass_idx, temperature in enumerate((0.02, 0.08, 0.15), start=1):
            user = (
                f"Problem:\n{state['problem_text']}\n\n"
                f"Candidate answers:\n{options}\n\n"
                f"Evidence:\n{evidence}\n\n"
                "Rules:\n"
                "- Pick exactly one candidate from the list.\n"
                "- Do not invent new values.\n"
                'RESULT_JSON: {"answer": <int>, "method": "verifier", "independent_check_passed": true}\n'
                "FINAL_ANSWER: <integer>\n"
            )
            try:
                response = self.client.complete(
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_tokens=320,
                )
                parsed, _, _, _ = _parse_strict_answer(response, modulus=state.get("modulus"))
                if parsed is not None and int(parsed) in pool:
                    votes.append(int(parsed))
                    logs.append(f"verifier pass={pass_idx} vote={parsed}")
                else:
                    logs.append(f"verifier pass={pass_idx} vote=none")
            except Exception as exc:
                logs.append(f"verifier pass={pass_idx} error={str(exc)[:180]}")

        return {"verifier_votes": votes, "logs": logs}

    def _node_aggregate(self, state: ShowcaseState) -> ShowcaseState:
        candidates = [item for item in state.get("candidates", []) if item.get("answer") is not None]
        if not candidates:
            fallback = 210
            modulus = state.get("modulus")
            if isinstance(modulus, int) and modulus > 0:
                fallback = fallback % modulus
            return {
                "final_answer": int(fallback % 100000),
                "final_reason": "no_candidates_default_210",
                "top_margin": 0.0,
                "top_answer_support": 0,
            }

        scores: dict[int, float] = {}
        support: dict[int, int] = {}
        for item in candidates:
            answer = int(item["answer"])
            scores[answer] = scores.get(answer, 0.0) + self._candidate_score_dict(item)
            support[answer] = support.get(answer, 0) + 1

        votes = [int(v) for v in state.get("verifier_votes", [])]
        for vote in votes:
            scores[vote] = scores.get(vote, 0.0) + 1.6
            support[vote] = support.get(vote, 0) + 1

        ranked = sorted(
            scores.items(),
            key=lambda item: (item[1], support.get(item[0], 0), item[0]),
            reverse=True,
        )

        best_answer, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = float(best_score - second_score)

        if best_answer in {0, 1} and support.get(best_answer, 0) <= 1:
            alternatives = [item for item in ranked if item[0] not in {0, 1}]
            if alternatives:
                best_answer = alternatives[0][0]
                best_score = alternatives[0][1]
                margin = float(best_score - second_score)

        modulus = state.get("modulus")
        if isinstance(modulus, int) and modulus > 0:
            best_answer = int(best_answer % modulus)
        best_answer = int(best_answer % 100000)

        reason = (
            f"aggregate top_score={best_score:.2f} margin={margin:.2f} "
            f"support={support.get(best_answer, 0)}"
        )
        return {
            "final_answer": best_answer,
            "final_reason": reason,
            "top_margin": margin,
            "top_answer_support": int(support.get(best_answer, 0)),
            "logs": [reason],
        }

    def _route_after_aggregate(self, state: ShowcaseState) -> str:
        if bool(state.get("done")):
            return "finalize"

        refine_round = int(state.get("refine_round", 0) or 0)
        answer = int(state.get("final_answer", 0) or 0)
        margin = float(state.get("top_margin", 0.0) or 0.0)
        support = int(state.get("top_answer_support", 0) or 0)

        elapsed = time.time() - float(state["started_at"])
        time_left = max(0.0, float(state["time_budget_sec"]) - elapsed)

        weak_small = answer in {0, 1} and support <= 1
        weak_margin = margin < 0.95
        weak_support = support <= 1

        if refine_round < 2 and time_left > 35 and (weak_small or weak_margin or weak_support):
            return "refine"

        return "finalize"

    def _node_refine(self, state: ShowcaseState) -> ShowcaseState:
        category = str(state.get("category") or "general")
        complexity = str(state.get("complexity") or "medium")
        next_round = int(state.get("refine_round", 0) or 0) + 1
        plan = self._attempt_plan(category=category, complexity=complexity, refined=True)
        return {
            "refine_round": next_round,
            "attempt_plan": [asdict(spec) for spec in plan],
            "logs": [f"refine round={next_round} attempts={len(plan)}"],
        }

    def _node_finalize(self, state: ShowcaseState) -> ShowcaseState:
        return {"done": True}

    def _attempt_plan(self, *, category: str, complexity: str, refined: bool) -> list[AttemptSpec]:
        base_rounds = 2 if complexity in {"medium", "hard"} else 1

        if refined:
            return [
                AttemptSpec(
                    index=100,
                    strategy="code-first-contradiction",
                    temperature=0.18,
                    max_tokens=960,
                    code_first=True,
                    followup_rounds=2,
                ),
                AttemptSpec(
                    index=101,
                    strategy="independent-derivation",
                    temperature=0.24,
                    max_tokens=960,
                    code_first=True,
                    followup_rounds=2,
                ),
                AttemptSpec(
                    index=102,
                    strategy="low-temp-verification",
                    temperature=0.07,
                    max_tokens=700,
                    code_first=True,
                    followup_rounds=1,
                ),
                AttemptSpec(
                    index=103,
                    strategy="structured-json-only",
                    temperature=0.05,
                    max_tokens=520,
                    code_first=False,
                    followup_rounds=1,
                ),
            ]

        strategies: list[tuple[str, float, bool]] = [
            ("proof-first-invariants", 0.08, False),
            ("code-first-check", 0.16, True),
            ("alternate-derivation", 0.24, True),
            ("adversarial-check", 0.33, True),
            ("low-temp-verification", 0.05, True),
            ("modular-structure", 0.14, True),
        ]

        if category == "geometry":
            strategies[0] = ("synthetic-then-analytic", 0.10, False)
            strategies[2] = ("coordinate-crosscheck", 0.20, True)
        elif category == "number_theory":
            strategies[0] = ("residue-and-valuation", 0.08, True)
            strategies[2] = ("factor-structure-check", 0.22, True)

        if complexity == "easy":
            strategies = strategies[:4]

        return [
            AttemptSpec(
                index=i,
                strategy=name,
                temperature=temp,
                max_tokens=1150 if complexity == "hard" else 900,
                code_first=code_first,
                followup_rounds=base_rounds,
            )
            for i, (name, temp, code_first) in enumerate(strategies)
        ]

    def _run_single_attempt(
        self,
        *,
        problem_text: str,
        modulus: Optional[int],
        spec: AttemptSpec,
        deadline: float,
        category: str,
        complexity: str,
    ) -> AttemptResult:
        if time.time() >= deadline:
            return AttemptResult(
                attempt_index=spec.index,
                strategy=spec.strategy,
                temperature=spec.temperature,
                response_text="",
                answer=None,
                answer_source="none",
                tool_answers=[],
                tool_errors=["deadline_exceeded_before_attempt"],
                code_blocks=0,
                independent_check_passed=False,
                code_verified=False,
                method="",
                generation_error="deadline_exceeded_before_attempt",
            )

        system = (
            "You are an IMO-level AIMO solver. "
            "Be concise and exact. "
            "Never call tools/functions; plain text only. "
            "If needed, include one fenced python block that prints a single integer. "
            "Then output RESULT_JSON and FINAL_ANSWER."
        )
        mode_line = "code-first required" if spec.code_first else "proof-first"
        user = (
            f"Category={category}, Complexity={complexity}, Strategy={spec.strategy}, Mode={mode_line}\n"
            f"Problem:\n{problem_text}\n\n"
            "Rules:\n"
            "- Keep reasoning concise (max 12 lines).\n"
            "- Do not output 0/1 unless mathematically forced.\n"
            "- If you use python, print only the candidate integer.\n"
            'RESULT_JSON: {"answer": <int>, "method": "<short>", "independent_check_passed": <true|false>}\n'
            "FINAL_ANSWER: <integer>\n"
        )

        try:
            response = self.client.complete(
                system=system,
                user=user,
                temperature=spec.temperature,
                max_tokens=spec.max_tokens,
            )
            generation_error = None
        except Exception as exc:
            return AttemptResult(
                attempt_index=spec.index,
                strategy=spec.strategy,
                temperature=spec.temperature,
                response_text="",
                answer=None,
                answer_source="none",
                tool_answers=[],
                tool_errors=[f"generation_error: {str(exc)[:220]}"],
                code_blocks=0,
                independent_check_passed=False,
                code_verified=False,
                method="",
                generation_error=str(exc)[:320],
            )

        answer, source, method, independent = _parse_strict_answer(response, modulus=modulus)
        tool_answers, tool_errors, code_blocks = self._run_tool_checks(response, modulus=modulus)

        if answer is None and tool_answers:
            answer = Counter(tool_answers).most_common(1)[0][0]
            source = "tool_majority"

        code_verified = self._is_strong_tool_confirmation(answer, tool_answers)
        independent = independent or bool(tool_answers)

        need_followup = (
            (answer is None)
            or (spec.code_first and code_blocks == 0)
            or (answer in {0, 1} and not (independent or code_verified))
            or (tool_errors and not code_verified)
        )

        current_response = response
        for follow_idx in range(spec.followup_rounds):
            if not need_followup:
                break
            if time.time() >= deadline:
                break

            observation = self._tool_observation(tool_answers, tool_errors)
            follow_user = (
                f"Refine attempt ({follow_idx + 1}).\nProblem:\n{problem_text}\n\n"
                f"Previous response:\n{current_response[-2800:]}\n\n"
                f"Tool observations:\n{observation}\n\n"
                "Fix mistakes. If uncertain, include one python code block that prints one integer.\n"
                'Output RESULT_JSON: {"answer": <int>, "method": "<short>", "independent_check_passed": <true|false>}\n'
                "Then FINAL_ANSWER: <integer>\n"
            )
            try:
                current_response = self.client.complete(
                    system=system,
                    user=follow_user,
                    temperature=min(0.2, spec.temperature),
                    max_tokens=min(820, spec.max_tokens),
                )
            except Exception as exc:
                tool_errors.append(f"followup_generation_error: {str(exc)[:220]}")
                break

            parsed_answer, parsed_source, parsed_method, parsed_independent = _parse_strict_answer(
                current_response, modulus=modulus
            )
            extra_answers, extra_errors, extra_blocks = self._run_tool_checks(
                current_response, modulus=modulus
            )
            tool_answers.extend(extra_answers)
            tool_errors.extend(extra_errors)
            code_blocks += extra_blocks

            if parsed_answer is not None:
                answer = parsed_answer
                source = parsed_source
            elif tool_answers:
                answer = Counter(tool_answers).most_common(1)[0][0]
                source = "tool_majority"

            if parsed_method:
                method = parsed_method
            if parsed_independent:
                independent = True

            code_verified = self._is_strong_tool_confirmation(answer, tool_answers)
            independent = independent or bool(tool_answers)

            need_followup = (
                (answer is None)
                or (spec.code_first and code_blocks == 0)
                or (answer in {0, 1} and not (independent or code_verified))
                or (tool_errors and not code_verified and follow_idx + 1 < spec.followup_rounds)
            )

        if (
            time.time() < deadline - 8
            and ((answer is None) or (answer in {0, 1} and not code_verified) or code_blocks == 0)
        ):
            forced_answer, forced_error = self._force_program_synthesis(
                problem_text=problem_text,
                modulus=modulus,
            )
            if forced_answer is not None:
                tool_answers.append(int(forced_answer))
                if answer is None or (answer in {0, 1} and forced_answer not in {0, 1}):
                    answer = int(forced_answer)
                    source = "program_synthesis"
                independent = True
                code_verified = self._is_strong_tool_confirmation(answer, tool_answers)
            elif forced_error:
                tool_errors.append(f"program_synthesis_error: {forced_error[:220]}")

        if answer is None and tool_answers:
            answer = Counter(tool_answers).most_common(1)[0][0]
            source = "tool_majority"

        if answer is not None:
            normalized = normalize_answer(answer, modulus=modulus)
            answer = int(normalized) if normalized is not None else None

        return AttemptResult(
            attempt_index=spec.index,
            strategy=spec.strategy,
            temperature=spec.temperature,
            response_text=current_response,
            answer=answer,
            answer_source=source,
            tool_answers=tool_answers,
            tool_errors=tool_errors,
            code_blocks=code_blocks,
            independent_check_passed=independent,
            code_verified=code_verified,
            method=method,
            generation_error=generation_error,
        )

    def _run_tool_checks(
        self, response_text: str, *, modulus: Optional[int]
    ) -> tuple[list[int], list[str], int]:
        answers: list[int] = []
        errors: list[str] = []

        blocks = extract_python_blocks(response_text)
        selected_blocks = blocks[:4]

        for block in selected_blocks:
            prepared = self._ensure_print_last_expr(block)
            run = execute_python(prepared, policy=self.sandbox_policy)
            if run.success:
                parsed = self._parse_tool_stdout(run.stdout, modulus=modulus)
                if parsed is not None:
                    answers.append(parsed)
            elif run.error:
                errors.append(run.error)

        return answers, errors, len(selected_blocks)

    def _parse_tool_stdout(self, stdout: str, *, modulus: Optional[int]) -> Optional[int]:
        parsed, _, _, _ = _parse_strict_answer(stdout, modulus=modulus)
        if parsed is not None:
            return int(parsed)

        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if len(lines) == 1 and _INT_LINE_RE.match(lines[0]):
            value = normalize_answer(lines[0], modulus=modulus)
            if value is not None:
                return int(value)

        if lines:
            final_line_match = _FINAL_LINE_RE.match(lines[-1])
            if final_line_match:
                value = normalize_answer(final_line_match.group(1), modulus=modulus)
                if value is not None:
                    return int(value)

        return None

    def _ensure_print_last_expr(self, code: str) -> str:
        lines = code.strip().split("\n")
        if not lines:
            return code

        last_idx = len(lines) - 1
        while last_idx >= 0 and not lines[last_idx].strip():
            last_idx -= 1

        if last_idx < 0:
            return code

        last = lines[last_idx].strip()
        if (
            last.startswith("print(")
            or last.startswith("for ")
            or last.startswith("while ")
            or last.startswith("if ")
            or last.startswith("def ")
            or last.startswith("class ")
            or "=" in last
            or last.startswith("import ")
            or last.startswith("from ")
        ):
            return code

        lines[last_idx] = f"print({last})"
        return "\n".join(lines)

    def _force_program_synthesis(
        self, *, problem_text: str, modulus: Optional[int]
    ) -> tuple[Optional[int], Optional[str]]:
        system = (
            "You are a python synthesis assistant. "
            "Return only one fenced python code block and no other text. "
            "No tool/function calls."
        )
        user = (
            "Write compact Python that computes the final integer answer for this olympiad-style problem.\n"
            "The script must print exactly one integer on the last line.\n"
            "Do not print explanations.\n"
            f"Problem:\n{problem_text}\n"
            f"Expected modulus (if present): {modulus}\n"
            "Output format: one ```python ... ``` block only."
        )

        try:
            response = self.client.complete(system=system, user=user, temperature=0.05, max_tokens=720)
        except Exception as exc:
            return None, str(exc)

        blocks = extract_python_blocks(response)
        if not blocks:
            candidate = response.strip()
            if candidate:
                blocks = [candidate]

        if not blocks:
            return None, "no_python_block"

        prepared = self._ensure_print_last_expr(blocks[0])
        run = execute_python(prepared, policy=self.sandbox_policy)
        if not run.success:
            return None, run.error or "program_execution_failed"

        parsed = self._parse_tool_stdout(run.stdout, modulus=modulus)
        if parsed is None:
            return None, "program_output_not_parseable"

        return int(parsed), None

    def _tool_observation(self, tool_answers: list[int], tool_errors: list[str]) -> str:
        lines: list[str] = []
        if tool_answers:
            lines.append(f"- Parsed integers from python: {tool_answers[:10]}")
        else:
            lines.append("- No integer parsed from python output.")
        if tool_errors:
            lines.append(f"- Tool errors: {tool_errors[:3]}")
        return "\n".join(lines)

    def _is_strong_tool_confirmation(
        self, answer: Optional[int], tool_answers: list[int]
    ) -> bool:
        if answer is None or not tool_answers:
            return False
        counts = Counter(tool_answers)
        top_answer, top_count = counts.most_common(1)[0]
        if int(top_answer) != int(answer):
            return False
        if top_count >= 2:
            return True
        return len(counts) == 1

    def _result_to_dict(self, result: AttemptResult) -> dict[str, Any]:
        payload = asdict(result)
        payload["score"] = float(result.score)
        return payload

    def _candidate_score_dict(self, item: dict[str, Any]) -> float:
        raw = item.get("score")
        if isinstance(raw, (int, float)):
            return float(raw)

        try:
            reconstructed = AttemptResult(
                attempt_index=int(item.get("attempt_index", 0)),
                strategy=str(item.get("strategy", "")),
                temperature=float(item.get("temperature", 0.0)),
                response_text=str(item.get("response_text", "")),
                answer=(None if item.get("answer") is None else int(item.get("answer"))),
                answer_source=str(item.get("answer_source", "none")),
                tool_answers=[int(x) for x in item.get("tool_answers", [])],
                tool_errors=[str(x) for x in item.get("tool_errors", [])],
                code_blocks=int(item.get("code_blocks", 0)),
                independent_check_passed=bool(item.get("independent_check_passed", False)),
                code_verified=bool(item.get("code_verified", False)),
                method=str(item.get("method", "")),
                generation_error=(
                    None
                    if item.get("generation_error") is None
                    else str(item.get("generation_error"))
                ),
                stage=str(item.get("stage", "attempt")),
            )
            return float(reconstructed.score)
        except Exception:
            return 0.01
