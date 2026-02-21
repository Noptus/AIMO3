"""Parsing utilities for AIMO3 prompts and model outputs."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER\s*:\s*([-+]?\d+)", flags=re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
INTEGER_RE = re.compile(r"(?<!\d)([-+]?\d{1,12})(?!\d)")
FULL_INTEGER_RE = re.compile(r"^\s*([-+]?\d+)\s*$")
RESULT_JSON_RE = re.compile(r"RESULT_JSON\s*:\s*(\{[\s\S]{0,400}\})", flags=re.IGNORECASE)
ANSWER_LINE_HINT_RE = re.compile(
    r"(?:final\s+answer|answer\s*(?:is|=|:)|therefore.*answer|thus.*answer|hence.*answer)",
    flags=re.IGNORECASE,
)

MOD_PATTERNS = [
    re.compile(
        r"remainder\s+when[\s\S]{0,220}?divided\s+by\s*\$([^$]{1,48})\$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:mod(?:ulo)?|modulus)\s*(?:is|=|of)?\s*\$([^$]{1,48})\$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"remainder\s+when[\s\S]{0,220}?divided\s+by\s*([0-9][0-9\^\{\}\(\)\+\-\*/\s]{0,32})",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:mod(?:ulo)?|modulus)\s*(?:is|=|of)?\s*([0-9][0-9\^\{\}\(\)\+\-\*/\s]{0,32})",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\bmod\s*(\d{2,6})\b", flags=re.IGNORECASE),
]

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class AnswerParse:
    answer: int | None
    source: str


@dataclass(frozen=True)
class StructuredParse:
    answer: int | None
    method: str
    independent_check_passed: bool
    source: str


def _safe_eval_int(expr: str) -> int | None:
    """Evaluate a simple integer arithmetic expression safely."""

    expr = _normalize_expr(expr)

    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Load,
        ast.ParenExpr if hasattr(ast, "ParenExpr") else ast.Expr,
    )

    for child in ast.walk(node):
        if not isinstance(child, allowed_nodes):
            return None
        if isinstance(child, ast.Constant) and not isinstance(child.value, (int, float)):
            return None

    try:
        value = eval(compile(node, "<expr>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return None

    if not isinstance(value, (int, float)):
        return None

    if isinstance(value, float):
        if abs(value - round(value)) > 1e-9:
            return None
        value = int(round(value))

    return int(value)


def _normalize_expr(expr: str) -> str:
    normalized = expr.strip()
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\cdot", "*").replace("\\times", "*")
    normalized = normalized.replace("{", "(").replace("}", ")")
    normalized = normalized.replace("^", "**")
    normalized = normalized.replace("−", "-")
    normalized = re.sub(r"[^0-9\+\-\*/\(\)\s]", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def parse_modulus(problem_text: str) -> int | None:
    """Try to extract the modulus required by the problem statement."""

    candidates: list[tuple[int, int]] = []
    for pattern in MOD_PATTERNS:
        for match in pattern.finditer(problem_text):
            candidate = match.group(1).strip().rstrip(".,;:?)")
            value = _safe_eval_int(candidate)
            if value is None:
                continue
            if 2 <= value <= 1_000_000:
                candidates.append((match.start(), value))

    if candidates:
        # Prefer the final explicit modulus mention in the problem text.
        candidates.sort(key=lambda item: item[0])
        return candidates[-1][1]

    return None


def normalize_answer(value: int | str, modulus: int | None = None) -> int | None:
    """Normalize a raw integer answer into AIMO3 range."""

    raw: int | None
    if isinstance(value, int):
        raw = value
    else:
        raw = _safe_eval_int(value.strip())

    if raw is None:
        return None

    if modulus:
        return raw % modulus

    if 0 <= raw <= 99_999:
        return raw

    return raw % 100_000


def parse_structured_result(text: str, modulus: int | None = None) -> StructuredParse:
    """Parse optional structured output of the form RESULT_JSON: {...}."""

    method = ""
    independent = False

    def _coerce_from_payload(payload: dict[str, object]) -> StructuredParse:
        raw_answer = payload.get("answer")
        answer: int | None = None
        if isinstance(raw_answer, int):
            answer = normalize_answer(raw_answer, modulus=modulus)
        elif isinstance(raw_answer, str):
            answer = normalize_answer(raw_answer, modulus=modulus)

        raw_method = payload.get("method")
        parsed_method = raw_method.strip() if isinstance(raw_method, str) else ""

        raw_independent = payload.get("independent_check_passed")
        parsed_independent = bool(raw_independent) if isinstance(raw_independent, bool) else False
        return StructuredParse(
            answer=answer,
            method=parsed_method,
            independent_check_passed=parsed_independent,
            source="structured_json",
        )

    # Preferred explicit marker.
    for match in RESULT_JSON_RE.finditer(text):
        chunk = match.group(1).strip()
        try:
            payload = json.loads(chunk)
        except Exception:
            continue
        if isinstance(payload, dict):
            return _coerce_from_payload(payload)

    # Conservative fallback: inspect short JSON-looking lines with answer key.
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "answer" not in line.lower():
            continue
        if not (line.startswith("{") and line.endswith("}")):
            continue
        if len(line) > 400:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return _coerce_from_payload(payload)

    return StructuredParse(
        answer=None,
        method=method,
        independent_check_passed=independent,
        source="none",
    )


def parse_answer_from_text(text: str, modulus: int | None = None) -> AnswerParse:
    """Extract a final numeric answer from model output."""

    structured = parse_structured_result(text, modulus=modulus)
    if structured.answer is not None:
        return AnswerParse(answer=structured.answer, source=structured.source)

    match = FINAL_ANSWER_RE.search(text)
    if match:
        value = normalize_answer(match.group(1), modulus=modulus)
        return AnswerParse(answer=value, source="final_answer_tag")

    boxed = BOXED_RE.findall(text)
    if boxed:
        value = normalize_answer(boxed[-1], modulus=modulus)
        return AnswerParse(answer=value, source="boxed")

    final_line_candidates = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and ANSWER_LINE_HINT_RE.search(line)
    ]
    for line in reversed(final_line_candidates):
        ints = INTEGER_RE.findall(line)
        if ints:
            value = normalize_answer(ints[-1], modulus=modulus)
            return AnswerParse(answer=value, source="answer_line")

    # Conservative fallback: accept only plain integer outputs.
    full_match = FULL_INTEGER_RE.match(text)
    if full_match:
        value = normalize_answer(full_match.group(1), modulus=modulus)
        return AnswerParse(answer=value, source="plain_integer")

    return AnswerParse(answer=None, source="none")


def extract_python_blocks(text: str) -> list[str]:
    """Return fenced python code blocks."""

    return [block.strip() for block in CODE_BLOCK_RE.findall(text) if block.strip()]


def select_weighted_mode(values: Iterable[int], weights: Iterable[float]) -> int | None:
    """Weighted mode with deterministic tie-breaks."""

    weighted_scores: dict[int, float] = {}
    counts: Counter[int] = Counter()

    for value, weight in zip(values, weights):
        weighted_scores[value] = weighted_scores.get(value, 0.0) + max(weight, 0.01)
        counts[value] += 1

    if not weighted_scores:
        return None

    ranked = sorted(
        weighted_scores.items(),
        key=lambda item: (item[1], counts[item[0]], item[0]),
        reverse=True,
    )
    return ranked[0][0]


def parse_integer_from_stdout(stdout: str, modulus: int | None = None) -> int | None:
    """Extract the last integer from tool output."""

    ints = INTEGER_RE.findall(stdout)
    if not ints:
        return None
    return normalize_answer(ints[-1], modulus=modulus)
