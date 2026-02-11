"""Parsing utilities for AIMO3 prompts and model outputs."""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER\s*:\s*([-+]?\d+)", flags=re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
INTEGER_RE = re.compile(r"(?<!\d)([-+]?\d{1,12})(?!\d)")

MOD_PATTERNS = [
    re.compile(
        r"(?:mod(?:ulo)?|modulus)\s*(?:is|=|of)?\s*\$?\s*(10\s*\^\s*\{?\d+\}?|\d{2,6})",
        flags=re.IGNORECASE,
    ),
    re.compile(r"remainder\s+when\s+divided\s+by\s+(\d{2,6})", flags=re.IGNORECASE),
    re.compile(r"mod\s*(\d{2,6})", flags=re.IGNORECASE),
]

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class AnswerParse:
    answer: int | None
    source: str


def _safe_eval_int(expr: str) -> int | None:
    """Evaluate a simple integer arithmetic expression safely."""

    expr = expr.replace("^", "**")

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


def parse_modulus(problem_text: str) -> int | None:
    """Try to extract the modulus required by the problem statement."""

    for pattern in MOD_PATTERNS:
        match = pattern.search(problem_text)
        if not match:
            continue
        candidate = match.group(1)
        value = _safe_eval_int(candidate)
        if value is None:
            continue
        if 2 <= value <= 1_000_000:
            return value

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


def parse_answer_from_text(text: str, modulus: int | None = None) -> AnswerParse:
    """Extract a final numeric answer from model output."""

    match = FINAL_ANSWER_RE.search(text)
    if match:
        value = normalize_answer(match.group(1), modulus=modulus)
        return AnswerParse(answer=value, source="final_answer_tag")

    boxed = BOXED_RE.findall(text)
    if boxed:
        value = normalize_answer(boxed[-1], modulus=modulus)
        return AnswerParse(answer=value, source="boxed")

    final_line_candidates = [
        line.strip() for line in text.splitlines() if line.strip() and ("answer" in line.lower())
    ]
    for line in reversed(final_line_candidates):
        ints = INTEGER_RE.findall(line)
        if ints:
            value = normalize_answer(ints[-1], modulus=modulus)
            return AnswerParse(answer=value, source="answer_line")

    ints = INTEGER_RE.findall(text)
    if ints:
        value = normalize_answer(ints[-1], modulus=modulus)
        return AnswerParse(answer=value, source="last_integer")

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
