"""Deterministic mini-solvers for easy high-confidence sub-cases."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass

from .parsing import normalize_answer


@dataclass(frozen=True)
class MiniSolveResult:
    answer: int
    method: str
    reason: str
    confidence: float = 0.95
    independent_check_passed: bool = True


_INTEGER_EXPR_RE = re.compile(r"^[0-9\+\-\*/\^\(\)\s]+$")
_WHAT_IS_RE = re.compile(r"what\s+is\s+\$?([^$?\n]{1,120})", flags=re.IGNORECASE)
_REMAINDER_RE = re.compile(
    r"remainder\s+when\s+(.{1,120}?)\s+is\s+divided\s+by\s+([0-9\^\{\}\s]{1,24})",
    flags=re.IGNORECASE,
)


def _safe_eval_int(expr: str) -> int | None:
    expr = expr.strip().replace("^", "**")
    expr = expr.replace("{", "(").replace("}", ")")
    if not _INTEGER_EXPR_RE.match(expr):
        return None

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

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if abs(value - round(value)) > 1e-9:
            return None
        value = int(round(value))
    if not isinstance(value, int):
        return None
    return int(value)


def _solve_simple_expression(problem_text: str, modulus: int | None) -> MiniSolveResult | None:
    match = _WHAT_IS_RE.search(problem_text)
    if not match:
        return None

    expr = match.group(1).strip()
    value = _safe_eval_int(expr)
    if value is None:
        return None

    normalized = normalize_answer(value, modulus=modulus)
    if normalized is None:
        return None

    return MiniSolveResult(
        answer=normalized,
        method="mini_simple_expression",
        reason=f"Direct integer expression evaluation: {expr}",
        confidence=0.98,
    )


def _solve_remainder_expression(problem_text: str, modulus: int | None) -> MiniSolveResult | None:
    match = _REMAINDER_RE.search(problem_text)
    if not match:
        return None

    expr = match.group(1).strip()
    mod_expr = match.group(2).strip()
    parsed_mod = _safe_eval_int(mod_expr.replace("^", "**"))
    if parsed_mod is None or parsed_mod <= 0:
        return None

    # Respect explicit statement modulus over extracted one if they differ.
    target_mod = int(parsed_mod)
    value = _safe_eval_int(expr)
    if value is None:
        return None

    answer = value % target_mod
    normalized = normalize_answer(answer, modulus=modulus or target_mod)
    if normalized is None:
        return None

    return MiniSolveResult(
        answer=normalized,
        method="mini_remainder_eval",
        reason=f"Direct remainder computation for expression '{expr}' mod {target_mod}",
        confidence=0.96,
    )


def run_mini_solvers(problem_text: str, modulus: int | None = None) -> list[MiniSolveResult]:
    """Run deterministic mini-solvers and return high-confidence candidates."""

    results: list[MiniSolveResult] = []
    for solver in (_solve_remainder_expression, _solve_simple_expression):
        try:
            solved = solver(problem_text, modulus)
        except Exception:
            solved = None
        if solved is not None:
            results.append(solved)

    dedup: dict[int, MiniSolveResult] = {}
    for item in results:
        previous = dedup.get(item.answer)
        if previous is None or item.confidence > previous.confidence:
            dedup[item.answer] = item

    return sorted(dedup.values(), key=lambda item: item.confidence, reverse=True)
