"""Prompt templates for AIMO3 solving."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str


SYSTEM_PROMPT = """You are an olympiad-level mathematical reasoning assistant.
Solve integer-answer competition problems with rigorous steps.

Execution rules:
1) If arithmetic/combinatorics is heavy, write compact Python in fenced ```python``` blocks.
2) Prefer exact arithmetic and deterministic logic.
3) Respect any modulus requested in the problem statement.
4) End with exactly one line: FINAL_ANSWER: <integer>
"""

STYLE_GUIDE = {
    "algebra": "Prioritize symbolic simplification, invariants, and exact substitutions.",
    "number_theory": "Prioritize modular arithmetic, valuation checks, and divisibility structure.",
    "combinatorics": "Prioritize counting arguments, bijections, and parity constraints.",
    "geometry": "Use analytic transformations if needed; keep expressions exact.",
    "general": "Explore two independent paths before concluding when feasible.",
}


def classify_problem(problem_text: str) -> str:
    """Lightweight router for problem-specific prompting."""

    text = problem_text.lower()

    if re.search(r"triangle|circle|angle|perpendicular|parallel|midpoint", text):
        return "geometry"
    if re.search(r"prime|divisible|mod|remainder|gcd|lcm|integer", text):
        return "number_theory"
    if re.search(r"ways|arrange|permutation|combination|subset|graph|color", text):
        return "combinatorics"
    if re.search(r"polynomial|equation|root|coefficient|function", text):
        return "algebra"

    return "general"


def build_prompt(problem_text: str, *, attempt_index: int, modulus: int | None) -> PromptBundle:
    """Compose a stable, composable solver prompt."""

    category = classify_problem(problem_text)
    style = STYLE_GUIDE.get(category, STYLE_GUIDE["general"])
    modulus_hint = (
        f"Known modulus for final normalization: {modulus}."
        if modulus is not None
        else "No trusted modulus extracted yet. Infer carefully from the problem text."
    )

    user = f"""Problem category hint: {category}
Attempt index: {attempt_index}
{modulus_hint}
Strategy bias: {style}

Problem:
{problem_text}

Deliverable format:
- Keep reasoning concise and correct.
- Include Python only when it materially improves reliability.
- Final line must be exactly: FINAL_ANSWER: <integer>
"""

    return PromptBundle(system=SYSTEM_PROMPT, user=user)
