"""Prompt templates and routing for AIMO3 solving."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str


@dataclass(frozen=True)
class ProblemProfile:
    category: str
    complexity: str
    score: int


SYSTEM_PROMPT = """You are an olympiad-level mathematical reasoning assistant.
Solve integer-answer competition problems with rigorous steps and strong self-checks.

Execution rules:
1) Prefer exact algebraic / integer reasoning over approximations.
2) Use compact Python when calculations are heavy or error-prone.
3) Respect any modulus requested in the problem statement.
4) End with exactly one line: FINAL_ANSWER: <integer>
"""

REPAIR_SYSTEM_PROMPT = """You are correcting a previous olympiad solution.
Focus only on finding and fixing the error quickly and return one final integer.
Output must end with: FINAL_ANSWER: <integer>
"""

VERIFIER_SYSTEM_PROMPT = """You are a strict mathematical verifier.
Given candidate integer answers, pick the most defensible one from the candidates only.
Output must end with: FINAL_ANSWER: <integer>
"""

STYLE_GUIDE = {
    "algebra": "Prioritize symbolic simplification, invariants, and exact substitutions.",
    "number_theory": "Prioritize modular arithmetic, valuations, and divisibility structure.",
    "combinatorics": "Prioritize counting arguments, bijections, and combinatorial identities.",
    "geometry": "Translate to analytic/geometric invariants carefully and verify equalities exactly.",
    "general": "Explore two independent paths before concluding when feasible.",
}


def classify_problem(problem_text: str) -> str:
    """Lightweight router for problem-specific prompting."""

    text = problem_text.lower()

    if re.search(r"triangle|circle|angle|perpendicular|parallel|midpoint|incircle|circumcircle", text):
        return "geometry"
    if re.search(r"prime|divisible|mod|remainder|gcd|lcm|valuation|legendre", text):
        return "number_theory"
    if re.search(r"ways|arrange|permutation|combination|subset|graph|color|tournament|catalan", text):
        return "combinatorics"
    if re.search(r"polynomial|equation|root|coefficient|function|recurrence", text):
        return "algebra"

    return "general"


def estimate_problem_profile(problem_text: str) -> ProblemProfile:
    """Estimate category + complexity to adapt solve budget."""

    text = problem_text.lower()
    category = classify_problem(problem_text)

    score = 0

    # Length and structural complexity.
    if len(problem_text) > 900:
        score += 2
    elif len(problem_text) > 450:
        score += 1

    # Markers typical of hard olympiad-number-theory/combinatorics tasks.
    hard_patterns = [
        r"\b\d+!\b",
        r"\b2\^\d{2,}\b",
        r"\b10\^\d{3,}\b",
        r"\bnu[_\s]?[25]\b",
        r"legendre|lifting the exponent|lte|valuation",
        r"catalan|stewart|radical axis|angle bisector",
        r"for all positive integers",
        r"largest non-negative integer",
    ]
    for pattern in hard_patterns:
        if re.search(pattern, text):
            score += 1

    # Extra weight when multiple high-complexity hints appear.
    symbol_hits = sum(1 for token in ["mod", "remainder", "divides", "unique", "minimum perimeter"] if token in text)
    if symbol_hits >= 3:
        score += 1

    if score >= 5:
        complexity = "hard"
    elif score >= 2:
        complexity = "medium"
    else:
        complexity = "easy"

    return ProblemProfile(category=category, complexity=complexity, score=score)


def build_prompt(
    problem_text: str,
    *,
    attempt_index: int,
    modulus: int | None,
    profile: ProblemProfile,
    hard_mode: bool,
) -> PromptBundle:
    """Compose a solve prompt with adaptive strategy guidance."""

    style = STYLE_GUIDE.get(profile.category, STYLE_GUIDE["general"])
    modulus_hint = (
        f"Known modulus for final normalization: {modulus}."
        if modulus is not None
        else "No trusted modulus extracted yet. Infer modulus carefully from the statement."
    )

    hard_guidance = ""
    if hard_mode or profile.complexity == "hard":
        hard_guidance = (
            "Hard-mode guidance:\n"
            "- Build a short plan before detailed derivation.\n"
            "- Use at least one python verification for non-trivial arithmetic/combinatorics.\n"
            "- Before final line, run a final consistency check (bounds, parity, modulus)."
        )

    user = f"""Problem category hint: {profile.category}
Complexity estimate: {profile.complexity} (score={profile.score})
Attempt index: {attempt_index}
{modulus_hint}
Strategy bias: {style}
{hard_guidance}

Problem:
{problem_text}

Deliverable format:
- Keep reasoning concise but rigorous.
- Use Python only when it materially improves reliability.
- Final line must be exactly: FINAL_ANSWER: <integer>
"""

    return PromptBundle(system=SYSTEM_PROMPT, user=user)


def build_repair_prompt(
    problem_text: str,
    *,
    previous_response: str,
    tool_feedback: str,
    modulus: int | None,
    profile: ProblemProfile,
) -> PromptBundle:
    """Prompt for a focused second-pass correction."""

    modulus_hint = (
        f"Normalize final answer modulo {modulus}." if modulus is not None else "Infer and apply the correct modulus."
    )

    user = f"""The prior attempt likely has an extraction/consistency issue.
Category: {profile.category}
Complexity: {profile.complexity}
{modulus_hint}

Problem:
{problem_text}

Previous attempt:
{previous_response}

Tool feedback:
{tool_feedback}

Fix only the decisive error and return one final integer.
Final line must be exactly: FINAL_ANSWER: <integer>
"""

    return PromptBundle(system=REPAIR_SYSTEM_PROMPT, user=user)


def build_verifier_prompt(
    problem_text: str,
    *,
    candidate_answers: list[int],
    candidate_summaries: list[str],
    modulus: int | None,
    profile: ProblemProfile,
) -> PromptBundle:
    """Prompt used to arbitrate between top candidate answers."""

    modulus_hint = (
        f"Any final value must be normalized modulo {modulus}."
        if modulus is not None
        else "Use the modulus specified in the problem statement."
    )

    joined = "\n".join(f"- Candidate {i + 1}: {ans}" for i, ans in enumerate(candidate_answers))
    summaries = "\n".join(f"- {line}" for line in candidate_summaries)

    user = f"""Choose the most defensible answer from the candidate list only.
Category: {profile.category}
Complexity: {profile.complexity}
{modulus_hint}

Problem:
{problem_text}

Candidate answers:
{joined}

Candidate evidence (high-level):
{summaries}

Rules:
- You must select exactly one value from the candidate list.
- Do not introduce a new number.
- Final line must be exactly: FINAL_ANSWER: <integer>
"""

    return PromptBundle(system=VERIFIER_SYSTEM_PROMPT, user=user)
