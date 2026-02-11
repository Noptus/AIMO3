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
    archetype: str


SYSTEM_PROMPT = """You are an olympiad-level mathematical reasoning assistant.
Solve integer-answer competition problems with rigorous steps and strong self-checks.

Execution rules:
1) Prefer exact integer/symbolic reasoning over approximations.
2) Use Python for heavy arithmetic / combinatorics and to verify candidate values.
3) Respect any modulus requested in the problem statement.
4) End with exactly one line: FINAL_ANSWER: <integer>
"""

REPAIR_SYSTEM_PROMPT = """You are correcting a previous olympiad solution.
Focus only on finding and fixing the decisive error quickly.
Output must end with: FINAL_ANSWER: <integer>
"""

VERIFIER_SYSTEM_PROMPT = """You are a strict mathematical verifier.
Given candidate integer answers, pick the most defensible one from the candidates only.
Output must end with: FINAL_ANSWER: <integer>
"""

CONSISTENCY_AUDIT_SYSTEM_PROMPT = """You are a strict consistency auditor for olympiad solutions.
Given candidate answers and evidence, select the answer that survives the strongest contradiction checks.
Choose from provided candidates only.
Output must end with: FINAL_ANSWER: <integer>
"""

ADVERSARIAL_PROBE_SYSTEM_PROMPT = """You are an adversarial olympiad checker.
Try to falsify weak consensus answers using an independent derivation.
If the leading candidate fails, output the corrected integer.
Output must end with: FINAL_ANSWER: <integer>
"""

SMALL_ANSWER_GUARD_SYSTEM_PROMPT = """You are an olympiad correctness guard.
Current runs overproduce trivial answers (0 or 1). Re-check carefully.
Return FINAL_ANSWER: <integer> and use 0/1 only if rigorously unavoidable.
"""

GEOMETRY_RECHECK_SYSTEM_PROMPT = """You are a geometry specialist verifier for olympiad problems.
Re-check candidate answers using geometric invariants and one independent analytic check.
Choose from the provided candidates only.
Output must end with: FINAL_ANSWER: <integer>
"""

SELECTOR_SYSTEM_PROMPT = """You are a strict solution selector for olympiad math.
You are given multiple candidate traces and must pick the single most defensible answer from the provided candidates only.
Output must end with: FINAL_ANSWER: <integer>
"""

FALLBACK_GUESS_SYSTEM_PROMPT = """You are a final fallback estimator for an olympiad integer-answer problem.
Return the best-supported integer estimate.
Avoid 0 or 1 unless strongly justified.
Output must end with: FINAL_ANSWER: <integer>
"""

EXTRACTOR_SYSTEM_PROMPT = """You are a final-answer extractor.
Return only one line in this format: FINAL_ANSWER: <integer>
No extra text.
"""

AGENT_FOLLOWUP_SYSTEM_PROMPT = """You are an olympiad math agent using a Python sandbox.
You are given tool observations from executed Python code.
Continue from those observations and end with: FINAL_ANSWER: <integer>
"""

STYLE_GUIDE = {
    "algebra": "Prioritize symbolic simplification, invariants, and exact substitutions.",
    "number_theory": "Prioritize modular arithmetic, valuations, divisor structure, and parity.",
    "combinatorics": "Prioritize counting arguments, bijections, recurrences, and generating-pattern checks.",
    "geometry": "Translate geometry constraints into exact relations and verify candidate values analytically.",
    "general": "Try two independent approaches before finalizing when feasible.",
}

ARCHETYPE_GUIDE = {
    "valuation_mod": (
        "Use valuation/divisibility identities first (Legendre/LTE-style reasoning), "
        "then verify final exponent arithmetic in Python."
    ),
    "combinatorial_count": (
        "Identify canonical counting structure (Catalan/binomial/DP/inclusion-exclusion), "
        "then verify formula numerically for consistency."
    ),
    "functional_equation": (
        "Transform function equation into multiplicative/additive structure and parameterize all solutions."
    ),
    "geometry_configuration": (
        "Extract invariant equalities first (power/radical-axis/bisector/similarity) before numeric substitution."
    ),
    "divisor_arithmetic": (
        "Reduce to divisor/prime-factor structure and avoid brute-force unless bounded and verified."
    ),
    "general_olympiad": "Proceed with exact reasoning and a strict end-of-solution sanity check.",
}


def _attempt_strategy(profile: ProblemProfile, attempt_index: int) -> str:
    """Diversify reasoning paths across attempts for self-consistency."""

    if profile.archetype == "valuation_mod":
        options = [
            "valuation-first: derive v_p identities before any expansion.",
            "constructive-check: derive formula then verify with compact Python checks.",
            "parity-and-mod: establish residue constraints first, then compute final valuation.",
        ]
        return options[attempt_index % len(options)]

    if profile.archetype == "combinatorial_count":
        options = [
            "combinatorial-identity-first: find counting formula before numeric evaluation.",
            "recursive/decomposition-first: split structure into independent pieces.",
            "code-aided-count-check: derive symbolic expression then verify with quick script.",
        ]
        return options[attempt_index % len(options)]

    if profile.archetype == "functional_equation":
        options = [
            "structure-first: transform into additive/multiplicative form.",
            "parameterization-first: characterize all admissible function families.",
            "constraint-tightening: use bounds and sample points to prune cases.",
        ]
        return options[attempt_index % len(options)]

    if profile.archetype == "geometry_configuration":
        options = [
            "invariant-first: radical-axis/power/similarity before coordinate setup.",
            "analytic-fallback: switch to coordinates only if synthetic route stalls.",
            "double-check route: independent verification of final numeric relation.",
        ]
        return options[attempt_index % len(options)]

    options = [
        "proof-first then compute.",
        "compute-first then prove consistency.",
        "independent second route for final validation.",
    ]
    return options[attempt_index % len(options)]


def _attempt_execution_mode(attempt_index: int) -> str:
    """Approximate a 3:4 proof-first/code-first mix across attempts."""

    cycle = [
        "proof-first",
        "code-first",
        "code-first",
        "proof-first",
        "code-first",
        "proof-first",
        "code-first",
    ]
    return cycle[attempt_index % len(cycle)]


def _detect_archetype(text: str) -> str:
    if re.search(r"nu[_\s]?[25]|valuation|legendre|lifting the exponent|lte", text):
        return "valuation_mod"
    if re.search(r"catalan|tournament|arrange|number of possible|ordering", text):
        return "combinatorial_count"
    if re.search(r"f\s*\(|for all positive integers|function\s*[:\u2236]", text):
        return "functional_equation"
    if re.search(r"triangle|circle|incircle|circumcircle|angle bisector|radical axis", text):
        return "geometry_configuration"
    if re.search(r"divisor|divides|coprime|norwegian|remainder", text):
        return "divisor_arithmetic"
    return "general_olympiad"


def classify_problem(problem_text: str) -> str:
    """Route problem into high-level category."""

    text = problem_text.lower()
    archetype = _detect_archetype(text)

    if archetype == "geometry_configuration":
        return "geometry"
    if archetype in {"valuation_mod", "divisor_arithmetic"}:
        return "number_theory"
    if archetype == "combinatorial_count":
        return "combinatorics"
    if archetype == "functional_equation":
        return "algebra"

    # Fallback keyword routing.
    if re.search(r"triangle|circle|angle|perpendicular|parallel|midpoint", text):
        return "geometry"
    if re.search(r"prime|divisible|mod|remainder|gcd|lcm", text):
        return "number_theory"
    if re.search(r"ways|arrange|permutation|combination|subset|graph|color", text):
        return "combinatorics"
    if re.search(r"polynomial|equation|root|coefficient|function|recurrence", text):
        return "algebra"

    return "general"


def estimate_problem_profile(problem_text: str) -> ProblemProfile:
    """Estimate category + complexity to adapt solve budget."""

    text = problem_text.lower()
    archetype = _detect_archetype(text)
    category = classify_problem(problem_text)

    score = 0

    # Length and symbolic density.
    if len(problem_text) > 900:
        score += 2
    elif len(problem_text) > 450:
        score += 1

    hard_patterns = [
        r"\b\d+!\b",
        r"\b2\^\d{2,}\b",
        r"\b10\^\d{3,}\b",
        r"\bfor all\b",
        r"largest non-negative integer",
        r"sufficiently large",
        r"unique such triangle",
        r"across all",
    ]
    for pattern in hard_patterns:
        if re.search(pattern, text):
            score += 1

    if archetype in {"valuation_mod", "geometry_configuration", "combinatorial_count"}:
        score += 1

    if score >= 5:
        complexity = "hard"
    elif score >= 2:
        complexity = "medium"
    else:
        complexity = "easy"

    return ProblemProfile(category=category, complexity=complexity, score=score, archetype=archetype)


def build_prompt(
    problem_text: str,
    *,
    attempt_index: int,
    modulus: int | None,
    profile: ProblemProfile,
    hard_mode: bool,
) -> PromptBundle:
    """Compose a solve prompt with adaptive AIMO guidance."""

    style = STYLE_GUIDE.get(profile.category, STYLE_GUIDE["general"])
    archetype_hint = ARCHETYPE_GUIDE.get(profile.archetype, ARCHETYPE_GUIDE["general_olympiad"])
    attempt_strategy = _attempt_strategy(profile, attempt_index)
    execution_mode = _attempt_execution_mode(attempt_index)
    modulus_hint = (
        f"Known modulus for final normalization: {modulus}."
        if modulus is not None
        else "No trusted modulus extracted yet. Infer modulus carefully from the statement."
    )

    hard_guidance = ""
    if hard_mode or profile.complexity == "hard":
        hard_guidance = (
            "Hard-mode checklist:\n"
            "- Draft a short plan before derivation.\n"
            "- Use at least one Python verification on decisive arithmetic.\n"
            "- Validate final value against bounds/parity/modulus before the last line."
        )

    geometry_guidance = ""
    if profile.category == "geometry":
        geometry_guidance = (
            "Geometry checklist:\n"
            "- Start with explicit named invariants (similarity, power of a point, angle/chord, ratio lemmas).\n"
            "- If synthetic derivation stalls, switch to analytic coordinates or vectors with clean variable definitions.\n"
            "- Run one independent cross-check (synthetic vs analytic or two analytic parameterizations).\n"
            "- Avoid decimal approximations unless they are proven exact afterward."
        )

    reliability_protocol = (
        "Reliability protocol:\n"
        "- Derive a primary solution route.\n"
        "- Run at least one contradiction check (bounds/parity/residue/monotonicity).\n"
        "- Verify final candidate against the required modulus and statement constraints.\n"
        "- If two routes disagree, resolve disagreement before final line."
    )

    user = f"""AIMO profile:
- Category: {profile.category}
- Archetype: {profile.archetype}
- Complexity: {profile.complexity} (score={profile.score})
- Attempt index: {attempt_index}
- Strategy bias: {style}
- Archetype tactic: {archetype_hint}
- Attempt strategy: {attempt_strategy}
- Execution mode: {execution_mode}
- {modulus_hint}
{hard_guidance}
{geometry_guidance}
{reliability_protocol}

Problem:
{problem_text}

Deliverable format:
- Keep reasoning concise but rigorous.
- Use Python when it materially improves reliability.
- If mode is code-first, write compact executable checks early and use them to validate the final value.
- If mode is proof-first, derive the key identity before code checks.
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
Archetype: {profile.archetype}
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


def build_agent_followup_prompt(
    problem_text: str,
    *,
    previous_response: str,
    tool_observation: str,
    modulus: int | None,
    profile: ProblemProfile,
    round_index: int,
) -> PromptBundle:
    """Prompt for iterative agentic follow-up after sandbox execution."""

    modulus_hint = (
        f"Normalize the final answer modulo {modulus}."
        if modulus is not None
        else "Infer and apply the correct modulus from the statement."
    )

    user = f"""Agent follow-up round {round_index + 1}
Category: {profile.category}
Archetype: {profile.archetype}
Complexity: {profile.complexity}
{modulus_hint}

Problem:
{problem_text}

Previous model output:
{previous_response}

Python sandbox observations:
{tool_observation}

Rules:
- Use the observations to refine or correct the solution.
- If evidence is insufficient, emit one compact python block for the next check.
- Prefer exact arithmetic and explicit invariant checks over guessing.
- End with exactly one line: FINAL_ANSWER: <integer>
"""

    return PromptBundle(system=AGENT_FOLLOWUP_SYSTEM_PROMPT, user=user)


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
Archetype: {profile.archetype}
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


def build_small_answer_guard_prompt(
    problem_text: str,
    *,
    candidate_answers: list[int],
    candidate_summaries: list[str],
    modulus: int | None,
    profile: ProblemProfile,
) -> PromptBundle:
    """Prompt to challenge weakly supported trivial answers (0/1)."""

    modulus_hint = (
        f"Any final value must be normalized modulo {modulus}."
        if modulus is not None
        else "Use the modulus specified in the problem statement."
    )
    joined = "\n".join(f"- Candidate {i + 1}: {ans}" for i, ans in enumerate(candidate_answers))
    if not joined:
        joined = "- none extracted reliably yet"
    summaries = "\n".join(f"- {line}" for line in candidate_summaries)
    if not summaries:
        summaries = "- no reliable prior evidence"

    user = f"""Re-solve the decisive part and avoid trivial-answer collapse.
Category: {profile.category}
Archetype: {profile.archetype}
Complexity: {profile.complexity}
{modulus_hint}

Problem:
{problem_text}

Current candidate answers:
{joined}

Candidate evidence:
{summaries}

Rules:
- Re-check with at least one independent consistency check.
- Use 0 or 1 only if your derivation forces it.
- Final line must be exactly: FINAL_ANSWER: <integer>
"""

    return PromptBundle(system=SMALL_ANSWER_GUARD_SYSTEM_PROMPT, user=user)


def build_consistency_audit_prompt(
    problem_text: str,
    *,
    candidate_answers: list[int],
    candidate_summaries: list[str],
    modulus: int | None,
    profile: ProblemProfile,
) -> PromptBundle:
    """Prompt for contradiction-focused arbitration over top candidates."""

    modulus_hint = (
        f"Any final value must be normalized modulo {modulus}."
        if modulus is not None
        else "Use the modulus specified in the problem statement."
    )
    joined = "\n".join(f"- Candidate {i + 1}: {ans}" for i, ans in enumerate(candidate_answers))
    summaries = "\n".join(f"- {line}" for line in candidate_summaries)

    user = f"""Select the candidate that best survives consistency checks.
Category: {profile.category}
Archetype: {profile.archetype}
Complexity: {profile.complexity}
{modulus_hint}

Problem:
{problem_text}

Candidate answers:
{joined}

Candidate evidence:
{summaries}

Audit rules:
- Prefer candidates supported by multiple independent traces.
- Penalize candidates that violate bounds, parity, residue constraints, or key invariants.
- You must choose exactly one value from the candidate list.
- Do not introduce a new number.
- Final line must be exactly: FINAL_ANSWER: <integer>
"""

    return PromptBundle(system=CONSISTENCY_AUDIT_SYSTEM_PROMPT, user=user)


def build_adversarial_probe_prompt(
    problem_text: str,
    *,
    candidate_answers: list[int],
    candidate_summaries: list[str],
    modulus: int | None,
    profile: ProblemProfile,
) -> PromptBundle:
    """Prompt to challenge fragile consensus and recover from local minima."""

    modulus_hint = (
        f"Any final value must be normalized modulo {modulus}."
        if modulus is not None
        else "Use the modulus specified in the problem statement."
    )
    joined = "\n".join(f"- Candidate {i + 1}: {ans}" for i, ans in enumerate(candidate_answers))
    if not joined:
        joined = "- no stable candidate yet"
    summaries = "\n".join(f"- {line}" for line in candidate_summaries)
    if not summaries:
        summaries = "- no reliable prior evidence"

    user = f"""Challenge the current candidate set with an independent route.
Category: {profile.category}
Archetype: {profile.archetype}
Complexity: {profile.complexity}
{modulus_hint}

Problem:
{problem_text}

Current candidates:
{joined}

Candidate evidence:
{summaries}

Probe rules:
- Attempt to refute the current leading candidate using contradiction checks.
- Use an independent method (different route from the leading traces).
- If the leading candidate survives, output it; otherwise output the corrected value.
- Final line must be exactly: FINAL_ANSWER: <integer>
"""

    return PromptBundle(system=ADVERSARIAL_PROBE_SYSTEM_PROMPT, user=user)


def build_selector_prompt(
    problem_text: str,
    *,
    answer_options: list[int],
    option_evidence: list[str],
    modulus: int | None,
    profile: ProblemProfile,
) -> PromptBundle:
    """Prompt for a GenSelect-style arbitration using richer candidate evidence."""

    modulus_hint = (
        f"Any final value must be normalized modulo {modulus}."
        if modulus is not None
        else "Use the modulus specified in the problem statement."
    )

    joined_answers = "\n".join(f"- Option {i + 1}: {ans}" for i, ans in enumerate(answer_options))
    joined_evidence = "\n\n".join(
        f"Option {i + 1} evidence:\n{evidence}" for i, evidence in enumerate(option_evidence)
    )

    user = f"""Pick the single best answer option using consistency and mathematical defensibility.
Category: {profile.category}
Archetype: {profile.archetype}
Complexity: {profile.complexity}
{modulus_hint}

Problem:
{problem_text}

Answer options:
{joined_answers}

Evidence:
{joined_evidence}

Rules:
- You must select exactly one value from the answer options.
- Do not introduce a new number.
- Favor options supported by consistent independent traces and verified computations.
- Final line must be exactly: FINAL_ANSWER: <integer>
"""

    return PromptBundle(system=SELECTOR_SYSTEM_PROMPT, user=user)


def build_fallback_guess_prompt(
    problem_text: str,
    *,
    modulus: int | None,
    profile: ProblemProfile,
) -> PromptBundle:
    """Prompt used when all normal extraction paths failed."""

    modulus_hint = (
        f"Normalize final answer modulo {modulus}."
        if modulus is not None
        else "Use the modulus from the problem statement if present."
    )

    user = f"""All prior attempts failed to yield a stable extracted answer.
Category: {profile.category}
Archetype: {profile.archetype}
Complexity: {profile.complexity}
{modulus_hint}

Problem:
{problem_text}

Rules:
- Produce your best integer estimate from mathematical reasoning.
- Avoid 0 or 1 unless forced.
- Final line must be exactly: FINAL_ANSWER: <integer>
"""

    return PromptBundle(system=FALLBACK_GUESS_SYSTEM_PROMPT, user=user)


def build_geometry_recheck_prompt(
    problem_text: str,
    *,
    candidate_answers: list[int],
    candidate_summaries: list[str],
    modulus: int | None,
    profile: ProblemProfile,
) -> PromptBundle:
    """Prompt for a geometry-focused candidate recheck."""

    modulus_hint = (
        f"Any final value must be normalized modulo {modulus}."
        if modulus is not None
        else "Use the modulus specified in the problem statement."
    )

    joined = "\n".join(f"- Candidate {i + 1}: {ans}" for i, ans in enumerate(candidate_answers))
    summaries = "\n".join(f"- {line}" for line in candidate_summaries)

    user = f"""Re-evaluate candidate answers with geometry-specific rigor.
Category: {profile.category}
Archetype: {profile.archetype}
Complexity: {profile.complexity}
{modulus_hint}

Problem:
{problem_text}

Candidate answers:
{joined}

Candidate evidence:
{summaries}

Geometry verification rules:
- Check consistency with at least one invariant relation and one independent analytic check.
- Reject candidates that rely on unverified diagram assumptions.
- You must choose exactly one value from the candidate list.
- Do not introduce a new number.
- Final line must be exactly: FINAL_ANSWER: <integer>
"""

    return PromptBundle(system=GEOMETRY_RECHECK_SYSTEM_PROMPT, user=user)


def build_final_extractor_prompt(
    problem_text: str,
    *,
    previous_response: str,
    modulus: int | None,
    profile: ProblemProfile,
) -> PromptBundle:
    """Prompt to force a strict final-answer line when prior output was incomplete."""

    modulus_hint = (
        f"Normalize final answer modulo {modulus}." if modulus is not None else "Infer the modulus from the statement."
    )

    user = f"""Extract one final integer from the prior trace.
Category: {profile.category}
Archetype: {profile.archetype}
{modulus_hint}

Previous partial solution trace:
{previous_response}

Rules:
- Do not solve the problem again.
- Infer the best supported final integer from the trace.
- Return exactly one line and nothing else:
FINAL_ANSWER: <integer>
"""

    return PromptBundle(system=EXTRACTOR_SYSTEM_PROMPT, user=user)
