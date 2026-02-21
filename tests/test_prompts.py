import unittest

from aimo3.prompts import build_prompt, classify_problem, estimate_problem_profile


class PromptProfileTests(unittest.TestCase):
    def test_divisor_heavy_statement_not_misclassified_as_functional_equation(self) -> None:
        problem = (
            "Let n >= 6 be a positive integer. We call an integer n-Norwegian if it has three "
            "distinct positive divisors whose sum is n. Let f(n) denote the smallest n-Norwegian "
            "positive integer. Let M = 3^{2025!} and define g(c) = floor(2025! f(M+c)/M) / 2025!. "
            "Find the remainder when p+q is divided by 99991."
        )
        profile = estimate_problem_profile(problem)
        self.assertEqual(profile.category, "number_theory")
        self.assertEqual(profile.archetype, "divisor_arithmetic")
        self.assertIn(profile.complexity, {"medium", "hard"})

    def test_rectangle_partition_problem_routes_to_combinatorics(self) -> None:
        problem = (
            "A 500 x 500 square is divided into k rectangles with integer side lengths. "
            "No two rectangles have the same perimeter. The largest possible value of k is K. "
            "Find K modulo 10^5."
        )
        profile = estimate_problem_profile(problem)
        self.assertEqual(classify_problem(problem), "combinatorics")
        self.assertIn(profile.archetype, {"combinatorial_count", "general_olympiad"})
        self.assertIn(profile.complexity, {"medium", "hard"})

    def test_build_prompt_includes_aimo_guardrails(self) -> None:
        problem = "Find the remainder when 2^100000 is divided by 99991."
        profile = estimate_problem_profile(problem)
        prompt = build_prompt(
            problem_text=problem,
            attempt_index=0,
            modulus=99991,
            profile=profile,
            hard_mode=False,
        )
        self.assertIn("AIMO anti-shortcut protocol", prompt.user)
        self.assertIn("Category-specific checklist", prompt.user)
        self.assertIn("RESULT_JSON", prompt.user)
        self.assertIn("FINAL_ANSWER: <integer>", prompt.user)


if __name__ == "__main__":
    unittest.main()
