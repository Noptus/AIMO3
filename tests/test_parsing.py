import unittest

from aimo3.parsing import (
    parse_answer_from_text,
    parse_modulus,
    parse_structured_result,
    select_weighted_mode,
)


class ParsingTests(unittest.TestCase):
    def test_parse_modulus_power_expression(self) -> None:
        text = "Return the answer modulo 10^5."
        self.assertEqual(parse_modulus(text), 100000)

    def test_parse_modulus_remainder_phrase(self) -> None:
        text = "Find the remainder when divided by 57."
        self.assertEqual(parse_modulus(text), 57)

    def test_parse_modulus_non_decimal_power(self) -> None:
        text = "What is the remainder when x is divided by 5^7?"
        self.assertEqual(parse_modulus(text), 78125)

    def test_parse_modulus_latex_power_with_dollars(self) -> None:
        text = "Find the remainder when abc is divided by $10^{5}$."
        self.assertEqual(parse_modulus(text), 100000)

    def test_parse_modulus_latex_integer_with_punctuation(self) -> None:
        text = "What is the remainder when n is divided by $99991$?"
        self.assertEqual(parse_modulus(text), 99991)

    def test_parse_modulus_prefers_final_modulus_when_multiple_present(self) -> None:
        text = (
            "Let x satisfy x ≡ 18 (mod 107), x ≡ 65 (mod 103), and x ≡ 31 (mod 101). "
            "Compute x^2 + 23x + 55 modulo 50021."
        )
        self.assertEqual(parse_modulus(text), 50021)

    def test_parse_answer_prefers_final_answer_tag(self) -> None:
        text = "Some reasoning... FINAL_ANSWER: 12345"
        parsed = parse_answer_from_text(text)
        self.assertEqual(parsed.answer, 12345)
        self.assertEqual(parsed.source, "final_answer_tag")

    def test_parse_answer_boxed_fallback(self) -> None:
        text = "Hence the value is \\boxed{998}"
        parsed = parse_answer_from_text(text)
        self.assertEqual(parsed.answer, 998)
        self.assertEqual(parsed.source, "boxed")

    def test_parse_answer_ignores_non_final_answer_mentions(self) -> None:
        text = "We start answer exploration from 0 and continue searching."
        parsed = parse_answer_from_text(text)
        self.assertIsNone(parsed.answer)
        self.assertEqual(parsed.source, "none")

    def test_parse_answer_from_result_json(self) -> None:
        text = (
            'RESULT_JSON: {"answer": 424242, "method": "modular-check", '
            '"independent_check_passed": true}\n'
            "FINAL_ANSWER: 424242"
        )
        parsed = parse_answer_from_text(text)
        self.assertEqual(parsed.answer, 24242)
        self.assertEqual(parsed.source, "structured_json")

        structured = parse_structured_result(text)
        self.assertEqual(structured.answer, 24242)
        self.assertEqual(structured.method, "modular-check")
        self.assertTrue(structured.independent_check_passed)

    def test_weighted_mode(self) -> None:
        values = [1, 2, 2, 3]
        weights = [0.1, 1.0, 1.5, 10.0]
        self.assertEqual(select_weighted_mode(values, weights), 3)


if __name__ == "__main__":
    unittest.main()
