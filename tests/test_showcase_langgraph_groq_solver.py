import sys
import unittest
from importlib import util
from pathlib import Path


def _load_showcase_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "showcase"
        / "langgraph_groq_mastery"
        / "langgraph_groq_solver.py"
    )
    spec = util.spec_from_file_location("showcase_langgraph_groq_solver", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load showcase solver module.")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ShowcaseLangGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_showcase_module()

    def test_categorize_geometry(self) -> None:
        category, complexity = self.module._categorize(
            "In triangle ABC with circumcircle and angle bisector, find the remainder."
        )
        self.assertEqual(category, "geometry")
        self.assertIn(complexity, {"easy", "medium", "hard"})

    def test_safe_json_loads(self) -> None:
        payload = self.module._safe_json_loads('{"answer": 42, "independent_check_passed": true}')
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["answer"], 42)

    def test_safe_json_loads_invalid(self) -> None:
        payload = self.module._safe_json_loads("not-json")
        self.assertIsNone(payload)

    def test_pattern_solver_power_mix(self) -> None:
        problem = (
            "Let S = 82^43486 + 96^148182 - 33^29248 + 34^66931. "
            "Compute S modulo 100000."
        )
        answer, method = self.module.solve_deterministic_pattern(problem, modulus=100000)
        self.assertEqual(answer, 76383)
        self.assertEqual(method, "power_mix_closed_form")

    def test_pattern_solver_crt_quadratic(self) -> None:
        problem = (
            "Let x be the smallest nonnegative integer such that "
            "x ≡ 18 (mod 107), x ≡ 65 (mod 103), and x ≡ 31 (mod 101). "
            "Compute x^2 + 23x + 55 modulo 50021."
        )
        answer, method = self.module.solve_deterministic_pattern(problem, modulus=50021)
        self.assertEqual(answer, 5558)
        self.assertEqual(method, "crt_quadratic")


if __name__ == "__main__":
    unittest.main()
