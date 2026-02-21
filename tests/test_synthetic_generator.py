import importlib.util
from pathlib import Path
import unittest


def _load_generator_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_hard_synthetic_problems.py"
    spec = importlib.util.spec_from_file_location("synthetic_generator", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load synthetic generator script.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticGeneratorTests(unittest.TestCase):
    def test_extreme_rows_include_new_templates(self) -> None:
        module = _load_generator_module()
        rows = module.generate_rows(count=12, seed=7, difficulty="extreme")
        self.assertEqual(len(rows), 12)
        categories = {row["category"] for row in rows}
        self.assertIn("number_theory_floor_sum", categories)
        self.assertIn("number_theory_lucas_binomial", categories)
        self.assertTrue(all(str(row["id"]).startswith("xhard_") for row in rows))

    def test_hard_rows_use_hard_prefix(self) -> None:
        module = _load_generator_module()
        rows = module.generate_rows(count=5, seed=9, difficulty="hard")
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(str(row["id"]).startswith("hard_") for row in rows))


if __name__ == "__main__":
    unittest.main()
