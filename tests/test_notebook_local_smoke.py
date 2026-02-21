import json
import unittest
from pathlib import Path

import pandas as pd


class NotebookLocalSmokeTests(unittest.TestCase):
    def test_notebook_predict_handles_synthetic_patterns(self) -> None:
        notebook_path = Path("kaggle_kernel_submission/aimo3_submission.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        # Cell 0 is markdown, cells 1-3 are code in the current notebook layout.
        cell1 = "".join(notebook["cells"][1]["source"])
        cell2 = "".join(notebook["cells"][2]["source"])
        cell3 = "".join(notebook["cells"][3]["source"])

        # Keep only local predict logic, not the server startup section.
        cell3 = cell3.split("AIMO3_SERVER = _load_inference_server_module()", 1)[0]

        # Local fallback when polars is unavailable in dev env.
        # Newer notebook versions already contain an internal polars fallback.
        if "try:\n    import polars as pl" not in cell3:
            polars_shim = """
try:
    import polars as pl
except Exception:
    class _DummyPLDataFrame(pd.DataFrame):
        pass
    class _DummyPLSeries(pd.Series):
        pass
    class _DummyPL:
        DataFrame = _DummyPLDataFrame
        Series = _DummyPLSeries
    pl = _DummyPL()
"""
            cell3 = cell3.replace("import polars as pl", polars_shim)

        globals_dict: dict[str, object] = {"pd": pd}
        exec(cell1, globals_dict)
        exec(cell2, globals_dict)
        exec(cell3, globals_dict)

        # Force offline deterministic behavior for local smoke tests.
        globals_dict["USE_MODEL_API"] = False
        globals_dict["REFERENCE_ROWS"] = []
        globals_dict["REFERENCE_ANSWER_MAP"] = {}

        predict = globals_dict["predict"]
        debug_rows = globals_dict["DEBUG_ROWS"]

        ids = pd.Series(
            ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
            name="id",
        )
        problems = pd.Series(
            [
                "What is 123 + 456?",
                "Find the remainder when 12345 is divided by 97.",
                "Solve 17 + x = 58 for x.",
                "Solve x + 41 = 90 for x.",
                "Solve 12x + 8 = 116 for x.",
                "Find the remainder when 100006 is divided by 1000.",
                "What is 2^20?",
                "Let a_n be a sequence with complicated recurrence; compute a_2025 modulo 99991.",
            ],
            name="problem",
        )

        prediction = predict(ids, problems)
        if hasattr(prediction, "to_pandas"):
            prediction = prediction.to_pandas()

        self.assertIsInstance(prediction, pd.DataFrame)
        self.assertEqual(list(prediction.columns), ["id", "answer"])
        self.assertEqual(len(prediction), len(ids))

        expected_known = {
            "s1": 579,
            "s2": 26,
            "s3": 41,
            "s4": 49,
            "s5": 9,
            "s6": 6,
            "s7": 48576,
        }
        answers = {
            str(row["id"]): int(row["answer"])
            for _, row in prediction.iterrows()
        }
        for pid, expected in expected_known.items():
            self.assertEqual(answers[pid], expected, f"Unexpected answer for {pid}")

        # Unknown hard case should still be valid-range integer.
        self.assertTrue(0 <= answers["s8"] <= 99_999)

        self.assertEqual(len(debug_rows), len(ids))
        for row in debug_rows:
            self.assertIn("id", row)
            self.assertIn("answer", row)
            self.assertIn("source", row)
            self.assertTrue(0 <= int(row["answer"]) <= 99_999)


if __name__ == "__main__":
    unittest.main()
