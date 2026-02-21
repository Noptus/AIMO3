import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from aimo3.pipeline import load_submission_file, sanitize_submission, validate_submission_file


class PipelineSubmissionTests(unittest.TestCase):
    def test_sanitize_submission_repairs_invalid_and_duplicates(self) -> None:
        raw = pd.DataFrame(
            [
                {"id": "a", "answer": "10"},
                {"id": "a", "answer": "not_int"},
                {"id": "", "answer": -7},
                {"id": "c", "answer": 100_123},
            ]
        )
        fixed, summary = sanitize_submission(raw, default_answer=42)

        self.assertEqual(list(fixed.columns), ["id", "answer"])
        self.assertEqual(len(fixed), 3)
        self.assertTrue(summary.changed)
        self.assertEqual(summary.duplicate_ids, 1)
        self.assertEqual(summary.invalid_answers_fixed, 1)
        self.assertEqual(summary.out_of_range_normalized, 2)
        self.assertEqual(summary.empty_ids_fixed, 1)
        self.assertTrue((fixed["answer"] >= 0).all())
        self.assertTrue((fixed["answer"] <= 99_999).all())

    def test_sanitize_submission_aligns_to_input_ids(self) -> None:
        raw = pd.DataFrame(
            [
                {"id": "x", "answer": 5},
                {"id": "z", "answer": 7},
            ]
        )
        fixed, summary = sanitize_submission(raw, input_ids=["x", "y", "z"], default_answer=11)

        self.assertEqual(fixed["id"].tolist(), ["x", "y", "z"])
        self.assertEqual(fixed["answer"].tolist(), [5, 11, 7])
        self.assertEqual(summary.missing_ids_filled, 1)
        self.assertTrue(summary.changed)

    def test_validate_submission_file_strict_rejects_invalid_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad_submission.csv"
            pd.DataFrame([{"id": "x", "answer": "not_int"}]).to_csv(path, index=False)

            with self.assertRaises(ValueError):
                validate_submission_file(path, strict=True, default_answer=0)

    def test_validate_submission_file_non_strict_returns_fixed_dataframe(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            pd.DataFrame([{"id": "x", "answer": "not_int"}]).to_csv(path, index=False)

            fixed, report = validate_submission_file(path, strict=False, default_answer=7)
            self.assertEqual(fixed["answer"].tolist(), [7])
            self.assertTrue(report.changed_by_sanitization)
            self.assertEqual(report.file_format, "csv")

    def test_validate_submission_file_strict_rejects_row_mismatch_against_input_ids(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            pd.DataFrame([{"id": "x", "answer": 7}]).to_csv(path, index=False)

            with self.assertRaises(ValueError):
                validate_submission_file(
                    path,
                    strict=True,
                    default_answer=0,
                    input_ids=["x", "y"],
                )

    def test_validate_submission_file_strict_rejects_out_of_range_answer(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            pd.DataFrame([{"id": "x", "answer": 100_001}]).to_csv(path, index=False)

            with self.assertRaises(ValueError):
                validate_submission_file(path, strict=True, default_answer=0)

    def test_load_submission_file_parquet_csv_fallback_when_engine_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "submission.parquet"
            csv_path = Path(tmpdir) / "submission.csv"
            parquet_path.write_bytes(b"placeholder")
            pd.DataFrame([{"id": "x", "answer": 1}]).to_csv(csv_path, index=False)

            with patch("pandas.read_parquet", side_effect=ImportError("no parquet engine")):
                frame, file_format = load_submission_file(parquet_path)

            self.assertEqual(file_format, "parquet(csv-fallback)")
            self.assertEqual(frame["id"].tolist(), ["x"])
            self.assertEqual(frame["answer"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
