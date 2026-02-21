import datetime as dt
import os
import unittest
from unittest.mock import patch

from aimo3.kaggle_api import (
    _load_kaggle_from_env,
    _normalize_kernel_ref,
    _normalize_status,
    _parse_submission_datetime,
)


class KaggleApiHelperTests(unittest.TestCase):
    def test_normalize_status_handles_enum_prefix(self) -> None:
        self.assertEqual(_normalize_status("SubmissionStatus.PENDING"), "pending")
        self.assertEqual(_normalize_status("KernelWorkerStatus.COMPLETE"), "complete")
        self.assertEqual(_normalize_status("running"), "running")

    def test_parse_submission_datetime_common_formats(self) -> None:
        parsed = _parse_submission_datetime("2026-02-12 07:42:08.183000")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.tzinfo, dt.timezone.utc)
        self.assertEqual(parsed.year, 2026)
        self.assertEqual(parsed.month, 2)
        self.assertEqual(parsed.day, 12)

    def test_normalize_kernel_ref_handles_code_urls(self) -> None:
        self.assertEqual(
            _normalize_kernel_ref("/code/raphaelcaillon/aimo3-progress-prize-3-working"),
            "raphaelcaillon/aimo3-progress-prize-3-working",
        )
        self.assertEqual(
            _normalize_kernel_ref("https://www.kaggle.com/code/raphaelcaillon/aimo3-progress-prize-3-working"),
            "raphaelcaillon/aimo3-progress-prize-3-working",
        )

    def test_load_kaggle_from_env_supports_key_only_token(self) -> None:
        with patch.dict("os.environ", {"KAGGLE_API_TOKEN": "key_only_value"}, clear=True):
            _load_kaggle_from_env(default_username="owner_name")
            self.assertEqual("owner_name", os.environ.get("KAGGLE_USERNAME"))
            self.assertEqual("key_only_value", os.environ.get("KAGGLE_KEY"))


if __name__ == "__main__":
    unittest.main()
