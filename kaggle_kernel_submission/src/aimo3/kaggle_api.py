"""Kaggle API automation helpers."""

from __future__ import annotations

import datetime as dt
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SubmissionRecord:
    ref: str
    file_name: str
    status: str
    submitted_at: str
    public_score: str | None
    private_score: str | None


def _load_kaggle_from_env() -> None:
    """Support either standard Kaggle env vars or a combined token."""

    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return

    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        return

    if ":" in token:
        username, key = token.split(":", 1)
        os.environ.setdefault("KAGGLE_USERNAME", username.strip())
        os.environ.setdefault("KAGGLE_KEY", key.strip())


class KaggleAutomation:
    """Thin wrapper around the official Kaggle API with submission polling."""

    def __init__(self, competition: str) -> None:
        self.competition = competition

        _load_kaggle_from_env()

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except Exception as exc:
            raise RuntimeError(
                "kaggle package is required. Install with `pip install kaggle`."
            ) from exc

        self.api = KaggleApi()
        self.api.authenticate()

    def download_competition_files(self, output_dir: str | Path, *, unzip: bool = True) -> list[Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        self.api.competition_download_files(self.competition, path=str(output), quiet=False)

        zip_path = output / f"{self.competition}.zip"
        extracted: list[Path] = []

        if unzip and zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(output)
                extracted = [output / name for name in zf.namelist()]

        return extracted

    def submit(self, submission_csv: str | Path, message: str) -> str:
        submission_csv = str(Path(submission_csv).resolve())
        return self.api.competition_submit(submission_csv, message, self.competition)

    def list_submissions(self, limit: int = 20) -> list[SubmissionRecord]:
        entries = self.api.competition_submissions(self.competition)
        records: list[SubmissionRecord] = []

        for sub in entries[:limit]:
            records.append(
                SubmissionRecord(
                    ref=str(getattr(sub, "ref", "")),
                    file_name=str(getattr(sub, "fileName", "")),
                    status=str(getattr(sub, "status", "")),
                    submitted_at=str(getattr(sub, "date", "")),
                    public_score=_as_optional_str(getattr(sub, "publicScore", None)),
                    private_score=_as_optional_str(getattr(sub, "privateScore", None)),
                )
            )

        return records

    def wait_for_latest(
        self,
        *,
        poll_seconds: int = 20,
        timeout_minutes: int = 60,
    ) -> SubmissionRecord | None:
        """Poll until the latest submission is no longer pending."""

        deadline = time.time() + timeout_minutes * 60

        latest = self.list_submissions(limit=1)
        if not latest:
            return None

        current_ref = latest[0].ref

        while time.time() < deadline:
            now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            submission = self.list_submissions(limit=1)
            if not submission:
                time.sleep(poll_seconds)
                continue

            top = submission[0]
            if top.ref != current_ref:
                current_ref = top.ref

            status = top.status.lower()
            print(
                f"[{now} UTC] status={top.status} public={top.public_score} private={top.private_score}"
            )

            if status not in {"pending", "running"}:
                return top

            time.sleep(poll_seconds)

        return self.list_submissions(limit=1)[0]


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
