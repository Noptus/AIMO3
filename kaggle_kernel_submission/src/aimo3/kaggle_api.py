"""Kaggle API automation helpers."""

from __future__ import annotations

import datetime as dt
import os
import re
import shutil
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


@dataclass(frozen=True)
class KernelPushRecord:
    kernel_ref: str
    version_number: int | None
    url: str | None


@dataclass(frozen=True)
class KernelRunRecord:
    kernel_ref: str
    status: str
    failure_message: str | None


def _load_kaggle_from_env(*, default_username: str | None = None) -> None:
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
        return

    # Also support key-only token style:
    # - KAGGLE_API_TOKEN=<key>
    # - KAGGLE_USERNAME must exist separately or be injected by caller.
    if default_username:
        os.environ.setdefault("KAGGLE_USERNAME", default_username.strip())
    os.environ.setdefault("KAGGLE_KEY", token.strip())


def _normalize_status(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    # Examples:
    # - "SubmissionStatus.PENDING"
    # - "KernelWorkerStatus.COMPLETE"
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


def _parse_submission_datetime(value: str) -> dt.datetime | None:
    text = str(value).strip()
    if not text:
        return None

    formats = (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
    )
    for fmt in formats:
        try:
            parsed = dt.datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    return None


def _normalize_kernel_ref(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""

    # URL form: https://www.kaggle.com/code/<owner>/<slug>
    m = re.search(r"/code/([^/]+)/([^/?#]+)", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    # API push ref form: /code/<owner>/<slug>
    if text.startswith("/code/"):
        pieces = [p for p in text.split("/") if p]
        if len(pieces) >= 3:
            return f"{pieces[1]}/{pieces[2]}"

    return text


class KaggleAutomation:
    """Thin wrapper around the official Kaggle API with submission polling."""

    def __init__(self, competition: str, *, default_username: str | None = None) -> None:
        self.competition = competition

        _load_kaggle_from_env(default_username=default_username)

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

    def submit_code(
        self,
        *,
        kernel: str,
        output_file_name: str,
        message: str,
        kernel_version: int | None = None,
    ) -> str:
        response = self.api.competition_submit_code(
            output_file_name,
            message,
            self.competition,
            kernel=kernel,
            kernel_version=kernel_version,
            quiet=True,
        )
        message_text = getattr(response, "message", None)
        return str(message_text if message_text else response)

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

    def count_submissions_on_utc_date(self, day: dt.date | None = None, *, limit: int = 100) -> int:
        target = day or dt.datetime.utcnow().date()
        total = 0
        for sub in self.list_submissions(limit=limit):
            parsed = _parse_submission_datetime(sub.submitted_at)
            if not parsed:
                continue
            if parsed.date() == target:
                total += 1
        return total

    def push_kernel(self, kernel_dir: str | Path, *, timeout_sec: int | None = None) -> KernelPushRecord:
        folder = Path(kernel_dir).expanduser()
        response = self.api.kernels_push(str(folder), timeout=timeout_sec)
        error = str(getattr(response, "error", "") or "").strip()
        if error:
            raise RuntimeError(f"Kernel push failed: {error}")

        version_value = getattr(response, "versionNumber", None)
        if version_value is None:
            version_value = getattr(response, "version_number", None)
        version_number: int | None = None
        if version_value is not None:
            try:
                version_number = int(version_value)
            except Exception:
                version_number = None

        kernel_ref = _normalize_kernel_ref(getattr(response, "ref", ""))
        if not kernel_ref:
            kernel_ref = _normalize_kernel_ref(getattr(response, "url", ""))
        url = _as_optional_str(getattr(response, "url", None))
        return KernelPushRecord(
            kernel_ref=kernel_ref,
            version_number=version_number,
            url=url,
        )

    def kernel_status(self, kernel: str) -> KernelRunRecord:
        kernel_ref = _normalize_kernel_ref(kernel)
        response = self.api.kernels_status(kernel_ref)
        raw_status = getattr(response, "status", "")
        normalized = _normalize_status(raw_status)
        return KernelRunRecord(
            kernel_ref=kernel_ref,
            status=normalized,
            failure_message=_as_optional_str(getattr(response, "failureMessage", None)),
        )

    def wait_for_kernel(
        self,
        kernel: str,
        *,
        poll_seconds: int = 20,
        timeout_minutes: int = 180,
    ) -> KernelRunRecord:
        deadline = time.time() + timeout_minutes * 60
        while time.time() < deadline:
            state = self.kernel_status(kernel)
            now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now} UTC] kernel_status={state.status}", flush=True)
            if state.status in {"complete", "error", "failed", "cancelled"}:
                return state
            time.sleep(poll_seconds)
        return self.kernel_status(kernel)

    def download_kernel_output(
        self,
        kernel: str,
        output_dir: str | Path,
        *,
        overwrite_dir: bool = False,
    ) -> list[Path]:
        kernel_ref = _normalize_kernel_ref(kernel)
        out = Path(output_dir).expanduser()
        if overwrite_dir and out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        self.api.kernels_output(kernel_ref, path=str(out), force=True, quiet=False)
        return sorted(p for p in out.iterdir() if p.is_file())

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

            status = _normalize_status(top.status)
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
