#!/usr/bin/env python3
"""Build machine-usable reference CSVs from extracted AIMO3 reference text."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def _normalize_problem_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u2019", "'")
    text = text.replace("\u00d7", "x")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_dataset(input_txt: Path, output_dir: Path) -> None:
    raw = input_txt.read_text(encoding="utf-8", errors="ignore")

    pattern = re.compile(
        r"\nProblem\s+(\d+)\nProblem:\s*(.*?)\nAnswer:\s*([0-9]+)",
        flags=re.DOTALL,
    )

    rows: list[dict[str, str | int]] = []
    for match in pattern.finditer(raw):
        idx = int(match.group(1))
        statement = _normalize_problem_text(match.group(2))
        answer = int(match.group(3))
        rows.append({"id": idx, "problem": statement, "answer": answer})

    if len(rows) != 10:
        raise RuntimeError(f"Expected 10 reference problems, parsed {len(rows)}")

    rows.sort(key=lambda row: int(row["id"]))

    output_dir.mkdir(parents=True, exist_ok=True)

    with_answers = output_dir / "reference_with_answers.csv"
    with with_answers.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "problem", "answer"])
        writer.writeheader()
        writer.writerows(rows)

    problems_only = output_dir / "reference_problems.csv"
    with problems_only.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "problem"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], "problem": row["problem"]})

    answers_only = output_dir / "reference_answers.csv"
    with answers_only.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], "answer": row["answer"]})

    print(f"Wrote {with_answers}")
    print(f"Wrote {problems_only}")
    print(f"Wrote {answers_only}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reference CSV files from extracted AIMO3 PDF text")
    parser.add_argument(
        "--input-txt",
        default="reference/AIMO3_Reference_Problems.extracted.clean.txt",
        help="Path to extracted text file",
    )
    parser.add_argument("--output-dir", default="reference")
    args = parser.parse_args()

    build_dataset(Path(args.input_txt), Path(args.output_dir))


if __name__ == "__main__":
    main()
