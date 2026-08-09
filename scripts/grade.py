#!/usr/bin/env python
"""Local grading helper — mirrors exactly what the GitHub Actions workflow
(`.github/workflows/grading.yml`) runs, so you can check your weighted
score before pushing.

Usage:
    python scripts/grade.py

Writes `grade_report.json` and prints a weighted score table to stdout,
using the same rubric weights as docs/RUBRIC.md.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "report.json"
GRADE_REPORT_PATH = REPO_ROOT / "grade_report.json"

# rubric category -> (pytest marker, weight in the auto-gradable 95%)
# See docs/RUBRIC.md. "Portfolio & Presentation" (5%) is manually graded
# and intentionally excluded here.
CATEGORIES: list[tuple[str, str, float]] = [
    ("VAE Implementation", "vae", 20.0),
    ("GAN Implementation", "gan", 30.0),
    ("Advanced Techniques", "advanced", 20.0),
    ("Evaluation & Analysis", "evaluation", 15.0),
    ("Platform Development", "platform", 10.0),
]


def run_pytest() -> dict:
    """Run the full suite once with a JSON report, tolerating test failures."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--json-report",
        f"--json-report-file={REPORT_PATH}",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT)  # non-zero exit on test failures is expected mid-course
    with open(REPORT_PATH, encoding="utf-8") as f:
        return json.load(f)


def bucket_tests(report: dict) -> dict[str, list[dict]]:
    """Group every collected test by which rubric marker(s) it carries."""
    buckets: dict[str, list[dict]] = {marker: [] for _, marker, _ in CATEGORIES}
    buckets["unmarked"] = []
    for test in report.get("tests", []):
        keywords = test.get("keywords", {})
        matched = False
        for _, marker, _ in CATEGORIES:
            if marker in keywords:
                buckets[marker].append(test)
                matched = True
        if not matched:
            buckets["unmarked"].append(test)
    return buckets


def score_report(report: dict) -> dict:
    buckets = bucket_tests(report)
    rows = []
    total_score = 0.0
    total_weight = 0.0

    for label, marker, weight in CATEGORIES:
        tests = buckets[marker]
        total = len(tests)
        passed = sum(1 for t in tests if t.get("outcome") == "passed")
        fraction = (passed / total) if total else 0.0
        earned = fraction * weight
        total_score += earned
        total_weight += weight
        rows.append(
            {
                "category": label,
                "marker": marker,
                "passed": passed,
                "total": total,
                "weight": weight,
                "earned": round(earned, 2),
            }
        )

    return {
        "rows": rows,
        "auto_gradable_score": round(total_score, 2),
        "auto_gradable_weight": total_weight,
        "manual_review_note": "Portfolio & Presentation (5%) is graded manually by your instructor.",
        "summary": report.get("summary", {}),
    }


def print_table(result: dict) -> None:
    print("\n=== Advanced Generative Art Studio — Grade Report ===\n")
    print(f"{'Category':<24}{'Tests':<12}{'Weight':<10}{'Earned':<10}")
    print("-" * 56)
    for row in result["rows"]:
        tests_str = f"{row['passed']}/{row['total']}"
        print(f"{row['category']:<24}{tests_str:<12}{row['weight']:<10}{row['earned']:<10}")
    print("-" * 56)
    print(f"{'TOTAL (auto-gradable)':<24}{'':<12}{result['auto_gradable_weight']:<10}{result['auto_gradable_score']:<10}")
    print(f"\nNote: {result['manual_review_note']}")
    print(f"Full auto-gradable total is out of {result['auto_gradable_weight']:.0f} points ")
    print("(the remaining 5% — Portfolio & Presentation — is scored by your instructor).\n")


def main() -> None:
    report = run_pytest()
    result = score_report(report)
    with open(GRADE_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print_table(result)


if __name__ == "__main__":
    main()
