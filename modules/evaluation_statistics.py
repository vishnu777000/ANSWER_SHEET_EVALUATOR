"""Run a small, curated benchmark through the answer-evaluation rubric."""

import csv
import math
from pathlib import Path

from modules.local_rubric import DECISION_THRESHOLD, evaluate_answer
from modules.storage import get_results_for_subject, get_subjects


BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "answer_benchmark.csv"
)
BENCHMARK_LABELS = {"acceptable", "partial", "incorrect"}


def _load_benchmark_cases():
    with BENCHMARK_PATH.open(encoding="utf-8-sig", newline="") as benchmark_file:
        reader = csv.DictReader(benchmark_file)
        required_columns = {
            "case_id",
            "topic",
            "reference_answer",
            "student_answer",
            "expected_label",
            "split",
            "provenance",
        }
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"Benchmark file must contain columns: "
                f"{', '.join(sorted(required_columns))}."
            )

        cases = []
        seen_ids = set()
        for line_number, row in enumerate(reader, start=2):
            if not all(row.get(column, "").strip() for column in required_columns):
                raise ValueError(
                    f"Benchmark row {line_number} has one or more empty required fields."
                )
            if row["case_id"] in seen_ids:
                raise ValueError(
                    f"Benchmark row {line_number} repeats case ID {row['case_id']}."
                )
            if row["expected_label"] not in BENCHMARK_LABELS:
                raise ValueError(
                    f"Benchmark row {line_number} has an unsupported expected label."
                )
            seen_ids.add(row["case_id"])
            cases.append(row)

    if not cases:
        raise ValueError("Benchmark file contains no test cases.")
    return cases


def get_evaluation_statistics(split="test"):
    """Run the selected CSV benchmark split through the current answer rubric."""
    all_cases = _load_benchmark_cases()
    available_splits = {case["split"] for case in all_cases}
    if split not in available_splits:
        raise ValueError(
            f"Benchmark split {split!r} is unavailable; "
            f"available splits: {', '.join(sorted(available_splits))}."
        )

    cases = []

    benchmark_cases = [case for case in all_cases if case["split"] == split]
    for benchmark_case in benchmark_cases:
        grading = evaluate_answer(
            benchmark_case["student_answer"],
            benchmark_case["reference_answer"],
            max_marks=1,
            use_semantic=False,
        )
        coverage = grading["coverage"]
        predicted_pass = coverage >= DECISION_THRESHOLD
        expected_label = benchmark_case["expected_label"]
        expected_pass = expected_label == "acceptable"

        cases.append({
            "id": benchmark_case["case_id"],
            "topic": benchmark_case["topic"],
            "answer": benchmark_case["student_answer"] or "(blank answer)",
            "reference": benchmark_case["reference_answer"],
            "expected_label": expected_label,
            "provenance": benchmark_case["provenance"],
            "expected": expected_pass,
            "predicted": predicted_pass,
            "coverage": coverage,
            "correct": expected_pass == predicted_pass,
        })

    statistics = {
        "sample_size": len(cases),
        "split": split,
        "threshold": DECISION_THRESHOLD,
        "cases": cases,
    }
    if split == "development":
        true_positive = sum(
            case["expected"] and case["predicted"] for case in cases
        )
        true_negative = sum(
            not case["expected"] and not case["predicted"] for case in cases
        )
        false_positive = sum(
            not case["expected"] and case["predicted"] for case in cases
        )
        false_negative = sum(
            case["expected"] and not case["predicted"] for case in cases
        )
        correct_count = sum(case["correct"] for case in cases)
        accuracy = correct_count / len(cases) if cases else 0.0
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )
        statistics["metrics"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            ),
        }
    return statistics


def get_saved_results_summary(filepath):
    """Summarize real, saved gradebook percentages without mixing in benchmark data."""
    subject_summaries = []
    all_percentages = []

    for subject in get_subjects(filepath):
        result = get_results_for_subject(filepath, subject)
        percentages = []
        for row in result["rows"]:
            raw_percentage = row.get("Percentage", "").strip().removesuffix("%")
            try:
                percentage = float(raw_percentage)
            except (TypeError, ValueError):
                continue
            if math.isfinite(percentage) and 0 <= percentage <= 100:
                percentages.append({
                    "student_name": row.get("Student Name", ""),
                    "reg_number": row.get("Register No.", ""),
                    "percentage": percentage,
                })

        if not percentages:
            continue

        scores = [record["percentage"] for record in percentages]
        all_percentages.extend(scores)
        subject_summaries.append({
            "subject": subject,
            "student_count": len(scores),
            "average_percentage": sum(scores) / len(scores),
            "highest_percentage": max(scores),
            "lowest_percentage": min(scores),
            "students": percentages,
        })

    return {
        "student_count": len(all_percentages),
        "subject_count": len(subject_summaries),
        "average_percentage": (
            sum(all_percentages) / len(all_percentages)
            if all_percentages else None
        ),
        "subjects": subject_summaries,
    }
