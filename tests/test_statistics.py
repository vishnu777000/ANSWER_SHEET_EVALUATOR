import os
import tempfile
import unittest

from modules.evaluation_statistics import (
    _load_benchmark_cases,
    get_evaluation_statistics,
    get_saved_results_summary,
)
from modules.storage import save_to_excel


class EvaluationStatisticsTests(unittest.TestCase):

    def test_test_benchmark_statistics_omit_binary_metrics(self):
        statistics = get_evaluation_statistics()
        cases = statistics["cases"]

        self.assertEqual(statistics["split"], "test")
        self.assertEqual(statistics["sample_size"], 102)
        self.assertEqual(len(cases), statistics["sample_size"])
        self.assertEqual(statistics["threshold"], 0.375)
        self.assertNotIn("metrics", statistics)
        self.assertNotIn("confusion_matrix", statistics)
        self.assertTrue(all("correct" in case for case in cases))
        self.assertIn("Artificial intelligence", {case["topic"] for case in cases})

    def test_development_cases_include_all_four_metrics(self):
        statistics = get_evaluation_statistics(split="development")

        self.assertEqual(statistics["threshold"], 0.375)
        self.assertEqual(statistics["sample_size"], 30)
        self.assertEqual(statistics["split"], "development")
        self.assertAlmostEqual(statistics["metrics"]["accuracy"], 24 / 30)
        self.assertAlmostEqual(statistics["metrics"]["precision"], 16 / 20)
        self.assertAlmostEqual(statistics["metrics"]["recall"], 16 / 18)
        self.assertAlmostEqual(
            statistics["metrics"]["f1_score"],
            2 * (16 / 20) * (16 / 18) / ((16 / 20) + (16 / 18)),
        )
        self.assertEqual(
            set(statistics["metrics"]),
            {"accuracy", "precision", "recall", "f1_score"},
        )

    def test_benchmark_is_populated_and_topic_held_out(self):
        cases = _load_benchmark_cases()
        development = [case for case in cases if case["split"] == "development"]
        test = [case for case in cases if case["split"] == "test"]

        self.assertEqual(len(cases), 132)
        self.assertEqual(len(development), 30)
        self.assertEqual(len(test), 102)
        self.assertEqual(
            {case["topic"] for case in development}
            & {case["topic"] for case in test},
            set(),
        )
        self.assertEqual(
            {case["expected_label"] for case in cases},
            {"acceptable", "partial", "incorrect"},
        )
        case_ids = {case["case_id"] for case in test}
        self.assertTrue(any(case_id.startswith("NLP") for case_id in case_ids))
        self.assertTrue(any(case_id.startswith("ML") for case_id in case_ids))
        self.assertTrue(any(case_id.startswith("CD") for case_id in case_ids))
        self.assertTrue(any(case_id.startswith("CN") for case_id in case_ids))
        self.assertTrue(any(case["provenance"] == "project_sample" for case in cases))
        self.assertTrue(any(case["provenance"] == "curated_variant" for case in cases))

    def test_saved_result_summary_uses_gradebook_records(self):
        with tempfile.TemporaryDirectory() as directory:
            filepath = os.path.join(directory, "results.xlsx")
            for name, percentage in (("Ada", 80), ("Lin", 60)):
                save_to_excel(
                    filepath,
                    "Databases",
                    name,
                    name.upper(),
                    {
                        "questions": [{"label": "Q1", "marks": percentage}],
                        "total_marks": percentage,
                        "percentage": percentage,
                    },
                )

            summary = get_saved_results_summary(filepath)

        self.assertEqual(summary["student_count"], 2)
        self.assertEqual(summary["subject_count"], 1)
        self.assertEqual(summary["average_percentage"], 70)
        subject = summary["subjects"][0]
        self.assertEqual(subject["subject"], "Databases")
        self.assertEqual(subject["student_count"], 2)
        self.assertEqual(subject["highest_percentage"], 80)
        self.assertEqual(subject["lowest_percentage"], 60)
        self.assertEqual(
            [student["student_name"] for student in subject["students"]],
            ["Ada", "Lin"],
        )


if __name__ == "__main__":
    unittest.main()
