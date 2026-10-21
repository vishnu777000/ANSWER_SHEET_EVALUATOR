
import unittest

from modules.evaluator import evaluate_answers


class EvaluatorTests(unittest.TestCase):

    def test_result_totals(self):
        students = [
            {
                "question_number": 1,
                "answer": "A database stores organized data."
            }
        ]

        references = [
            {
                "question_number": 1,
                "answer": "A database stores organized data.",
                "max_marks": 5
            }
        ]

        result = evaluate_answers(
            key_questions=references,
            ans_questions=students,
            max_marks=5,
        )

        self.assertIn("questions", result)
        self.assertIn("total_marks", result)
        self.assertIn("max_total", result)
        self.assertIn("percentage", result)
        self.assertEqual(result["max_total"], 5.0)
        self.assertEqual(result["total_marks"], 5.0)
        self.assertEqual(result["percentage"], 100.0)
        self.assertEqual(
            result["percentage"],
            round(result["total_marks"] / result["max_total"] * 100, 2),
        )

    def test_missing_answer_gets_zero(self):
        students = [
            {"question_number": 1, "answer": ""}
        ]

        references = [
            {
                "question_number": 1,
                "answer": "A database stores organized data.",
                "max_marks": 5
            }
        ]

        result = evaluate_answers(
            key_questions=references,
            ans_questions=students,
            max_marks=5,
        )

        self.assertEqual(result["total_marks"], 0.0)

    def test_unmatched_question_gets_zero(self):
        students = [
            {"question_number": 2, "answer": "Some answer"}
        ]

        references = [
            {
                "question_number": 1,
                "answer": "Reference answer",
                "max_marks": 5
            }
        ]

        result = evaluate_answers(
            key_questions=references,
            ans_questions=students,
            max_marks=5,
        )

        self.assertEqual(result["total_marks"], 0.0)

    def test_subquestion_labels_match_without_overwriting_sibling_answers(self):
        result = evaluate_answers(
            key_questions={
                "Q1(a)": "A database stores organized data.",
                "Q1(b)": "A table organizes records into rows.",
            },
            ans_questions={
                "Q1(a)": "A database stores organized data.",
                "Q1(b)": "A table organizes records into rows.",
            },
            max_marks=5,
        )

        self.assertEqual(result["total_marks"], 10.0)
        self.assertEqual(
            [question["student_answer"] for question in result["questions"]],
            [
                "A database stores organized data.",
                "A table organizes records into rows.",
            ],
        )


if __name__ == "__main__":
    unittest.main()
