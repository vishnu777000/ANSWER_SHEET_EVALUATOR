import unittest
from unittest.mock import patch

from modules.local_rubric import evaluate_answer
from modules.parser import segment_questions


REFERENCE = (
    "A database is an organized collection of related data stored "
    "electronically so that it can be accessed, managed, and updated efficiently."
)
PARAPHRASE = (
    "What is a database? A database is a collection of related information "
    "stored electronically. It helps users access, manage, and update data."
)
NORMALIZATION_REFERENCE = (
    "Normalization is the process of organizing tables in a relational database "
    "to reduce data redundancy and prevent insertion, update, and deletion "
    "anomalies. It commonly uses normal forms such as 1NF, 2NF, and 3NF."
)
NORMALIZATION_STUDENT = (
    "Normalization organizes database tables and reduces duplicate data. "
    "It helps avoid update problems."
)


class LocalRubricTests(unittest.TestCase):

    def test_normalization_paraphrase_gets_credit_for_synonyms_and_inflections(self):
        result = evaluate_answer(
            NORMALIZATION_STUDENT,
            NORMALIZATION_REFERENCE,
            10,
        )

        self.assertGreaterEqual(result["marks"], 6.0)
        self.assertIn("organization", result["covered_concepts"])
        self.assertIn("redundancy", result["covered_concepts"])
        self.assertIn("prevention", result["covered_concepts"])
        self.assertIn("anomaly", result["covered_concepts"])
        self.assertIn("normal forms", result["missing_concepts"])

    def test_exact_match_gets_full_marks(self):
        result = evaluate_answer(REFERENCE, REFERENCE, 7)

        self.assertEqual(result["marks"], 7)
        self.assertEqual(result["coverage"], 1)

    def test_primary_key_answer_from_screenshot_scores_as_correct(self):
        reference = segment_questions(
            "Q2. Explain a primary key. [5 marks] A primary key is a column or "
            "set of columns that uniquely identifies each row in a table. It "
            "must contain unique values and cannot contain NULL values."
        )["Q2"]
        student = segment_questions(
            "Q2. Explain a primary key. A primary key uniquely identifies every "
            "row in a table. Values must be unique and it cannot be NULL."
        )["Q2"]

        result = evaluate_answer(student, reference, 10)

        self.assertGreaterEqual(result["marks"], 7.0)
        self.assertNotIn("null", result["incorrect_concepts"])
        self.assertIn("unique", result["covered_concepts"])
        self.assertIn("column", result["missing_concepts"])

    def test_matching_negated_null_constraint_is_not_a_contradiction(self):
        result = evaluate_answer(
            "A primary key cannot be NULL.",
            "A primary key cannot contain NULL values.",
            5,
        )

        self.assertNotIn("null", result["incorrect_concepts"])

    def test_database_paraphrase_gets_substantially_more_than_wrong_answer(self):
        paraphrase = evaluate_answer(PARAPHRASE, REFERENCE, 10)
        wrong = evaluate_answer(
            "A database is an unrelated fruit that grows on trees.",
            REFERENCE,
            10,
        )

        self.assertGreaterEqual(paraphrase["marks"], 7.0)
        self.assertGreater(paraphrase["marks"], wrong["marks"] + 5)
        self.assertIn("electronic storage", paraphrase["covered_concepts"])
        self.assertIn("organization", paraphrase["missing_concepts"])
        self.assertIn("Concepts covered:", paraphrase["feedback"])

    def test_partial_answer_scores_below_full_paraphrase(self):
        partial = evaluate_answer(
            "A database is a collection of related information.",
            REFERENCE,
            10,
        )
        complete = evaluate_answer(PARAPHRASE, REFERENCE, 10)

        self.assertGreater(partial["marks"], 0)
        self.assertLess(partial["marks"], complete["marks"])
        self.assertIn("access", partial["missing_concepts"])

    def test_empty_answer_gets_zero(self):
        result = evaluate_answer("", REFERENCE, 10)

        self.assertEqual(result["marks"], 0)
        self.assertIn("No student answer", result["feedback"])

    def test_repeating_text_does_not_increase_marks(self):
        once = evaluate_answer(
            "A database is a collection of related information.",
            REFERENCE,
            10,
        )
        repeated = evaluate_answer(
            "A database is a collection of related information. " * 8,
            REFERENCE,
            10,
        )

        self.assertEqual(repeated["marks"], once["marks"])

    def test_irrelevant_content_does_not_change_concept_coverage_score(self):
        accurate = evaluate_answer(PARAPHRASE, REFERENCE, 10)
        padded = evaluate_answer(
            PARAPHRASE
            + " The answer also discusses unrelated astronomy, sports, and cooking.",
            REFERENCE,
            10,
        )

        self.assertEqual(padded["marks"], accurate["marks"])
        self.assertEqual(padded["coverage"], accurate["coverage"])

    def test_negated_reference_concepts_are_penalized_and_reported(self):
        affirmative = evaluate_answer(
            "A database is a collection of related information stored electronically "
            "and can be accessed, managed, and updated.",
            REFERENCE,
            10,
        )
        negated = evaluate_answer(
            "A database is not a collection of related information and cannot be accessed.",
            REFERENCE,
            10,
        )

        self.assertLess(negated["marks"], affirmative["marks"])
        self.assertIn("collection", negated["incorrect_concepts"])
        self.assertIn("access", negated["incorrect_concepts"])
        self.assertIn("contradicted", negated["feedback"])

    def test_marks_scale_with_configured_maximum(self):
        five_marks = evaluate_answer(PARAPHRASE, REFERENCE, 5)
        twenty_marks = evaluate_answer(PARAPHRASE, REFERENCE, 20)
        zero_marks = evaluate_answer(PARAPHRASE, REFERENCE, 0)

        self.assertAlmostEqual(
            twenty_marks["marks"],
            five_marks["marks"] * 4,
            delta=0.02,
        )
        self.assertLessEqual(twenty_marks["marks"], 20)
        self.assertEqual(zero_marks["marks"], 0)

    def test_marks_are_equal_to_concept_coverage_of_maximum(self):
        result = evaluate_answer(PARAPHRASE, REFERENCE, 10)

        self.assertEqual(result["similarity"], result["coverage"])
        self.assertEqual(result["marks"], round(10 * result["coverage"], 2))
        self.assertIn(
            f"{len(result['covered_concepts'])} of "
            f"{len(result['covered_concepts']) + len(result['missing_concepts'])} "
            "reference concepts covered",
            result["feedback"],
        )

    def test_related_information_and_related_data_receive_equal_credit(self):
        result = evaluate_answer(
            "A collection of related data.",
            "A collection of related information.",
            5,
        )

        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["marks"], 5.0)

    def test_electronic_storage_and_stored_electronically_are_equivalent(self):
        result = evaluate_answer(
            "The system has electronic storage.",
            "The system is stored electronically.",
            5,
        )

        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["marks"], 5.0)

    def test_non_finite_maximum_is_rejected(self):
        for maximum in (float("nan"), float("inf")):
            with self.subTest(maximum=maximum):
                with self.assertRaises(ValueError):
                    evaluate_answer(PARAPHRASE, REFERENCE, maximum)

    def test_missing_model_fails_explicitly_when_semantic_mode_is_requested(self):
        with patch(
            "modules.local_rubric._semantic_similarity",
            side_effect=OSError("model not cached"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "Semantic scoring was requested.*local.*model is unavailable",
            ):
                evaluate_answer(PARAPHRASE, REFERENCE, 10, use_semantic=True)

    def test_semantic_mode_reports_combined_scoring_method(self):
        with patch(
            "modules.local_rubric._semantic_similarity",
            return_value=0.9,
        ):
            result = evaluate_answer(PARAPHRASE, REFERENCE, 10, use_semantic=True)

        self.assertEqual(result["scoring_method"], "semantic-and-concept")
        self.assertEqual(result["semantic_similarity"], 0.9)
        self.assertEqual(result["marks"], round(10 * result["coverage"], 2))


if __name__ == "__main__":
    unittest.main()
