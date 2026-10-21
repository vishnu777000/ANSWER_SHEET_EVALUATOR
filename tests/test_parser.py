import unittest

from modules.parser import segment_questions, segment_reference_answers


class ParserTests(unittest.TestCase):

    def test_removes_question_prompt_before_answer_on_next_line(self):
        text = (
            "Q1. What is deep learning?\n"
            "Answer: Deep learning uses neural networks with many layers.\n"
            "Q2. What is a model?\n"
            "Answer: A model makes predictions from data."
        )

        self.assertEqual(
            segment_questions(text),
            {
                "Q1": "Deep learning uses neural networks with many layers.",
                "Q2": "A model makes predictions from data.",
            },
        )

    def test_removes_inline_question_prompt_and_answer_label(self):
        text = (
            "Question 1: What is deep learning? Answer: "
            "Deep learning uses neural networks with many layers."
        )

        self.assertEqual(
            segment_questions(text),
            {"Q1": "Deep learning uses neural networks with many layers."},
        )

    def test_recognizes_pdf_question_number_variants(self):
        text = (
            "Q. 1 What is deep learning?\n"
            "Answer: Deep learning uses neural networks.\n"
            "Question No. 2 What is a model?\n"
            "Answer: A model makes predictions."
        )

        self.assertEqual(
            segment_questions(text),
            {
                "Q1": "Deep learning uses neural networks.",
                "Q2": "A model makes predictions.",
            },
        )

    def test_keeps_student_answer_without_answer_label(self):
        text = "Q1. Deep learning uses neural networks."

        self.assertEqual(
            segment_questions(text),
            {"Q1": "Deep learning uses neural networks."},
        )

    def test_recognizes_bare_number_on_its_own_line(self):
        text = (
            "1\n"
            "A database stores related information electronically.\n"
            "2\n"
            "It helps users access and manage data."
        )

        self.assertEqual(
            segment_questions(text),
            {
                "Q1": "A database stores related information electronically.",
                "Q2": "It helps users access and manage data.",
            },
        )

    def test_recognizes_parenthesized_and_unpunctuated_question_numbers(self):
        text = (
            "(1) A database stores related information.\n"
            "2 A database can be queried."
        )

        self.assertEqual(
            segment_questions(text),
            {
                "Q1": "A database stores related information.",
                "Q2": "A database can be queried.",
            },
        )

    def test_does_not_treat_table_rows_as_unpunctuated_question_labels(self):
        text = (
            "Q1. Answer text.\n"
            "2 | Related data Data in the collection is related. | 1"
        )

        self.assertEqual(
            list(segment_questions(text)),
            ["Q1"],
        )

    def test_reference_answer_stops_before_rubric_sections(self):
        text = (
            "Q1. What is a database? [5 Marks]\n"
            "Model Answer:\n"
            "A database is an organized collection of related data stored "
            "electronically so that it can be accessed, managed, and updated "
            "efficiently.\n"
            "Key Points and Evaluation Scheme:\n"
            "S.No. Key Concept Expected Content Marks\n"
            "1 Collection It is a collection of data. 1\n"
            "Guidelines for Partial Credit:\n"
            "Correctly defines collection but misses other points.\n"
            "Q2. Define related data.\n"
            "Model Answer:\n"
            "Related data is information connected by a common subject."
        )

        self.assertEqual(
            segment_reference_answers(text),
            {
                "Q1": (
                    "A database is an organized collection of related data "
                    "stored electronically so that it can be accessed, managed, "
                    "and updated efficiently."
                ),
                "Q2": (
                    "Related data is information connected by a common subject."
                ),
            },
        )

    def test_removes_primary_key_prompt_and_mark_annotation(self):
        text = (
            "Q2. Explain a primary key. [5 marks] "
            "A primary key is a column or set of columns that uniquely "
            "identifies each row in a table. It must contain unique values "
            "and cannot contain NULL values."
        )

        self.assertEqual(
            segment_questions(text),
            {
                "Q2": (
                    "A primary key is a column or set of columns that uniquely "
                    "identifies each row in a table. It must contain unique "
                    "values and cannot contain NULL values."
                )
            },
        )

    def test_segments_numbered_subquestions_without_collapsing_labels(self):
        text = (
            "Q1(a) Define a database.\n"
            "A database stores organized data.\n"
            "1(b). Explain tables.\n"
            "Tables organize data into rows and columns."
        )

        self.assertEqual(
            segment_questions(text),
            {
                "Q1(a)": "A database stores organized data.",
                "Q1(b)": "Tables organize data into rows and columns.",
            },
        )


if __name__ == "__main__":
    unittest.main()
