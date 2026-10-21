import io
import os
import tempfile
import unittest
from unittest.mock import patch

from app import app


KEY_TEXT = (
    "Q1. A database is an organized collection of related data "
    "stored electronically.\n"
)
STUDENT_TEXT = (
    "Q1. A database is a collection of related information stored electronically.\n"
)


class FlaskRouteTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.upload_dir = os.path.join(self.temp_dir.name, "uploads")
        self.results_file = os.path.join(self.temp_dir.name, "results", "results.xlsx")
        app.config.update(
            TESTING=True,
            UPLOAD_FOLDER=self.upload_dir,
            RESULTS_FILE=self.results_file,
            MAX_CONTENT_LENGTH=1024 * 1024,
        )
        self.client = app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()
        app.config.update(
            TESTING=False,
            UPLOAD_FOLDER="uploads",
            RESULTS_FILE="results/results.xlsx",
            MAX_CONTENT_LENGTH=32 * 1024 * 1024,
        )

    @staticmethod
    def _upload(text, filename):
        return io.BytesIO(text.encode("utf-8")), filename

    def _single_form(self, max_marks="5"):
        return {
            "answer_key": self._upload(KEY_TEXT, "answer-key.txt"),
            "student_answer": self._upload(STUDENT_TEXT, "student.txt"),
            "student_name": "Test Student",
            "reg_number": "TEST001",
            "subject": "Integration Tests",
            "max_marks": max_marks,
        }

    def test_home_and_results_pages_render(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/results").status_code, 200)

    def test_single_evaluation_saves_results_and_cleans_uploads(self):
        response = self.client.post("/evaluate", data=self._single_form())

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["results"]["max_total"], 5)
        self.assertTrue(os.path.isfile(self.results_file))
        self.assertEqual(os.listdir(self.upload_dir), [])

        stored = self.client.get("/results/Integration%20Tests")
        self.assertEqual(stored.status_code, 200)
        result_rows = stored.get_json()
        self.assertEqual(len(result_rows["rows"]), 1)
        self.assertEqual(result_rows["rows"][0]["Student Name"], "Test Student")
        download = self.client.get("/download")
        self.assertEqual(download.status_code, 200)
        download.close()

    def test_extraction_review_returns_text_and_removes_temporary_uploads(self):
        response = self.client.post(
            "/extract_for_review",
            data={
                "answer_key": self._upload(KEY_TEXT, "answer-key.txt"),
                "student_answer": self._upload(STUDENT_TEXT, "student.txt"),
                "mode": "single",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["answer_key_text"], KEY_TEXT.strip())
        self.assertEqual(
            payload["student_answers"][0]["text"],
            STUDENT_TEXT.strip(),
        )
        self.assertEqual(os.listdir(self.upload_dir), [])

    def test_single_evaluation_uses_edited_review_text(self):
        reviewed_student = "Q1. A collection of related data stored electronically."
        reviewed_answer_body = "A collection of related data stored electronically."
        response = self.client.post(
            "/evaluate",
            data={
                "reviewed_key_text": KEY_TEXT,
                "reviewed_student_text": reviewed_student,
                "student_name": "Test Student",
                "reg_number": "TEST001",
                "subject": "Reviewed Tests",
                "max_marks": "5",
            },
        )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()["results"]
        question = result["questions"][0]
        self.assertEqual(question["student_answer"], reviewed_answer_body)
        self.assertEqual(question["marks"], round(5 * question["coverage"], 2))
        self.assertEqual(
            result["percentage"],
            round(result["total_marks"] / result["max_total"] * 100, 2),
        )

    def test_batch_evaluation_uses_edited_review_text(self):
        reviewed_student = "Q1. A collection of related data stored electronically."
        reviewed_answer_body = "A collection of related data stored electronically."
        response = self.client.post(
            "/evaluate_batch",
            data={
                "reviewed_key_text": KEY_TEXT,
                "reviewed_answers": '[{"filename":"TEST001_Student.txt","text":"'
                + reviewed_student
                + '"}]',
                "subject": "Reviewed Batch Tests",
                "max_marks": "5",
            },
        )

        self.assertEqual(response.status_code, 200)
        record = response.get_json()["batch_results"][0]
        self.assertEqual(record["status"], "Success")
        self.assertEqual(
            record["results"]["questions"][0]["student_answer"],
            reviewed_answer_body,
        )
        self.assertFalse(os.path.exists(self.upload_dir))

    def test_invalid_maximum_marks_returns_validation_error(self):
        response = self.client.post(
            "/evaluate",
            data=self._single_form(max_marks="not-a-number"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("whole number", response.get_json()["errors"][0])
        self.assertFalse(os.path.exists(self.upload_dir))

    def test_bad_file_signature_is_rejected_and_upload_is_removed(self):
        form = self._single_form()
        form["answer_key"] = (io.BytesIO(b"not a PDF"), "answer-key.pdf")

        response = self.client.post("/evaluate", data=form)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(os.listdir(self.upload_dir), [])

    def test_unsegmented_answer_is_rejected_and_uploads_are_removed(self):
        form = self._single_form()
        form["student_answer"] = (
            io.BytesIO(b"This answer has no question label."),
            "student.txt",
        )

        response = self.client.post("/evaluate", data=form)

        self.assertEqual(response.status_code, 400)
        self.assertIn("No numbered student answers", response.get_json()["errors"][0])
        self.assertEqual(os.listdir(self.upload_dir), [])

    def test_invalid_subject_is_rejected_before_saving_uploads(self):
        form = self._single_form()
        form["subject"] = "Invalid/Subject"

        response = self.client.post("/evaluate", data=form)

        self.assertEqual(response.status_code, 400)
        self.assertFalse(os.path.exists(self.upload_dir))

    def test_storage_error_is_generic_and_uploads_are_removed(self):
        with patch("app.save_to_excel", side_effect=OSError("private path detail")):
            response = self.client.post("/evaluate", data=self._single_form())

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private path detail", response.get_data(as_text=True))
        self.assertEqual(os.listdir(self.upload_dir), [])

    def test_request_size_limit_returns_json_error(self):
        app.config["MAX_CONTENT_LENGTH"] = 100
        response = self.client.post(
            "/evaluate",
            data={
                "answer_key": self._upload(KEY_TEXT, "key.txt"),
                "student_answer": self._upload(STUDENT_TEXT, "student.txt"),
            },
        )

        self.assertEqual(response.status_code, 413)
        self.assertIn("configured request size", response.get_json()["errors"][0])

    def test_preview_and_batch_evaluate_files_independently(self):
        preview = self.client.post(
            "/preview",
            data={
                "answer_key": self._upload(KEY_TEXT, "answer-key.txt"),
                "role": "key",
            },
        )
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.get_json()["question_count"], 1)
        self.assertEqual(os.listdir(self.upload_dir), [])

        batch = self.client.post(
            "/evaluate_batch",
            data={
                "answer_key": self._upload(KEY_TEXT, "answer-key.txt"),
                "student_answers": [
                    self._upload(STUDENT_TEXT, "21CS001_Alice.txt"),
                    (io.BytesIO(b"bad pdf"), "21CS002_Bob.pdf"),
                ],
                "subject": "Batch Tests",
                "max_marks": "5",
            },
        )
        self.assertEqual(batch.status_code, 200)
        payload = batch.get_json()
        self.assertEqual(payload["successful_evaluations"], 1)
        self.assertEqual(
            [record["status"] for record in payload["batch_results"]],
            ["Success", "Failed"],
        )
        self.assertEqual(os.listdir(self.upload_dir), [])


if __name__ == "__main__":
    unittest.main()
