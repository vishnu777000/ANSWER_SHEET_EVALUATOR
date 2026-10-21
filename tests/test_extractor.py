import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytesseract

from modules.extractor import (
    _extract_image,
    _extract_pdf,
    _ocr_image,
    validate_file_magic,
)


class OcrImageTests(unittest.TestCase):

    def test_image_extraction_recovers_question_label_from_header(self):
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as uploaded:
            path = uploaded.name
        try:
            Image.new("RGB", (100, 100), color="white").save(path)
            with patch(
                "modules.extractor._ocr_image",
                side_effect=[
                    "Whaf ???",
                    "A readable answer with several words.",
                    "tiny",
                    "noise ###",
                    (
                        "Q1. A database is a collection of related "
                        "information stored electronically. It helps users "
                        "access, manage, and update data."
                    ),
                    "Q1",
                ],
            ) as ocr:
                self.assertEqual(
                    _extract_image(path),
                    (
                        "Q1. A database is a collection of related "
                        "information stored electronically. It helps users "
                        "access, manage, and update data."
                    ),
                )
        finally:
            os.remove(path)

        self.assertEqual(ocr.call_count, 6)
        self.assertEqual(
            [call.kwargs["config"] for call in ocr.call_args_list],
            [
                "--psm 6",
                "--psm 11",
                "--psm 6",
                "--psm 11",
                "--psm 6",
                "--psm 11",
            ],
        )

    def test_tesseract_paths_are_configured_before_ocr(self):
        def verify_configuration(image, lang):
            self.assertEqual(lang, "eng")
            self.assertEqual(
                os.environ["TESSDATA_PREFIX"],
                r"C:\Program Files\Tesseract-OCR\tessdata",
            )
            self.assertEqual(
                pytesseract.pytesseract.tesseract_cmd,
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            )
            return "recognized text"

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("modules.extractor.os.path.isfile", return_value=True),
            patch.object(pytesseract.pytesseract, "tesseract_cmd", "original"),
            patch(
                "pytesseract.image_to_string",
                side_effect=verify_configuration,
            ) as ocr,
        ):
            self.assertEqual(_ocr_image(object()), "recognized text")
            ocr.assert_called_once()

    def test_missing_english_language_data_has_actionable_error(self):
        error = RuntimeError(
            "Error opening data file "
            r"C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata. "
            "Failed loading language 'eng'!"
        )
        with patch("pytesseract.image_to_string", side_effect=error):
            with self.assertRaisesRegex(
                ValueError,
                "Install eng.traineddata.*TESSDATA_PREFIX",
            ):
                _ocr_image(object())

    def test_unexpected_ocr_errors_keep_their_details(self):
        with patch(
            "pytesseract.image_to_string",
            side_effect=RuntimeError("unexpected OCR error"),
        ):
            with self.assertRaisesRegex(ValueError, "OCR failed: unexpected OCR error"):
                _ocr_image(object())


class PdfExtractionTests(unittest.TestCase):

    def test_ocr_failure_on_scanned_page_is_reported_not_silently_skipped(self):
        page = SimpleNamespace(
            extract_text=lambda: None,
            to_image=lambda resolution: SimpleNamespace(original=object()),
        )
        pdf = SimpleNamespace(pages=[page])

        with patch("modules.extractor.pdfplumber.open") as open_pdf:
            open_pdf.return_value.__enter__.return_value = pdf
            with patch(
                "modules.extractor._ocr_image",
                side_effect=ValueError("OCR unavailable"),
            ):
                with self.assertRaisesRegex(ValueError, "PDF page 1"):
                    _extract_pdf("sample.pdf")

    def test_mixed_pdf_keeps_text_and_ocr_page_content(self):
        pages = [
            SimpleNamespace(
                extract_text=lambda: "Selectable page text",
            ),
            SimpleNamespace(
                extract_text=lambda: "",
                to_image=lambda resolution: SimpleNamespace(original=object()),
            ),
        ]
        pdf = SimpleNamespace(pages=pages)

        with patch("modules.extractor.pdfplumber.open") as open_pdf:
            open_pdf.return_value.__enter__.return_value = pdf
            with patch(
                "modules.extractor._ocr_image",
                return_value="OCR page text",
            ):
                self.assertEqual(
                    _extract_pdf("sample.pdf"),
                    "Selectable page text\nOCR page text",
                )

    def test_pdf_extension_requires_pdf_signature(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as uploaded:
            path = uploaded.name
            uploaded.write(b"not a PDF")
            uploaded.flush()

        try:
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_file_magic(path, ".pdf")
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
