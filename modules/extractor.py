"""
modules/extractor.py — Text Extraction Module
==============================================
Extracts plain text from various file formats:
  PDF  → pdfplumber (with OCR fallback via Tesseract)
  Image → pytesseract OCR
  DOCX → python-docx
  TXT   → direct read
  XLSX/XLS → openpyxl / pandas (concatenate all cell values)
"""

import os
import re

import pdfplumber
import docx
import pandas as pd


# ── Magic-byte signatures (first bytes → expected extension) ─────────────────
_MAGIC_SIGNATURES = [
    (b"%PDF", {".pdf"}),
    (b"\x89PNG\r\n\x1a\n", {".png"}),
    (b"\xff\xd8\xff", {".jpg", ".jpeg"}),
    (b"PK\x03\x04", {".docx", ".xlsx", ".xls"}),  # ZIP-based Office formats
]


def _read_header(filepath: str, n: int = 8) -> bytes:
    with open(filepath, "rb") as f:
        return f.read(n)


def validate_file_magic(filepath: str, ext: str) -> None:
    """
    Verify file content matches its extension using magic bytes.
    TXT and legacy .xls (non-ZIP) skip strict checks when ambiguous.
    """
    ext = ext.lower()
    if ext in {".txt"}:
        return

    header = _read_header(filepath)
    if not header:
        raise ValueError("File is empty or could not be read.")

    if ext in {".xls"}:
        # Legacy Excel may be OLE compound document
        if header.startswith(b"PK\x03\x04") or header.startswith(b"\xd0\xcf\x11\xe0"):
            return
        raise ValueError(
            f"File content does not look like a valid Excel workbook (got extension {ext})."
        )

    matched = False
    for magic, allowed_exts in _MAGIC_SIGNATURES:
        if header.startswith(magic):
            matched = True
            if ext not in allowed_exts:
                expected = ", ".join(sorted(allowed_exts))
                raise ValueError(
                    f"File extension '{ext}' does not match file content "
                    f"(expected one of: {expected})."
                )
            break

    expected_by_ext = {
        ".pdf": b"%PDF",
        ".png": b"\x89PNG",
        ".jpg": b"\xff\xd8\xff",
        ".jpeg": b"\xff\xd8\xff",
        ".docx": b"PK\x03\x04",
        ".xlsx": b"PK\x03\x04",
    }
    sig = expected_by_ext.get(ext)
    if sig and not header.startswith(sig):
        raise ValueError(
            f"File content does not match a valid {ext} document. "
            "The file may be corrupted or mislabeled."
        )
    if ext == ".docx" and matched and header.startswith(b"PK\x03\x04"):
        return


def clean_extracted_text(text: str) -> str:
    """Normalize OCR/PDF noise and whitespace after extraction."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x0c", "\n")  # form feed from PDFs
    # Collapse runs of spaces/tabs (keep newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Fix common OCR confusions inside words
    text = re.sub(r"\b0(?=[a-z])", "o", text, flags=re.IGNORECASE)
    text = re.sub(r"\bl(?=[0-9])", "1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ocr_image(image, config="") -> str:
    """Run Tesseract OCR and report clear installation errors."""
    try:
        import pytesseract

        # Configure the common Windows installation path.
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        os.environ["TESSDATA_PREFIX"] = (
            r"C:\Program Files\Tesseract-OCR\tessdata"
        )

        if os.path.isfile(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        if config:
            return pytesseract.image_to_string(
                image,
                lang="eng",
                config=config,
            )

        return pytesseract.image_to_string(image, lang="eng")

    except ImportError as exc:
        raise ValueError(
            "pytesseract is not installed. Run: pip install pytesseract"
        ) from exc

    except Exception as exc:
        if exc.__class__.__name__ == "TesseractNotFoundError":
            raise ValueError(
                "Tesseract OCR engine was not found. Install Tesseract OCR "
                "and verify its installation path."
            ) from exc

        error_message = str(exc)
        lowered_error = error_message.lower()
        if (
            "error opening data file" in lowered_error
            or "failed loading language" in lowered_error
            or "could not initialize tesseract" in lowered_error
        ):
            raise ValueError(
                "Tesseract could not load English language data "
                "(eng.traineddata). Install eng.traineddata in the Tesseract "
                "tessdata directory (for example, "
                r"C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata"
                "), or set TESSDATA_PREFIX to that tessdata directory."
            ) from exc

        raise ValueError(
            f"OCR failed: {error_message}"
        ) from exc

# ── Dispatcher ────────────────────────────────────────────────────────────────

def extract_text(filepath: str) -> str:
    """
    Route file to the correct extractor based on extension.
    Returns extracted text as a single string.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if not ext:
        raise ValueError("File has no extension; cannot determine format.")

    validate_file_magic(filepath, ext)

    extractors = {
        ".pdf": _extract_pdf,
        ".png": _extract_image,
        ".jpg": _extract_image,
        ".jpeg": _extract_image,
        ".docx": _extract_docx,
        ".txt": _extract_txt,
        ".xlsx": _extract_excel,
        ".xls": _extract_excel,
    }

    extractor = extractors.get(ext)
    if not extractor:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Allowed: PDF, PNG, JPG, DOCX, TXT, XLSX, XLS."
        )

    try:
        raw = extractor(filepath)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to extract text from {ext} file: {exc}") from exc

    cleaned = clean_extracted_text(raw)
    if not cleaned:
        raise ValueError(
            f"No readable text found in the uploaded {ext} file. "
            "Try a clearer scan, a text-based PDF, or DOCX/TXT."
        )
    return cleaned


# ── PDF Extraction ────────────────────────────────────────────────────────────

def _extract_pdf(filepath: str) -> str:
    """
    Extract text from PDF using pdfplumber.
    Falls back to Tesseract OCR if a page has no selectable text
    (i.e., scanned/image-based PDF pages).
    """
    text_parts = []

    try:
        pdf_ctx = pdfplumber.open(filepath)
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    with pdf_ctx as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()

            if page_text and page_text.strip():
                text_parts.append(page_text)
            else:
                try:
                    page_image = page.to_image(resolution=300).original
                    ocr_text = _ocr_image(page_image)
                    if ocr_text.strip():
                        text_parts.append(ocr_text)
                except Exception as exc:
                    raise ValueError(
                        f"Could not OCR PDF page {page_number}: {exc}"
                    ) from exc

    if not text_parts:
        raise ValueError(
            "PDF has no selectable text and OCR did not return any content."
        )

    return "\n".join(text_parts)


# ── Image Extraction ──────────────────────────────────────────────────────────

def _extract_image(filepath: str) -> str:
    """Preprocess an uploaded image and try multiple OCR configurations."""
    try:
        from PIL import Image, ImageChops, ImageOps, ImageFilter
    except ImportError as exc:
        raise ValueError(
            "Pillow is not installed. Run: pip install Pillow"
        ) from exc

    try:
        with Image.open(filepath) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not open uploaded image: {exc}") from exc

    scale = max(2, min(4, 1800 // max(1, image.width)))
    image = image.resize(
        (image.width * scale, image.height * scale),
        Image.Resampling.LANCZOS,
    )

    gray = ImageOps.autocontrast(image.convert("L"))
    thresholded = gray.point(lambda pixel: 0 if pixel < 155 else 255)
    red, _, blue = image.split()
    blue_ink = ImageChops.subtract(blue, red).point(
        lambda difference: 255 - min(255, difference * 4)
    )
    candidates = [
        (gray, "--psm 6"),
        (gray, "--psm 11"),
        (thresholded, "--psm 6"),
        (thresholded.filter(ImageFilter.MedianFilter(size=3)), "--psm 11"),
        (blue_ink, "--psm 6"),
    ]

    results = []
    errors = []

    for candidate, config in candidates:
        try:
            result = _ocr_image(candidate, config=config).strip()
            if result:
                results.append(result)
        except ValueError as exc:
            errors.append(str(exc))

    if not results:
        detail = errors[0] if errors else "No text detected."
        raise ValueError(f"OCR could not read the image. {detail}")

    def readability_score(text):
        words = re.findall(r"[A-Za-z]{2,}", text)
        letters = sum(character.isalpha() for character in text)
        noise = sum(
            not (
                character.isalnum()
                or character.isspace()
                or character in ".,:;?!'-()"
            )
            for character in text
        )
        return len(words) * 3 + letters - noise * 2

    best_text = max(results, key=readability_score)

    try:
        header = gray.crop(
            (
                int(gray.width * 0.03),
                int(gray.height * 0.05),
                max(1, int(gray.width * 0.40)),
                max(1, int(gray.height * 0.28)),
            )
        )
        header_text = _ocr_image(header, config="--psm 11")
        label_match = re.search(
            r"(?im)^\s*(Q\s*\.?\s*\d{1,3}"
            r"(?:\s*\([a-z0-9]+\))?[.):]?)",
            header_text,
        )
        if label_match and not re.search(
            r"(?im)^\s*Q\s*\.?\s*\d{1,3}",
            best_text,
        ):
            best_text = f"{label_match.group(1)}\n{best_text}"
    except ValueError:
        pass

    return clean_extracted_text(best_text)


# ── DOCX Extraction ───────────────────────────────────────────────────────────

def _extract_docx(filepath: str) -> str:
    """
    Extract text from a .docx Word document paragraph by paragraph.
    Table cells are also captured.
    """
    try:
        doc = docx.Document(filepath)
    except Exception as exc:
        raise ValueError(f"Could not open DOCX file: {exc}") from exc

    parts = []

    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)

    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                parts.append(row_text)

    if not parts:
        raise ValueError("DOCX file contains no text.")

    return "\n".join(parts)


# ── TXT Extraction ────────────────────────────────────────────────────────────

def _extract_txt(filepath: str) -> str:
    """Read a plain text file with UTF-8 encoding (fallback to latin-1)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            content = f.read()
    if not content.strip():
        raise ValueError("Text file is empty.")
    return content


# ── Excel Extraction ──────────────────────────────────────────────────────────

def _extract_excel(filepath: str) -> str:
    """
    Read all sheets from an Excel file and concatenate all cell values.
    Useful when answer sheets are stored in tabular format.
    """
    try:
        xls = pd.ExcelFile(filepath)
    except Exception as exc:
        raise ValueError(f"Could not open Excel file: {exc}") from exc

    all_text = []

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name, header=None)
        for _, row in df.iterrows():
            row_text = " ".join(str(v) for v in row if pd.notna(v) and str(v).strip())
            if row_text:
                all_text.append(row_text)

    if not all_text:
        raise ValueError("Excel file contains no cell data.")

    return "\n".join(all_text)
