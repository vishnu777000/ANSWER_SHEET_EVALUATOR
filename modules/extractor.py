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


def _ocr_image(image) -> str:
    """Run Tesseract with graceful fallback if OCR is unavailable."""
    try:
        import pytesseract
        return pytesseract.image_to_string(image, lang="eng")
    except ImportError:
        raise ValueError(
            "pytesseract is not installed. Install it or upload TXT/DOCX/text-based PDF."
        )
    except Exception as exc:
        if exc.__class__.__name__ == "TesseractNotFoundError":
            raise ValueError(
                "Tesseract OCR is not installed or not on PATH. "
                "Install Tesseract or upload a text-selectable PDF/DOCX/TXT file."
            ) from exc
        raise ValueError(
            "Tesseract OCR is not installed or not on PATH. "
            "Install Tesseract or upload a text-selectable PDF/DOCX/TXT file."
        )

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
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text and page_text.strip():
                text_parts.append(page_text)
            else:
                try:
                    page_image = page.to_image(resolution=300).original
                    ocr_text = _ocr_image(page_image)
                    if ocr_text.strip():
                        text_parts.append(ocr_text)
                except Exception:
                    continue

    if not text_parts:
        raise ValueError(
            "PDF has no selectable text and OCR did not return any content."
        )

    return "\n".join(text_parts)


# ── Image Extraction ──────────────────────────────────────────────────────────

def _extract_image(filepath: str) -> str:
    """
    Run Tesseract OCR on an image file.
    Preprocessing (grayscale) improves accuracy on low-contrast images.
    """
    try:
        from PIL import Image
        image = Image.open(filepath).convert("L")
    except Exception as exc:
        raise ValueError(f"Could not open image: {exc}") from exc

    text = _ocr_image(image)
    if not text.strip():
        raise ValueError("OCR returned no text from the image.")
    return text


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
