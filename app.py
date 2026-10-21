"""
app.py — Main Flask Application Entry Point
============================================
Handles routing, file uploads, evaluation orchestration,
and Excel result storage.
"""

import os
import uuid
import re
import json
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from modules.extractor import extract_text, validate_file_magic
from modules.parser import segment_questions, segment_reference_answers
from modules.evaluator import evaluate_answers
from modules.storage import save_to_excel, save_batch_to_excel, get_subjects


# ── App Configuration ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULTS_FILE"] = "results/results.xlsx"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB limit

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "docx", "txt", "xlsx", "xls"}


def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file):
    """Save an uploaded file and return its path."""
    filename = secure_filename(file.filename or "")
    if not filename:
        raise ValueError("The uploaded file must have a valid filename.")

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    try:
        file.save(path)
        ext = os.path.splitext(filename)[1].lower()
        if ext:
            validate_file_magic(path, ext)
        return path
    except Exception:
        _remove_upload(path)
        raise


def _remove_upload(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            app.logger.warning("Failed to remove a temporary upload.")


def _parse_max_marks(value):
    try:
        marks = int(value)
    except (TypeError, ValueError):
        raise ValueError("Maximum marks must be a whole number between 1 and 100.")

    if not 1 <= marks <= 100:
        raise ValueError("Maximum marks must be between 1 and 100.")

    return marks


def _validate_subject(subject):
    subject = (subject or "").strip()
    if not subject:
        raise ValueError("Please enter a subject name.")
    if (
        len(subject) > 31
        or any(char in subject for char in "[]:*?/\\")
        or subject.startswith("'")
        or subject.endswith("'")
    ):
        raise ValueError(
            "Subject must be 1–31 characters and cannot contain: []:*?/\\ "
            "or begin/end with an apostrophe."
        )
    return subject


@app.errorhandler(RequestEntityTooLarge)
def handle_upload_too_large(error):
    app.logger.info("Rejected upload exceeding configured request size: %s", error)
    return jsonify({
        "success": False,
        "errors": ["Upload exceeds the configured request size limit."],
    }), 413


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the main upload page."""
    subjects = get_subjects(app.config["RESULTS_FILE"])
    return render_template("index.html", subjects=subjects)


@app.route("/extract_for_review", methods=["POST"])
def extract_for_review():
    """Extract uploaded answer text for review before evaluation."""
    answer_key_file = request.files.get("answer_key")
    if not answer_key_file or not answer_key_file.filename:
        return jsonify({"success": False, "errors": ["Answer key file is required."]}), 400
    if not allowed_file(answer_key_file.filename):
        return jsonify({"success": False, "errors": ["Unsupported answer key file format."]}), 400

    batch_mode = request.form.get("mode") == "batch"
    student_files = (
        request.files.getlist("student_answers")
        if batch_mode
        else [request.files.get("student_answer")]
    )
    student_files = [file for file in student_files if file and file.filename]
    if not student_files:
        return jsonify({
            "success": False,
            "errors": ["At least one student answer sheet file is required."],
        }), 400

    invalid_files = [
        file.filename for file in student_files
        if not allowed_file(file.filename)
    ]
    if invalid_files:
        return jsonify({
            "success": False,
            "errors": [
                "Unsupported student file format: " + ", ".join(invalid_files)
            ],
        }), 400

    saved_paths = []
    try:
        key_path = save_upload(answer_key_file)
        saved_paths.append(key_path)
        key_text = extract_text(key_path)

        student_texts = []
        for student_file in student_files:
            student_path = save_upload(student_file)
            saved_paths.append(student_path)
            student_texts.append({
                "filename": student_file.filename,
                "text": extract_text(student_path),
            })

        return jsonify({
            "success": True,
            "answer_key_text": key_text,
            "student_answers": student_texts,
        })
    except ValueError as exc:
        return jsonify({"success": False, "errors": [str(exc)]}), 400
    except Exception:
        app.logger.exception("Text extraction for review failed.")
        return jsonify({
            "success": False,
            "errors": ["Could not extract text for review due to a server error."],
        }), 500
    finally:
        for path in saved_paths:
            _remove_upload(path)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Handle evaluation request.
    Expects multipart form with:
      - answer_key: file
      - student_answer: file
      - student_name: string
      - reg_number: string
      - subject: string
      - max_marks: int (marks per question)
      - use_semantic: bool (use sentence transformers vs TF-IDF)
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    reviewed_text = (
        "reviewed_key_text" in request.form
        and "reviewed_student_text" in request.form
    )
    answer_key_file = request.files.get("answer_key")
    student_answer_file = request.files.get("student_answer")

    if reviewed_text:
        answer_key_file = None
        student_answer_file = None
    else:
        errors = []
        if not answer_key_file or not answer_key_file.filename:
            errors.append("Answer key file is required.")
        if not student_answer_file or not student_answer_file.filename:
            errors.append("Student answer file is required.")
        if errors:
            return jsonify({"success": False, "errors": errors}), 400
        if (
            not allowed_file(answer_key_file.filename)
            or not allowed_file(student_answer_file.filename)
        ):
            return jsonify({"success": False, "errors": ["Unsupported file format."]}), 400

    # ── Metadata ──────────────────────────────────────────────────────────────
    student_name = request.form.get("student_name", "Unknown").strip() or "Unknown"
    reg_number   = request.form.get("reg_number", "N/A").strip() or "N/A"
    try:
        subject = _validate_subject(request.form.get("subject", "General"))
        max_marks = _parse_max_marks(request.form.get("max_marks", "10"))
    except ValueError as exc:
        return jsonify({"success": False, "errors": [str(exc)]}), 400
    use_semantic = request.form.get("use_semantic", "false").lower() == "true"

    key_path = None
    ans_path = None

    try:
        # ── Step 1: Extract text ──────────────────────────────────────────────
        if reviewed_text:
            key_text = request.form["reviewed_key_text"]
            ans_text = request.form["reviewed_student_text"]
        else:
            key_path = save_upload(answer_key_file)
            ans_path = save_upload(student_answer_file)
            key_text = extract_text(key_path)
            ans_text = extract_text(ans_path)

        if not key_text.strip():
            return jsonify({"success": False, "errors": ["Could not extract text from answer key."]}), 400
        if not ans_text.strip():
            return jsonify({"success": False, "errors": ["Could not extract text from student answer."]}), 400

        # ── Step 2: Segment into questions ────────────────────────────────────
        key_questions = segment_reference_answers(key_text)
        ans_questions = segment_questions(ans_text)

        if not key_questions:
            return jsonify({"success": False, "errors": ["No questions detected in answer key."]}), 400
        if not ans_questions:
            return jsonify({
                "success": False,
                "errors": [
                    "No numbered student answers detected. Add question labels such as Q1. or 1."
                ],
            }), 400

        # ── Step 3: Evaluate each question ────────────────────────────────────
        results = evaluate_answers(
            key_questions=key_questions,
            ans_questions=ans_questions,
            max_marks=max_marks,
            use_semantic=use_semantic,
        )

        # ── Step 4: Store results in Excel ────────────────────────────────────
        save_to_excel(
            filepath=app.config["RESULTS_FILE"],
            subject=subject,
            student_name=student_name,
            reg_number=reg_number,
            results=results,
        )

        return jsonify({"success": True, "results": results})

    except ValueError as e:
        return jsonify({"success": False, "errors": [str(e)]}), 400
    except Exception:
        app.logger.exception("Single-answer evaluation failed.")
        return jsonify({
            "success": False,
            "errors": ["Evaluation failed due to a server error. Please try again."],
        }), 500

    finally:
        for path in [key_path, ans_path]:
            _remove_upload(path)


@app.route("/preview", methods=["POST"])
def preview():
    """
    Extract and segment text from uploaded files without scoring.
    Useful for verifying OCR/parsing before running full evaluation.
    """
    role = request.form.get("role", "student")
    file_key = "answer_key" if role == "key" else "student_answer"
    upload = request.files.get(file_key) or request.files.get("file")

    if not upload or upload.filename == "":
        return jsonify({"success": False, "errors": ["Upload a file to preview."]}), 400
    if not allowed_file(upload.filename):
        return jsonify({"success": False, "errors": ["Unsupported file format."]}), 400

    path = None
    try:
        path = save_upload(upload)
        text = extract_text(path)
        questions = (
            segment_reference_answers(text)
            if role == "key"
            else segment_questions(text)
        )
        preview_questions = [
            {"label": label, "text": (body[:500] + "…") if len(body) > 500 else body}
            for label, body in questions.items()
        ]
        return jsonify({
            "success": True,
            "role": role,
            "filename": upload.filename,
            "char_count": len(text),
            "question_count": len(questions),
            "text_preview": text[:2000] + ("…" if len(text) > 2000 else ""),
            "questions": preview_questions,
        })
    except ValueError as e:
        return jsonify({"success": False, "errors": [str(e)]}), 400
    except Exception as e:
        app.logger.exception("File preview failed.")
        return jsonify({
            "success": False,
            "errors": ["Preview failed due to a server error. Please try again."],
        }), 500
    finally:
        _remove_upload(path)


def _parse_student_info(filename: str):
    """
    Extract student name and registration number from filename.
    Examples:
      '21CS001_Priya_Ramesh.pdf' -> ('Priya Ramesh', '21CS001')
      'John_Doe_102.txt'         -> ('John Doe', '102')
      'student-answer1.txt'      -> ('Student-Answer1', 'N/A')
    """
    raw_name = os.path.splitext(filename)[0]
    cleaned = re.sub(r"^[0-9a-fA-F]{32}_", "", raw_name)
    parts = [p.strip() for p in re.split(r"[_\-]+", cleaned) if p.strip()]

    if not parts:
        return ("Unknown", "N/A")

    if any(c.isdigit() for c in parts[0]) and len(parts) > 1:
        reg_number = parts[0]
        student_name = " ".join(parts[1:]).title()
    elif any(c.isdigit() for c in parts[-1]) and len(parts) > 1:
        reg_number = parts[-1]
        student_name = " ".join(parts[:-1]).title()
    else:
        reg_number = "N/A"
        student_name = " ".join(parts).title()

    return student_name, reg_number


@app.route("/evaluate_batch", methods=["POST"])
def evaluate_batch():
    """
    Handle batch evaluation for multiple student files against a single answer key.
    Expects multipart form with:
      - answer_key: file
      - student_answers: multiple files
      - subject: string
      - max_marks: int
      - use_semantic: bool
    """
    reviewed_text = (
        "reviewed_key_text" in request.form
        and "reviewed_answers" in request.form
    )
    answer_key_file = request.files.get("answer_key")
    student_files = request.files.getlist("student_answers")
    student_files = [f for f in student_files if f and f.filename]
    reviewed_answers = None
    if reviewed_text:
        try:
            reviewed_answers = json.loads(request.form["reviewed_answers"])
        except json.JSONDecodeError:
            return jsonify({
                "success": False,
                "errors": ["Reviewed student answers are not valid JSON."],
            }), 400
        if (
            not isinstance(reviewed_answers, list)
            or not reviewed_answers
            or any(
                not isinstance(answer, dict)
                or not isinstance(answer.get("filename"), str)
                or not isinstance(answer.get("text"), str)
                for answer in reviewed_answers
            )
        ):
            return jsonify({
                "success": False,
                "errors": ["Reviewed student answers must include filenames and text."],
            }), 400
        if any(not allowed_file(answer["filename"]) for answer in reviewed_answers):
            return jsonify({
                "success": False,
                "errors": ["Reviewed answers contain an unsupported filename."],
            }), 400
    else:
        errors = []
        if not answer_key_file or not answer_key_file.filename:
            errors.append("Answer key file is required.")
        if not student_files:
            errors.append("At least one student answer sheet file is required.")
        if errors:
            return jsonify({"success": False, "errors": errors}), 400
        if not allowed_file(answer_key_file.filename):
            return jsonify({"success": False, "errors": ["Unsupported answer key file format."]}), 400

    try:
        subject = _validate_subject(request.form.get("subject", "General"))
        max_marks = _parse_max_marks(request.form.get("max_marks", "10"))
    except ValueError as exc:
        return jsonify({"success": False, "errors": [str(exc)]}), 400
    use_semantic = request.form.get("use_semantic", "false").lower() == "true"

    key_path = None
    saved_paths = []
    records_to_save = []
    batch_results = []

    try:
        if reviewed_text:
            key_text = request.form["reviewed_key_text"]
            student_entries = [
                (answer["filename"], answer["text"], None)
                for answer in reviewed_answers
            ]
        else:
            key_path = save_upload(answer_key_file)
            saved_paths = [key_path]
            key_text = extract_text(key_path)
            student_entries = [(file.filename, None, file) for file in student_files]

        if not key_text.strip():
            return jsonify({"success": False, "errors": ["Could not extract text from answer key."]}), 400

        key_questions = segment_reference_answers(key_text)
        if not key_questions:
            return jsonify({"success": False, "errors": ["No questions detected in answer key."]}), 400

        for filename, reviewed_student_text, sfile in student_entries:
            if not allowed_file(filename):
                student_name, reg_number = _parse_student_info(filename)
                batch_results.append({
                    "filename": filename,
                    "student_name": student_name,
                    "reg_number": reg_number,
                    "status": "Failed",
                    "error": "Unsupported file format",
                    "total_marks": 0,
                    "max_total": len(key_questions) * max_marks,
                    "percentage": 0.0,
                    "results": None,
                })
                continue

            ans_path = None
            try:
                if reviewed_student_text is not None:
                    ans_text = reviewed_student_text
                else:
                    ans_path = save_upload(sfile)
                    saved_paths.append(ans_path)
                    ans_text = extract_text(ans_path)
                if not ans_text.strip():
                    raise ValueError("No text could be extracted from file.")

                ans_questions = segment_questions(ans_text)
                if not ans_questions:
                    raise ValueError(
                        "No numbered student answers detected. Add labels such as Q1. or 1."
                    )
                results = evaluate_answers(
                    key_questions=key_questions,
                    ans_questions=ans_questions,
                    max_marks=max_marks,
                    use_semantic=use_semantic,
                )

                student_name, reg_number = _parse_student_info(filename)

                records_to_save.append({
                    "student_name": student_name,
                    "reg_number": reg_number,
                    "results": results,
                })

                batch_results.append({
                    "filename": filename,
                    "student_name": student_name,
                    "reg_number": reg_number,
                    "status": "Success",
                    "total_marks": results["total_marks"],
                    "max_total": results["max_total"],
                    "percentage": results["percentage"],
                    "results": results,
                })
            except Exception as e:
                student_name, reg_number = _parse_student_info(filename)
                if isinstance(e, ValueError):
                    error_message = str(e)
                else:
                    app.logger.exception("Batch evaluation failed for one uploaded file.")
                    error_message = "Could not process this file due to a server error."
                batch_results.append({
                    "filename": filename,
                    "student_name": student_name,
                    "reg_number": reg_number,
                    "status": "Failed",
                    "error": error_message,
                    "total_marks": 0,
                    "max_total": len(key_questions) * max_marks,
                    "percentage": 0.0,
                    "results": None,
                })

        if records_to_save:
            save_batch_to_excel(
                filepath=app.config["RESULTS_FILE"],
                subject=subject,
                student_records=records_to_save,
            )

        return jsonify({
            "success": True,
            "total_processed": len(student_entries),
            "successful_evaluations": len(records_to_save),
            "batch_results": batch_results,
        })

    except ValueError as e:
        return jsonify({"success": False, "errors": [str(e)]}), 400
    except Exception:
        app.logger.exception("Batch evaluation failed.")
        return jsonify({
            "success": False,
            "errors": ["Batch evaluation failed due to a server error. Please try again."],
        }), 500
    finally:
        for path in saved_paths:
            _remove_upload(path)


@app.route("/subjects", methods=["GET"])

def subjects():
    """Return list of subjects (Excel sheet names) for dropdown."""
    return jsonify(get_subjects(app.config["RESULTS_FILE"]))


@app.route("/download")
def download():
    """Download the results Excel file."""
    filepath = app.config["RESULTS_FILE"]
    if not os.path.exists(filepath):
        return jsonify({"error": "No results file found."}), 404
    return send_file(filepath, as_attachment=True, download_name="evaluation_results.xlsx")


@app.route("/results")
def results_page():
    """Render the results viewer page."""
    subjects = get_subjects(app.config["RESULTS_FILE"])
    return render_template("results.html", subjects=subjects)


@app.route("/results/<subject>")
def subject_results(subject):
    """Return JSON data for a specific subject sheet."""
    from modules.storage import get_results_for_subject
    data = get_results_for_subject(app.config["RESULTS_FILE"], subject)
    return jsonify(data)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true"},
        host="127.0.0.1",
        port=5000,
    )
