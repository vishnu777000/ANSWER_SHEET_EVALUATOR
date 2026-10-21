
import re
import math

from modules.local_rubric import evaluate_answer


def _normalize_questions(data):
    if isinstance(data, dict):
        return [(str(label), content) for label, content in data.items()]

    if isinstance(data, (list, tuple)):
        output = []

        for index, item in enumerate(data, start=1):
            if isinstance(item, dict):
                label = (
                    item.get("label")
                    or item.get("question_number")
                    or item.get("question_no")
                    or item.get("number")
                    or item.get("question_id")
                    or f"Q{index}"
                )
                output.append((str(label), item))
            else:
                output.append((f"Q{index}", item))

        return output

    return []


def _extract_text(item, prefer_reference=False):
    if item is None:
        return ""

    if isinstance(item, str):
        return item.strip()

    if isinstance(item, (list, tuple)):
        return " ".join(
            value for value in (
                _extract_text(x, prefer_reference)
                for x in item
            ) if value
        )

    if isinstance(item, dict):
        if prefer_reference:
            keys = (
                "reference_answer", "answer_key", "key_answer",
                "model_answer", "answer", "text", "content",
                "body", "question_text", "value"
            )
        else:
            keys = (
                "student_answer", "answer", "text", "content",
                "body", "question_text", "value"
            )

        for key in keys:
            value = item.get(key)
            if value is not None:
                extracted = _extract_text(value, prefer_reference)
                if extracted:
                    return extracted

        return ""

    return str(item).strip()


def _question_number(label):
    match = re.search(
        r"(?i)(?:question|ques|q)?\s*(\d+)",
        str(label).strip()
    )
    return int(match.group(1)) if match else None


def _question_label(label):
    normalized = re.sub(r"\s+", "", str(label).strip().casefold())
    match = re.fullmatch(
        r"(?:question|ques|q)?\.?(\d{1,3})(\([a-z0-9]+\))?",
        normalized,
    )
    if match:
        return f"q{int(match.group(1))}{match.group(2) or ''}"
    return normalized


def _safe_number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def evaluate_answers(
    key_questions=None,
    ans_questions=None,
    max_marks=10,
    use_semantic=False,
    student_answers=None,
    reference_answers=None,
    **kwargs
):
    """
    Offline grading function compatible with app.py.

    similarity mirrors concept coverage as a fraction from 0 to 1 for
    compatibility; semantic similarity is informational and does not affect marks.
    """

    if reference_answers is not None:
        key_questions = reference_answers

    if student_answers is not None:
        ans_questions = student_answers

    if key_questions is None:
        key_questions = kwargs.get("reference_questions", [])

    if ans_questions is None:
        ans_questions = kwargs.get("student_questions", [])

    per_question_max = _safe_number(max_marks, -1)

    if per_question_max < 0:
        raise ValueError("max_marks must be a non-negative number.")

    keys = _normalize_questions(key_questions)
    answers = _normalize_questions(ans_questions)

    if not keys:
        raise ValueError(
            "No questions detected in answer key. "
            "Check the reference PDF text extraction and question numbering."
        )

    answers_by_number = {}
    answers_by_label = {}

    for label, content in answers:
        number = _question_number(label)

        if number is not None:
            answers_by_number.setdefault(number, []).append(content)

        answers_by_label[_question_label(label)] = content

    results = []
    total_marks = 0.0
    max_total = 0.0

    for index, (key_label, key_content) in enumerate(keys, start=1):
        number = _question_number(key_label)
        if number is None:
            number = index

        reference_text = _extract_text(key_content, prefer_reference=True)

        student_content = answers_by_label.get(_question_label(key_label))
        if student_content is None:
            number_answers = answers_by_number.get(number, [])
            if len(number_answers) == 1:
                student_content = number_answers[0]

        student_text = _extract_text(student_content)

        grading = evaluate_answer(
            student_text,
            reference_text,
            per_question_max,
            use_semantic=use_semantic,
        )

        marks = _safe_number(grading.get("marks"), 0.0)
        marks = max(0.0, min(per_question_max, marks))

        coverage = _safe_number(grading.get("coverage"), 0.0)
        coverage = max(0.0, min(1.0, coverage))

        # Keep this as 0–1 because the existing template multiplies by 100.
        similarity = _safe_number(grading.get("similarity"), coverage)
        similarity = max(0.0, min(1.0, similarity))

        total_marks += marks
        max_total += per_question_max

        feedback = grading.get("feedback") or "Review this answer manually."

        results.append({
            "label": key_label,
            "question_number": str(number),
            "question": key_label,

            "key_answer": reference_text,
            "key_answer_text": reference_text,
            "answer_key": reference_text,
            "reference_answer": reference_text,
            "model_answer": reference_text,

            "student_answer": student_text,
            "answer": student_text,

            "marks": round(marks, 2),
            "score": round(marks, 2),
            "max_marks": round(per_question_max, 2),

            "coverage": round(coverage, 4),
            "similarity": round(similarity, 4),
            "semantic_similarity": grading.get("semantic_similarity"),
            "similarity_score": round(similarity, 4),
            "overall_similarity": round(similarity, 4),

            "covered_concepts": grading.get("covered_concepts", []),
            "missing_concepts": grading.get("missing_concepts", []),
            "incorrect_concepts": grading.get("incorrect_concepts", []),
            "scoring_method": grading.get("scoring_method", "concept-keyword"),
            "feedback": feedback,
        })

    percentage = (
        round(total_marks / max_total * 100, 2)
        if max_total > 0 else 0.0
    )

    return {
        "questions": results,
        "results": results,
        "total_marks": round(total_marks, 2),
        "max_total": round(max_total, 2),
        "max_marks": round(max_total, 2),
        "percentage": percentage,
    }
