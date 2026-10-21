
import re


QUESTION_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:"
    r"(?:question|ques|q)[ \t]*\.?[ \t]*(?:no\.?[ \t]*)?"
    r"(?P<word_number>\d{1,3})(?P<word_part>\([a-z0-9]+\))?"
    r"[ \t]*[.):\-]?"
    r"|(?P<plain_part_number>\d{1,3})"
    r"(?P<plain_part>\([a-z0-9]+\))[ \t]*[.):\-]?"
    r"|\((?P<wrapped_number>\d{1,3})\)[ \t]*[.):\-]?"
    r"|(?P<plain_number>\d{1,3})[ \t]*[.):\-]"
    r"|(?P<plain_space_number>(?:[1-9]\d?|100))[ \t]+(?=[A-Za-z])"
    r"|(?P<bare_number>\d{1,3})[ \t]*$"
    r")[ \t]*"
)

ANSWER_LABEL_PATTERN = re.compile(
    r"(?i)\b(?:answer|ans)\s*[:.)\-]\s*"
)

MARK_ANNOTATION_PATTERN = re.compile(
    r"[\[(]\s*\d+(?:\.\d+)?\s*marks?\s*[\])]",
    re.IGNORECASE,
)

PROMPT_PREFIX_PATTERN = re.compile(
    r"(?is)^\s*(?:(?:please\s+)?"
    r"(?:explain|define|describe|discuss|state|list|name|identify|write|"
    r"what\s+is|who\s+is|how\s+does|why\s+does)\b"
    r"[^.!?\n]*[.!?]\s*)+"
)

FOOTER_PATTERNS = [
    r"^\s*Page\s+\d+(?:\s+of\s+\d+)?\s*$",
    r"^\s*Sample Answer Sheet Evaluator Test Document\s*$",
]

MODEL_ANSWER_PATTERN = re.compile(
    r"(?im)^\s*(?:model|sample|suggested)\s+answer\s*:\s*"
)

REFERENCE_SECTION_PATTERN = re.compile(
    r"(?i)(?:"
    r"key\s+points?\s+and\s+evaluation\s+scheme|"
    r"evaluation\s+scheme|marking\s+scheme|"
    r"guidelines?\s+for\s+partial\s+credit|"
    r"partial[- ]credit\s+guidelines?|rubric"
    r")\s*:?"
)

REFERENCE_QUESTION_PATTERN = re.compile(
    r"(?i)^\s*(?:question|ques|q)\.?\s*(?:no\.?\s*)?"
    r"\d{1,3}(?:\([a-z0-9]+\))?\s*[.):\-]?"
)


def _clean_text(text):
    """Normalize line endings and remove known document footers."""
    lines = (
        str(text or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\x0c", "\n")
        .split("\n")
    )

    cleaned = []

    for line in lines:
        line = line.strip()

        if not line:
            cleaned.append("")
            continue

        if any(
            re.match(pattern, line, flags=re.IGNORECASE)
            for pattern in FOOTER_PATTERNS
        ):
            continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def segment_questions(text):
    """
    Split student text into numbered answers.

    Supported examples:
        Q1. Answer text
        Q1 Answer text
        Q1) Answer text
        Q.1 Answer text
        Q. 1: Answer text
        Question 1: Answer text
        Question No. 1: Answer text
        Q2(a). Subpart answer
        (1) Answer text
        1 Answer text
        1. Answer text
        2) Answer text
        1
        Answer text

    A question label can appear on its own line.
    Returns a dictionary such as:
        {"Q1": "Answer text", "Q2": "Answer text"}
    """
    text = _clean_text(text)

    if not text:
        return {}

    lines = text.splitlines()
    questions = {}
    current_label = None
    current_lines = []

    def save_current():
        nonlocal current_label, current_lines

        if current_label is None:
            return

        content = "\n".join(current_lines).strip()

        if content:
            # Remove an explicit Answer: or Ans: label.
            answer_label = ANSWER_LABEL_PATTERN.search(content)

            if answer_label:
                content = content[answer_label.end():].strip()

            # Remove annotations such as [5 Marks].
            content = MARK_ANNOTATION_PATTERN.sub(" ", content)

            # Remove question instructions repeated before the answer.
            content = PROMPT_PREFIX_PATTERN.sub("", content).strip()

            # Normalize whitespace without destroying paragraph boundaries.
            content = re.sub(r"[ \t]+", " ", content)
            content = re.sub(r"\n{3,}", "\n\n", content).strip()

            if content:
                if current_label in questions:
                    questions[current_label] += "\n" + content
                else:
                    questions[current_label] = content

    for line in lines:
        match = QUESTION_PATTERN.match(line)

        if match:
            save_current()

            number_text = (
                match.group("word_number")
                or match.group("plain_part_number")
                or match.group("wrapped_number")
                or match.group("plain_number")
                or match.group("plain_space_number")
                or match.group("bare_number")
            )

            subpart = (
                match.group("word_part")
                or match.group("plain_part")
                or ""
            )

            current_label = f"Q{int(number_text)}{subpart.lower()}"

            # Keep any text appearing after the question label.
            remainder = line[match.end():].strip()
            current_lines = [remainder] if remainder else []

        elif current_label is not None:
            current_lines.append(line)

    save_current()

    return questions


def segment_reference_answers(text):
    """Return numbered reference answers without marking-scheme sections."""
    lines = []
    in_rubric = False

    for line in str(text or "").splitlines():
        if in_rubric:
            if REFERENCE_QUESTION_PATTERN.match(line):
                in_rubric = False
            else:
                continue

        section = REFERENCE_SECTION_PATTERN.search(line)
        if section:
            answer_prefix = line[:section.start()].rstrip()
            if answer_prefix:
                lines.append(answer_prefix)
            in_rubric = True
        else:
            lines.append(line)

    questions = segment_questions("\n".join(lines))
    reference_answers = {}

    for label, content in questions.items():
        model_answer = MODEL_ANSWER_PATTERN.search(content)
        if model_answer:
            content = content[model_answer.end():]

        section = REFERENCE_SECTION_PATTERN.search(content)
        if section:
            content = content[:section.start()]

        content = content.strip(" \t\r\n:;")
        if content:
            reference_answers[label] = content

    return reference_answers
