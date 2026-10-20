"""
modules/parser.py — Question Segmentation Module
=================================================
Detects and splits structured answer sheets into individual question blocks.

Supported patterns (regex-based):
  Q1, Q2, Q.1, Q.2         → Numbered questions
  1., 2., 3.                → Plain numbered list
  Part A, Part B, Section A → Section headers
  1(a), 1(b), (a), (b)     → Sub-questions
  Q1:, Q2:                  → Questions with colons
  Question 1, Question 2    → Full word format

Returns an ordered dict: { "Q1": "answer text", "Q2": "...", ... }
"""

import re
from collections import OrderedDict


# ── Answer Prefix Pattern ─────────────────────────────────────────────────────
ANSWER_PREFIX = re.compile(
    r'^\s*(?:answer|ans(?:wer)?|a)\s*[:.\-]\s*',
    re.IGNORECASE | re.MULTILINE,
)

# Question header lines to strip (lines starting with interrogative words ending with ?)
QUESTION_LINE = re.compile(
    r'^\s*(?:what|define|explain|describe|list|state|name|write|how|why|when|where|which|who)\b.*\?\s*$',
    re.IGNORECASE | re.MULTILINE,
)


# ── Master Pattern ────────────────────────────────────────────────────────────
# Matches all supported question/section header formats.
# Order matters — more specific patterns first.

QUESTION_PATTERNS = [
    # "Part A", "Section B", "Unit III"
    r"(part\s+[a-z0-9]+|section\s+[a-z0-9]+|unit\s+[a-z0-9]+)\s*[:\-]?",
    # "Q1:", "Q.1:", "Q1 )", "Q1."
    r"(q\.?\s*\d+\s*[\):\.\-]?)",
    # "Question 1", "Question No. 2"
    r"(question\s+(?:no\.?\s*)?\d+\s*[\):\.\-]?)",
    # "1.", "2.", "10." — plain numbered
    r"(\b\d{1,2}[\.\)]\s)",
    # "1(a)", "1(b)", "2(i)"
    r"(\b\d+\s*\([a-z]+\)\s*[\.\-]?)",
    # "(a)", "(b)", "(i)", "(ii)"
    r"(\([a-z]+\)\s*[\.\-]?)",
]

# Combined pattern: any of the above at the START of a line
# re.IGNORECASE handles case-insensitivity globally (no inline (?i) needed)
COMBINED_PATTERN = re.compile(
    r"^\s*(?:" + "|".join(QUESTION_PATTERNS) + r")",
    re.MULTILINE | re.IGNORECASE,
)


# ── Main Segmentation Function ────────────────────────────────────────────────

def segment_questions(text: str) -> OrderedDict:
    """
    Split text into question segments.

    Steps:
      1. Normalize whitespace and line endings.
      2. Find all question header positions using regex.
      3. Extract the text between consecutive headers.
      4. Return as an OrderedDict for stable ordering.

    If no patterns are detected (free-form text), the entire text
    is returned as a single entry "Q1".
    """
    text = _normalize(text)
    matches = list(COMBINED_PATTERN.finditer(text))

    # ── No structure detected → treat whole text as one answer ───────────────
    if not matches:
        return OrderedDict([("Q1", text.strip())])

    segments = OrderedDict()
    for i, match in enumerate(matches):
        # Label: strip and normalize the matched header
        label = _normalize_label(match.group(0))

        # Content: text from end of this header to start of next
        start = match.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if content:  # Skip empty segments
            segments[label] = _clean_answer_text(content)

    return segments


# ── Answer Text Cleaner ───────────────────────────────────────────────────────

def _clean_answer_text(text: str) -> str:
    """
    Clean a parsed segment by removing question header lines and answer prefixes.
    E.g., 'What is AI?\\nAnswer: AI is...' → 'AI is...'
    """
    # Remove question header lines (e.g., "What is AI?")
    text = QUESTION_LINE.sub('', text).strip()
    # Strip "Answer:", "Ans:", "A:" prefixes on each line
    while ANSWER_PREFIX.search(text):
        text = ANSWER_PREFIX.sub('', text).strip()
    # Inline "Answer: ..." on the same line as leftover question text
    text = re.sub(
        r'(?:^|\n)\s*(?:answer|ans(?:wer)?|a)\s*[:.\-]\s*',
        ' ',
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text


# ── Label Normalizer ──────────────────────────────────────────────────────────

def _normalize_label(raw: str) -> str:
    """
    Normalize a raw matched label into a clean key.
    Examples:
      "q1:"  → "Q1"
      "Q.2." → "Q2"
      "part a" → "PART_A"
      "(b)"   → "(b)"
    """
    label = raw.strip().rstrip(":.)-").strip()
    label = re.sub(r"\s+", "_", label)  # Replace spaces with underscore
    label = re.sub(r"[^\w\(\)]", "", label)  # Remove special chars except brackets
    return label.upper()


# ── Text Normalizer ───────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """
    Normalize line endings and excessive whitespace.
    Preserves paragraph structure for accurate segmentation.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ── Utility: Match Questions Between Key and Student ─────────────────────────

def match_questions(key_questions: OrderedDict, ans_questions: OrderedDict) -> list:
    """
    Align student answers to answer key questions.

    Strategy:
      1. Exact label match (e.g. Q1 → Q1)
      2. Fuzzy label match (e.g. QUESTION_1 ~ Q1) via normalized comparison
      3. Positional fallback (match by order if labels don't align)

    Returns a list of tuples:
      (label, key_answer, student_answer_or_None)
    """
    matched = []
    used_ans_labels = set()

    key_labels = list(key_questions.keys())
    ans_labels  = list(ans_questions.keys())

    for key_label in key_labels:
        student_answer = None

        # ── Strategy 1: Exact match ───────────────────────────────────────────
        if key_label in ans_questions:
            student_answer = ans_questions[key_label]
            used_ans_labels.add(key_label)

        else:
            # ── Strategy 2: Fuzzy match ───────────────────────────────────────
            key_normalized = _strip_to_digits(key_label)
            for ans_label in ans_labels:
                if ans_label in used_ans_labels:
                    continue
                if _strip_to_digits(ans_label) == key_normalized:
                    student_answer = ans_questions[ans_label]
                    used_ans_labels.add(ans_label)
                    break

        # ── Strategy 3: Positional fallback ──────────────────────────────────
        if student_answer is None:
            key_idx = key_labels.index(key_label)
            if key_idx < len(ans_labels):
                fallback_label = ans_labels[key_idx]
                if fallback_label not in used_ans_labels:
                    student_answer = ans_questions[fallback_label]
                    used_ans_labels.add(fallback_label)

        matched.append((key_label, key_questions[key_label], student_answer))

    return matched


def _strip_to_digits(label: str) -> str:
    """Extract only digits from a label for fuzzy comparison. '(B)' → '' , 'Q2' → '2'"""
    return re.sub(r"\D", "", label)
