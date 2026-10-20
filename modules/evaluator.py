"""
modules/evaluator.py — NLP Evaluation Module
=============================================
For each question pair (key answer vs student answer):
  1. Preprocess text (lowercase, remove punctuation, stopwords)
  2. Vectorize using TF-IDF
  3. Compute cosine similarity
  4. (Optional) Compute semantic similarity via Sentence Transformers
  5. Assign marks and generate feedback

Uses modules/parser.py's match_questions() for alignment.
"""

import re
import string
import math
from difflib import SequenceMatcher
from collections import Counter, OrderedDict

from modules.parser import match_questions

# ── NLTK Setup ────────────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    STOPWORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    STOPWORDS = {"a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
                 "of", "and", "or", "but", "with", "this", "that", "are", "was"}
    LEMMATIZER = None

# Expand common exam acronyms so paraphrases align with full key phrases (e.g. AI ↔ artificial intelligence)
_ACRONYM_EXPANSIONS = {
    r"\bai\b": "artificial intelligence",
    r"\bml\b": "machine learning",
    r"\bnlp\b": "natural language processing",
    r"\bdl\b": "deep learning",
    r"\bnn\b": "neural network",
}


# ── Similarity → Marks Mapping ────────────────────────────────────────────────
# Uses a continuous power curve: marks = max_marks × similarity^0.65
# Minimum threshold: similarity < 0.15 gets 0 marks.
# This gives partial credit more generously while requiring high similarity
# for full marks, avoiding harsh cliff-edge band boundaries.


# ── Main Evaluator ────────────────────────────────────────────────────────────

def evaluate_answers(
    key_questions: OrderedDict,
    ans_questions: OrderedDict,
    max_marks: int = 10,
    use_semantic: bool = False,
) -> dict:
    """
    Evaluate all questions and return structured results.

    Returns:
    {
      "questions": [
        {
          "label": "Q1",
          "key_answer": "...",
          "student_answer": "...",
          "similarity": 0.78,
          "marks": 7,
          "max_marks": 10,
          "feedback": "Good answer. Covers most key concepts.",
          "missing": False,
        }, ...
      ],
      "total_marks": 35,
      "max_total": 50,
      "percentage": 70.0,
    }
    """
    # Align questions using parser's matching logic
    matched = match_questions(key_questions, ans_questions)

    question_results = []
    total_earned = 0
    max_total = len(matched) * max_marks

    for label, key_ans, student_ans in matched:
        if student_ans is None:
            # ── Missing answer: penalize fully ───────────────────────────────
            result = {
                "label": label,
                "key_answer": key_ans,
                "student_answer": "",
                "similarity": 0.0,
                "marks": 0,
                "max_marks": max_marks,
                "feedback": "❌ Answer not provided. Full marks deducted.",
                "missing": True,
            }
        else:
            # ── Preprocess both answers ───────────────────────────────────────
            key_tokens    = preprocess(key_ans)
            student_tokens = preprocess(student_ans)

            # ── Compute similarity ────────────────────────────────────────────
            metrics = _lexical_metrics(key_ans, student_ans, key_tokens, student_tokens)
            if use_semantic:
                semantic = _semantic_similarity(key_ans, student_ans)
                similarity = 0.72 * semantic + 0.28 * metrics["combined"]
                metrics["semantic"] = round(semantic, 4)
            else:
                similarity = metrics["combined"]

            # ── Assign marks ──────────────────────────────────────────────────
            marks = _assign_marks(similarity, max_marks)
            feedback = _generate_feedback(similarity, key_tokens, student_tokens, key_ans)

            result = {
                "label": label,
                "key_answer": key_ans,
                "student_answer": student_ans,
                "similarity": round(similarity, 4),
                "metrics": metrics,
                "marks": marks,
                "max_marks": max_marks,
                "feedback": feedback,
                "missing": False,
            }

        total_earned += result["marks"]
        question_results.append(result)

    percentage = round((total_earned / max_total * 100), 2) if max_total > 0 else 0.0

    return {
        "questions": question_results,
        "total_marks": total_earned,
        "max_total": max_total,
        "percentage": percentage,
    }


# ── Text Preprocessing ────────────────────────────────────────────────────────

def preprocess(text: str) -> list:
    """
    Clean and tokenize text:
      1. Lowercase
      2. Remove punctuation and digits
      3. Tokenize by whitespace
      4. Remove stopwords
      5. Apply lemmatization (if NLTK available)
    Returns list of processed tokens.
    """
    text = text.lower()
    for pattern, replacement in _ACRONYM_EXPANSIONS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if LEMMATIZER:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens


# ── TF-IDF Cosine Similarity ─────────────────────────────────────────────────

def _tfidf_cosine_similarity(tokens_a: list, tokens_b: list) -> float:
    """
    Compute cosine similarity between two token lists using TF-IDF weights.

    TF-IDF formula used:
      TF(t, d)  = count(t in d) / len(d)
      IDF(t)    = log((1 + N) / (1 + df(t))) + 1   [sklearn-style smoothing]
      TF-IDF    = TF * IDF

    Cosine similarity = dot(A, B) / (|A| * |B|)
    """
    if not tokens_a or not tokens_b:
        return 0.0

    # Build vocabulary
    vocab = list(set(tokens_a) | set(tokens_b))
    N = 2  # Two documents

    # Term frequency
    tf_a = Counter(tokens_a)
    tf_b = Counter(tokens_b)

    def tf(counter, token, length):
        return counter[token] / length if length else 0

    # Document frequency per term
    def df(token):
        return (1 if token in tf_a else 0) + (1 if token in tf_b else 0)

    # IDF with smoothing
    def idf(token):
        return math.log((1 + N) / (1 + df(token))) + 1

    # TF-IDF vectors
    len_a, len_b = len(tokens_a), len(tokens_b)
    vec_a = {t: tf(tf_a, t, len_a) * idf(t) for t in vocab}
    vec_b = {t: tf(tf_b, t, len_b) * idf(t) for t in vocab}

    # Cosine similarity
    dot   = sum(vec_a[t] * vec_b[t] for t in vocab)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)

# ── Keyword Overlap (Jaccard) ─────────────────────────────────────────────────

def _keyword_overlap(tokens_a: list, tokens_b: list) -> float:
    """Compute Jaccard-like keyword overlap between two token lists."""
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _keyword_recall(key_tokens: list, student_tokens: list) -> float:
    """Fraction of key concepts present in the student answer (partial-credit friendly)."""
    if not key_tokens or not student_tokens:
        return 0.0
    key_set = set(key_tokens)
    student_set = set(student_tokens)
    if not key_set:
        return 0.0
    return len(key_set & student_set) / len(key_set)


def _sequence_similarity(text_a: str, text_b: str) -> float:
    """Character-level similarity robust to paraphrasing and reordering."""
    if not text_a or not text_b:
        return 0.0
    a = " ".join(preprocess(text_a))
    b = " ".join(preprocess(text_b))
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _lexical_metrics(text_a: str, text_b: str, tokens_a: list, tokens_b: list) -> dict:
    """Compute individual scores and a blended composite for marking."""
    tfidf = _tfidf_cosine_similarity(tokens_a, tokens_b)
    jaccard = _keyword_overlap(tokens_a, tokens_b)
    recall = _keyword_recall(tokens_a, tokens_b)
    sequence = _sequence_similarity(text_a, text_b)

    weighted = (
        0.32 * tfidf
        + 0.30 * recall
        + 0.13 * jaccard
        + 0.25 * sequence
    )
    # Reward answers that cover key terms even when TF-IDF is low (short paraphrases)
    coverage = 0.58 * recall + 0.42 * sequence
    combined = max(weighted, coverage * 0.94)

    return {
        "tfidf": round(tfidf, 4),
        "keyword_recall": round(recall, 4),
        "jaccard": round(jaccard, 4),
        "sequence": round(sequence, 4),
        "combined": round(min(1.0, combined), 4),
    }


# ── Semantic Similarity (Sentence Transformers) ───────────────────────────────

# Module-level cache for Sentence Transformer model
_ST_MODEL = None

def _get_st_model():
    """Lazy-load and cache the Sentence Transformer model."""
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL

def _semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute semantic similarity using Sentence Transformers.
    Falls back to TF-IDF if the library is unavailable.
    Model: 'all-MiniLM-L6-v2' — fast and accurate.
    """
    try:
        from sentence_transformers import util
        model = _get_st_model()
        emb_a = model.encode(text_a, convert_to_tensor=True)
        emb_b = model.encode(text_b, convert_to_tensor=True)
        score = util.pytorch_cos_sim(emb_a, emb_b).item()
        return max(0.0, float(score))
    except ImportError:
        key_tokens = preprocess(text_a)
        student_tokens = preprocess(text_b)
        return _lexical_metrics(text_a, text_b, key_tokens, student_tokens)["combined"]


# ── Mark Assignment ───────────────────────────────────────────────────────────

def _assign_marks(similarity: float, max_marks: int) -> int:
    """
    Map similarity score to marks using a continuous power curve.
    Uses: marks = max_marks × similarity^0.65
    Minimum threshold: similarity < 0.15 gets 0 marks.
    """
    if similarity < 0.08:
        return 0
    scaled = similarity ** 0.48
    marks = max_marks * scaled
    if similarity >= 0.78:
        marks = max(marks, max_marks * min(1.0, similarity * 0.98))
    return min(max_marks, round(marks))


# ── Feedback Generator ────────────────────────────────────────────────────────

def _generate_feedback(similarity: float, key_tokens: list, student_tokens: list, key_text: str = "") -> str:
    """
    Generate human-readable feedback based on similarity score
    and identify key concepts that are missing from the student's answer.
    """
    key_set     = set(key_tokens)
    student_set = set(student_tokens)
    missing     = key_set - student_set

    # Show readable keywords from original text instead of lemmatized forms
    if key_text and missing:
        original_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', key_text.lower()))
        readable_missing = []
        for orig_word in sorted(original_words, key=len, reverse=True):
            if LEMMATIZER:
                lemma = LEMMATIZER.lemmatize(orig_word)
            else:
                lemma = orig_word
            if lemma in missing and orig_word not in STOPWORDS:
                readable_missing.append(orig_word)
        missing_display = readable_missing[:5]
    else:
        # Limit missing keywords shown to top 5 (most meaningful are longer words)
        missing_display = sorted(missing, key=len, reverse=True)[:5]

    missing_str = ", ".join(missing_display) if missing_display else "none"

    if similarity >= 0.85:
        grade = "✅ Excellent"
        comment = "The answer closely matches the key. Well done!"
    elif similarity >= 0.70:
        grade = "✅ Good"
        comment = f"Good answer. Minor concepts missing: {missing_str}."
    elif similarity >= 0.55:
        grade = "⚠️ Satisfactory"
        comment = f"Partially correct. Missing concepts: {missing_str}."
    elif similarity >= 0.40:
        grade = "⚠️ Needs Improvement"
        comment = f"Weak answer. Several key ideas absent: {missing_str}."
    elif similarity >= 0.25:
        grade = "❌ Poor"
        comment = f"Answer is mostly off-topic. Key missing: {missing_str}."
    else:
        grade = "❌ Very Poor / Off-topic"
        comment = "Answer does not match the expected response."

    return f"{grade} — {comment}"
