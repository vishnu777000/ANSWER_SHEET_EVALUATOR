import math
import os
import re
from functools import lru_cache
from pathlib import Path


STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "to", "of", "in", "on", "for", "and", "or", "but",
    "with", "by", "as", "at", "from", "that", "this", "it", "its",
    "can", "could", "should", "would", "will", "may", "might",
    "has", "have", "had", "do", "does", "did", "which", "who",
    "what", "when", "where", "how", "than", "then", "into",
    "their", "there", "these", "those", "they", "them", "we",
    "you", "your", "our", "not", "also", "such", "using", "used",
    "use", "helps", "help", "explain", "define", "describe", "user",
    "users", "so", "that", "it", "be", "able", "while", "very",
    "must", "each", "contain", "contains", "mark", "marks",
    "commonly", "process", "uses", "during",
}

CONCEPT_ALIASES = {
    "information": "data",
    "info": "data",
    "organised": "organize",
    "organized": "organize",
    "organization": "organize",
    "organisation": "organize",
    "collection": "collect",
    "collections": "collect",
    "related": "relate",
    "relates": "relate",
    "stored": "store",
    "storing": "store",
    "storage": "store",
    "electronically": "electronic",
    "accessed": "access",
    "accessing": "access",
    "accessible": "access",
    "retrieved": "access",
    "retrieval": "access",
    "retrieve": "access",
    "managed": "manage",
    "management": "manage",
    "updated": "update",
    "updating": "update",
    "efficiently": "efficient",
    "efficiency": "efficient",
    "uniquely": "unique",
    "organizing": "organize",
    "organizes": "organize",
    "reduces": "reduce",
    "reduced": "reduce",
    "redundant": "redundancy",
    "duplicate": "redundancy",
    "duplicates": "redundancy",
    "duplicated": "redundancy",
    "avoids": "prevent",
    "avoid": "prevent",
    "problems": "anomaly",
    "problem": "anomaly",
    "anomalies": "anomaly",
    "processes": "process",
    "processed": "process",
    "processing": "process",
    "inspired": "inspire",
    "inspiration": "inspire",
    "based": "base",
    "dataset": "data",
    "datasets": "data",
    "examples": "example",
    "improves": "improve",
    "improved": "improve",
    "improving": "improve",
    "performance": "perform",
    "performs": "perform",
    "evaluation": "evaluate",
    "evaluates": "evaluate",
    "evaluated": "evaluate",
    "evaluating": "evaluate",
    "checks": "check",
    "checked": "check",
    "checking": "check",
    "predictions": "predict",
    "prediction": "predict",
    "predicts": "predict",
    "predicted": "predict",
    "predicting": "predict",
    "estimates": "estimate",
    "estimated": "estimate",
    "estimating": "estimate",
    "outcomes": "outcome",
    "decisions": "decision",
    "categories": "category",
    "classes": "class",
    "identifies": "identify",
    "identified": "identify",
    "identifying": "identify",
    "interconnected": "interconnect",
    "interconnects": "interconnect",
    "unlabeled": "unlabel",
    "unlabelled": "unlabel",
}

NEGATION_WORDS = {
    "not", "never", "no", "cannot", "can't", "without", "isn't",
    "aren't", "wasn't", "weren't", "doesn't", "don't", "didn't",
}

REVIEW_COVERAGE_MIN = 0.25
REVIEW_COVERAGE_MAX = 0.55
DECISION_THRESHOLD = 0.375
SEMANTIC_DISAGREEMENT_MIN = 0.25


def _tokens(text):
    normalized = str(text or "").lower()
    for contraction, expanded in (
        ("can't", "can not"),
        ("cannot", "can not"),
        ("isn't", "is not"),
        ("aren't", "are not"),
        ("wasn't", "was not"),
        ("weren't", "were not"),
        ("doesn't", "does not"),
        ("don't", "do not"),
        ("didn't", "did not"),
    ):
        normalized = normalized.replace(contraction, expanded)

    words = re.findall(r"\d+[a-z]+|[a-z][a-z0-9+#]*|\d+", normalized)
    concepts = []
    positions = []

    for index, word in enumerate(words):
        word = word.strip(".-")
        if not word:
            continue
        if word in STOP_WORDS or len(word) <= 1:
            continue

        concept = _normalize_concept(word)

        if concept in STOP_WORDS:
            continue

        concepts.append(concept)
        positions.append(index)

    return words, concepts, positions


def _normalize_concept(word):
    """Normalize common English inflections without conflating word meaning."""
    alias = CONCEPT_ALIASES.get(word)
    if alias is not None:
        return alias

    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return {
            "mak": "make",
            "us": "use",
            "produc": "produce",
            "reduc": "reduce",
            "increas": "increase",
            "analys": "analysis",
        }.get(stem, stem)
    if len(word) > 4 and word.endswith("ed"):
        stem = word[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return {
            "inspir": "inspire",
            "bas": "base",
            "produc": "produce",
            "reduc": "reduce",
            "improv": "improve",
            "evaluat": "evaluate",
            "identifi": "identify",
        }.get(stem, stem)
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def normalize_text(text):
    _, concepts, _ = _tokens(text)
    return " ".join(concepts)


def extract_keywords(text):
    _, concepts, _ = _tokens(text)
    return concepts


def calculate_coverage(student_answer, reference_answer):
    reference_concepts = _concept_set(reference_answer)
    student_concepts = _concept_set(student_answer)
    if not reference_concepts:
        return 0.0
    return len(reference_concepts & student_concepts) / len(reference_concepts)


def calculate_precision(student_answer, reference_answer):
    student_concepts = _concept_set(student_answer)
    reference_concepts = _concept_set(reference_answer)
    if not student_concepts:
        return 0.0
    return len(student_concepts & reference_concepts) / len(student_concepts)


def _concept_set(text):
    concepts = set(extract_keywords(text))
    if {"normal", "form"} <= concepts or any(
        concept in {"1nf", "2nf", "3nf"} for concept in concepts
    ):
        concepts.difference_update({"normal", "form", "nf", "1nf", "2nf", "3nf"})
        concepts.add("normal_form")
    if {"store", "electronic"} <= concepts:
        concepts.difference_update({"store", "electronic"})
        concepts.add("electronic_storage")
    return concepts


def _cached_model_path():
    model_cache_name = "models--sentence-transformers--all-MiniLM-L6-v2"
    cache_roots = {
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.environ.get("TRANSFORMERS_CACHE"),
        str(Path.home() / ".cache" / "huggingface" / "hub"),
    }

    for cache_root in filter(None, cache_roots):
        model_root = Path(cache_root) / model_cache_name
        snapshots = model_root / "snapshots"
        if snapshots.is_dir():
            for snapshot in snapshots.iterdir():
                if (snapshot / "config.json").is_file():
                    return snapshot

    legacy_cache = (
        Path.home()
        / ".cache"
        / "torch"
        / "sentence_transformers"
        / "sentence-transformers_all-MiniLM-L6-v2"
    )
    if (legacy_cache / "modules.json").is_file():
        return legacy_cache

    return None


@lru_cache(maxsize=1)
def _load_cached_sentence_model():
    model_path = _cached_model_path()
    if model_path is None:
        raise FileNotFoundError(
            "all-MiniLM-L6-v2 is not present in the local model cache; "
            "no network download was attempted."
        )

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        str(model_path),
        local_files_only=True,
    )


def _semantic_similarity(student_answer, reference_answer):
    model = _load_cached_sentence_model()
    embeddings = model.encode(
        [student_answer, reference_answer],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return max(0.0, min(1.0, float(embeddings[0] @ embeddings[1])))


def _negated_concepts(text):
    words, concepts, positions = _tokens(text)
    negated = set()
    concepts_by_position = dict(zip(positions, concepts))
    negation_pending = False

    for position, raw_word in enumerate(words):
        word = raw_word.strip(".-")
        if word in NEGATION_WORDS:
            negation_pending = True
            continue

        concept = concepts_by_position.get(position)
        if concept is not None:
            if negation_pending:
                negated.add(concept)
            negation_pending = False

        if raw_word.endswith((".", "!", "?")):
            negation_pending = False

    return negated


def _concept_label(concept):
    labels = {
        "organize": "organization",
        "collect": "collection",
        "relate": "relatedness",
        "data": "data/information",
        "store": "storage",
        "electronic": "electronic storage",
        "electronic_storage": "electronic storage",
        "normal_form": "normal forms",
        "access": "access",
        "manage": "management",
        "update": "updating",
        "efficient": "efficiency",
        "prevent": "prevention",
    }
    return labels.get(concept, concept.replace("_", " "))


def _format_concepts(concepts):
    return ", ".join(_concept_label(item) for item in sorted(concepts))


def _review_reasons(
    coverage,
    semantic_similarity=None,
    incorrect_concepts=None,
    student_answer="",
    reference_answer="",
):
    reasons = []
    if not student_answer:
        reasons.append("No student answer detected; verify text extraction.")
    if not reference_answer:
        reasons.append("Reference answer is missing; grading needs manual review.")
    if incorrect_concepts:
        reasons.append("The answer contradicts one or more reference concepts.")
    if REVIEW_COVERAGE_MIN <= coverage <= REVIEW_COVERAGE_MAX:
        reasons.append("Concept coverage is borderline.")
    if semantic_similarity is not None:
        coverage_pass = coverage >= DECISION_THRESHOLD
        semantic_pass = semantic_similarity >= DECISION_THRESHOLD
        if (
            coverage_pass != semantic_pass
            and abs(coverage - semantic_similarity) >= SEMANTIC_DISAGREEMENT_MIN
        ):
            reasons.append(
                "Semantic similarity and concept coverage disagree; check meaning manually."
            )
    return reasons


def evaluate_answer(
    student_answer,
    reference_answer,
    max_marks=5,
    use_semantic=False,
):
    try:
        max_marks = float(max_marks)
    except (TypeError, ValueError):
        raise ValueError("Maximum marks must be a number.")

    if not math.isfinite(max_marks):
        raise ValueError("Maximum marks must be a finite number.")

    if max_marks < 0:
        raise ValueError("Maximum marks cannot be negative.")

    student_answer = str(student_answer or "").strip()
    reference_answer = str(reference_answer or "").strip()

    if not student_answer:
        review_reasons = _review_reasons(
            0.0,
            student_answer=student_answer,
            reference_answer=reference_answer,
        )
        return {
            "marks": 0.0,
            "max_marks": max_marks,
            "coverage": 0.0,
            "precision": 0.0,
            "similarity": 0.0,
            "semantic_similarity": None,
            "covered_concepts": [],
            "missing_concepts": extract_keywords(reference_answer),
            "incorrect_concepts": [],
            "feedback": "No student answer was detected; no reference concepts were covered.",
            "review_recommended": bool(review_reasons),
            "review_reasons": review_reasons,
        }

    if not reference_answer:
        review_reasons = _review_reasons(
            0.0,
            student_answer=student_answer,
            reference_answer=reference_answer,
        )
        return {
            "marks": 0.0,
            "max_marks": max_marks,
            "coverage": 0.0,
            "precision": 0.0,
            "similarity": 0.0,
            "semantic_similarity": None,
            "covered_concepts": [],
            "missing_concepts": [],
            "incorrect_concepts": [],
            "feedback": "Reference answer is missing; grading was skipped.",
            "review_recommended": bool(review_reasons),
            "review_reasons": review_reasons,
        }

    if max_marks == 0:
        return {
            "marks": 0.0,
            "max_marks": 0.0,
            "coverage": 0.0,
            "precision": 0.0,
            "similarity": 0.0,
            "semantic_similarity": None,
            "covered_concepts": [],
            "missing_concepts": [],
            "incorrect_concepts": [],
            "feedback": "Maximum marks are zero.",
            "review_recommended": False,
            "review_reasons": [],
        }

    reference_concepts = _concept_set(reference_answer)
    student_concepts = _concept_set(student_answer)
    matched = reference_concepts & student_concepts
    reference_negated = _negated_concepts(reference_answer)
    student_negated = _negated_concepts(student_answer)
    negated = {
        concept
        for concept in matched
        if (concept in reference_negated) != (concept in student_negated)
    }
    covered = matched - negated
    missing = reference_concepts - covered

    coverage = len(covered) / len(reference_concepts) if reference_concepts else 0.0
    precision = (
        len(covered) / len(student_concepts)
        if student_concepts else 0.0
    )

    scoring_method = "concept-keyword"
    semantic_score = None
    if use_semantic:
        try:
            semantic_score = _semantic_similarity(student_answer, reference_answer)
            scoring_method = "concept-keyword-with-semantic-review"
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                "Semantic scoring was requested, but the local "
                "all-MiniLM-L6-v2 model is unavailable. Install "
                "requirements-semantic.txt and cache the model before "
                "enabling semantic scoring."
            ) from exc

    # Each distinct reference concept earns an equal share of the marks.
    score_ratio = coverage
    marks = round(max_marks * score_ratio, 2)
    review_reasons = _review_reasons(
        coverage,
        semantic_similarity=semantic_score,
        incorrect_concepts=negated,
        student_answer=student_answer,
        reference_answer=reference_answer,
    )

    covered_labels = [_concept_label(item) for item in sorted(covered)]
    missing_labels = [_concept_label(item) for item in sorted(missing)]
    incorrect_labels = [_concept_label(item) for item in sorted(negated)]

    details = [
        "Concepts covered: "
        + (_format_concepts(covered) if covered else "none")
        + ".",
    ]
    if missing:
        details.append("Important concepts missing: " + _format_concepts(missing) + ".")
    if negated:
        details.append("Incorrect or contradicted concepts: " + _format_concepts(negated) + ".")

    details.append(
        f"Score: {marks:.2f}/{max_marks:g}; "
        f"{len(covered)} of {len(reference_concepts)} reference concepts covered "
        f"({coverage * 100:.1f}% concept coverage)."
    )
    if semantic_score is not None:
        details.append(
            f"Semantic similarity (review signal only; it does not affect marks): "
            f"{semantic_score * 100:.1f}%."
        )
    return {
        "marks": marks,
        "max_marks": max_marks,
        "coverage": round(coverage, 4),
        "precision": round(precision, 4),
        "similarity": round(score_ratio, 4),
        "semantic_similarity": (
            round(semantic_score, 4)
            if semantic_score is not None
            else None
        ),
        "covered_concepts": covered_labels,
        "missing_concepts": missing_labels,
        "incorrect_concepts": incorrect_labels,
        "scoring_method": scoring_method,
        "feedback": " ".join(part for part in details if part),
        "review_recommended": bool(review_reasons),
        "review_reasons": review_reasons,
    }
