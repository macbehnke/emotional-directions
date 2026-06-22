from __future__ import annotations

import warnings
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from embeddings.base import normalize_vector


def mean_vector(vectors: Iterable[np.ndarray]) -> np.ndarray:
    vectors = [np.asarray(vector, dtype=np.float32) for vector in vectors]
    validate_same_dimension(vectors)
    return normalize_vector(np.mean(vectors, axis=0))


def validate_same_dimension(vectors: list[np.ndarray]) -> None:
    if not vectors:
        raise ValueError("Brak wektorow do walidacji.")
    shapes = {vector.shape for vector in vectors}
    if len(shapes) != 1:
        raise ValueError(f"Niezgodne wymiary embeddingow: {sorted(shapes)}")
    for vector in vectors:
        if np.isnan(vector).any():
            raise ValueError("Embedding zawiera NaN.")


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        warnings.warn(f"Brak pliku: {path}")
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def candidate_path(candidate_root: Path, language: str, category: str) -> Path:
    return candidate_root / language / f"{category}.txt"


def vocabulary_path(vocabulary_root: Path, language: str) -> Path:
    return vocabulary_root / f"{language}.txt"


def candidate_records_from_lines(
    lines: list[str],
    candidate_category: str,
    candidate_unit: str,
) -> list[dict[str, str]]:
    if candidate_unit not in {"word", "text"}:
        raise ValueError("candidate_unit musi byc 'word' albo 'text'.")

    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in lines:
        if candidate_unit == "text":
            units = [line]
        else:
            units = re.findall(r"\b\w+\b", line.lower(), flags=re.UNICODE)

        for unit in units:
            normalized = unit.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            records.append({
                "candidate": normalized,
                "candidate_category": candidate_category,
                "candidate_source": line,
            })
    return records


def load_candidates(
    candidate_root: Path,
    vocabulary_root: Path,
    language: str,
    category: str,
    scope: str,
    candidate_unit: str = "word",
) -> list[dict[str, str]]:
    """Load rankable candidate units.

    candidate_unit="word" ranks individual words extracted from candidate
    lines. candidate_unit="text" ranks each full line, including phrases.
    """
    if scope not in {"category", "language", "vocabulary"}:
        raise ValueError("search_scope musi byc 'category', 'language' albo 'vocabulary'.")

    records: list[dict[str, str]] = []
    if scope == "vocabulary":
        path = vocabulary_path(vocabulary_root, language)
        if path.exists():
            return candidate_records_from_lines(load_lines(path), "vocabulary", "word")
        warnings.warn(
            f"Brak slownika {path}; awaryjnie buduje slownik z data/candidates/{language}."
        )
        language_dir = candidate_root / language
        if not language_dir.exists():
            warnings.warn(f"Brak katalogu kandydatow: {language_dir}")
            return []
        for candidate_file in sorted(language_dir.glob("*.txt")):
            records.extend(candidate_records_from_lines(load_lines(candidate_file), "vocabulary", "word"))
        return records

    if scope == "category":
        path = candidate_path(candidate_root, language, category)
        return candidate_records_from_lines(load_lines(path), category, candidate_unit)

    language_dir = candidate_root / language
    if not language_dir.exists():
        warnings.warn(f"Brak katalogu kandydatow: {language_dir}")
        return []
    for path in sorted(language_dir.glob("*.txt")):
        candidate_category = path.stem
        records.extend(candidate_records_from_lines(load_lines(path), candidate_category, candidate_unit))
    return records


def contains_any(text: str, terms: list[str]) -> bool:
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in terms if term)
