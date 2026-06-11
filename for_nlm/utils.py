from __future__ import annotations

import warnings
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


def load_candidates(candidate_root: Path, language: str, category: str, scope: str) -> list[dict[str, str]]:
    if scope not in {"category", "language"}:
        raise ValueError("search_scope musi byc 'category' albo 'language'.")

    records: list[dict[str, str]] = []
    if scope == "category":
        path = candidate_path(candidate_root, language, category)
        return [
            {"candidate": item, "candidate_category": category}
            for item in load_lines(path)
        ]

    language_dir = candidate_root / language
    if not language_dir.exists():
        warnings.warn(f"Brak katalogu kandydatow: {language_dir}")
        return []
    for path in sorted(language_dir.glob("*.txt")):
        candidate_category = path.stem
        for item in load_lines(path):
            records.append({"candidate": item, "candidate_category": candidate_category})
    return records


def contains_any(text: str, terms: list[str]) -> bool:
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in terms if term)
