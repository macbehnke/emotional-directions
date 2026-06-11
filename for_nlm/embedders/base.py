from __future__ import annotations

import hashlib
import json
import math
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from config import ModelConfig


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    if np.isnan(vector).any():
        raise ValueError("Embedding zawiera NaN.")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or math.isclose(norm, 0.0):
        warnings.warn("Embedding ma norme 0; zwracam wektor bez normalizacji.")
        return vector
    return vector / norm


class EmbeddingCache:
    """JSONL cache keyed by model, language, text, dimension, and pooling."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                self._data[record["key"]] = np.asarray(record["vector"], dtype=np.float32)

    def get(self, key: str) -> np.ndarray | None:
        return self._data.get(key)

    def set(self, key: str, vector: np.ndarray) -> None:
        self._data[key] = np.asarray(vector, dtype=np.float32)
        record = {"key": key, "vector": self._data[key].astype(float).tolist()}
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


class Embedder(ABC):
    def __init__(self, model: ModelConfig, cache: EmbeddingCache) -> None:
        self.model = model
        self.cache = cache

    def embed(self, text: str, language: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Nie mozna embedowac pustego tekstu.")
        key = self.cache_key(text, language)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        vector = np.asarray(self._embed_uncached(text, language), dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f"Embedding musi byc 1D, dostalem shape={vector.shape}.")
        if self.model.embedding_dimension and vector.shape[0] != self.model.embedding_dimension:
            warnings.warn(
                f"Model {self.model.name}: oczekiwano wymiaru "
                f"{self.model.embedding_dimension}, dostalem {vector.shape[0]}."
            )
        if self.model.normalize:
            vector = normalize_vector(vector)
        self.cache.set(key, vector)
        return vector

    def cache_key(self, text: str, language: str) -> str:
        payload = {
            "model_name": self.model.name,
            "model_id": self.model.model_id,
            "language": language,
            "text": text,
            "embedding_dimension": self.model.embedding_dimension,
            "pooling_strategy": self.model.pooling_strategy,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @abstractmethod
    def _embed_uncached(self, text: str, language: str) -> np.ndarray:
        raise NotImplementedError


class HashEmbedder(Embedder):
    """Deterministic test embedder. It is only for pipeline smoke tests."""

    def _embed_uncached(self, text: str, language: str) -> np.ndarray:
        dim = self.model.embedding_dimension or 128
        digest = hashlib.sha256(
            f"{self.model.model_id}:{language}:{text}".encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(seed)
        return rng.normal(size=dim).astype(np.float32)


def get_embedder(model: ModelConfig, cache: EmbeddingCache) -> Embedder:
    if model.backend == "hash":
        return HashEmbedder(model, cache)
    if model.backend == "gemini":
        from embeddings.gemini_embedder import GeminiEmbedder

        return GeminiEmbedder(model, cache)
    if model.backend == "sentence-transformers":
        from embeddings.qwen_embedder import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model, cache)
    if model.backend == "bielik":
        from embeddings.bielik_embedder import BielikEmbedder

        return BielikEmbedder(model, cache)
    if model.backend == "openai":
        from embeddings.openai_embedder import OpenAIEmbedder

        return OpenAIEmbedder(model, cache)
    raise ValueError(f"Nieznany backend: {model.backend}")
