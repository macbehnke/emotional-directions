from __future__ import annotations

import numpy as np

from embeddings.base import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Qwen3, MMLW-RoBERTa, and other local sentence-transformers baselines."""

    _loaded_models: dict[str, object] = {}

    def _embed_uncached(self, text: str, language: str) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Zainstaluj: pip install sentence-transformers") from exc

        if self.model.model_id not in self._loaded_models:
            self._loaded_models[self.model.model_id] = SentenceTransformer(self.model.model_id)
        model = self._loaded_models[self.model.model_id]
        return np.asarray(model.encode(text, normalize_embeddings=False), dtype=np.float32)
