from __future__ import annotations

import os

import numpy as np

from embeddings.base import Embedder


class OpenAIEmbedder(Embedder):
    """Optional OpenAI comparison backend, e.g. text-embedding-3-large."""

    def _embed_uncached(self, text: str, language: str) -> np.ndarray:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Brak OPENAI_API_KEY w zmiennych srodowiskowych.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Zainstaluj: pip install openai") from exc

        client = OpenAI()
        response = client.embeddings.create(model=self.model.model_id, input=text)
        return np.asarray(response.data[0].embedding, dtype=np.float32)
