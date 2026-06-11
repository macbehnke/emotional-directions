from __future__ import annotations

import os

import numpy as np

from embeddings.base import Embedder


class GeminiEmbedder(Embedder):
    """Google Gemini embedding backend."""

    def _embed_uncached(self, text: str, language: str) -> np.ndarray:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Brak GEMINI_API_KEY albo GOOGLE_API_KEY w zmiennych srodowiskowych.")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError("Zainstaluj: pip install google-generativeai") from exc

        genai.configure(api_key=api_key)
        response = genai.embed_content(model=self.model.model_id, content=text)
        return np.asarray(response["embedding"], dtype=np.float32)
