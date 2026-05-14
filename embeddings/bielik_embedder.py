from __future__ import annotations

import numpy as np
import torch

from embeddings.base import Embedder


class BielikEmbedder(Embedder):
    """Experimental embeddings from Bielik hidden states.

    Bielik is a generative language model, not a dedicated embedding model.
    Use these vectors as an exploratory baseline only.
    """

    _loaded: dict[str, tuple[object, object, str]] = {}

    def _embed_uncached(self, text: str, language: str) -> np.ndarray:
        try:
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Zainstaluj: pip install transformers torch") from exc

        if self.model.model_id not in self._loaded:
            extra = self.model.extra or {}
            device = self._resolve_device(extra.get("device", "auto"))
            dtype = self._resolve_dtype(extra.get("torch_dtype", "auto"), device)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model.model_id,
                trust_remote_code=bool(extra.get("trust_remote_code", True)),
                token=self._hf_token(),
            )
            model_kwargs = {
                "output_hidden_states": True,
                "trust_remote_code": bool(extra.get("trust_remote_code", True)),
                "token": self._hf_token(),
            }
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            try:
                model = AutoModel.from_pretrained(self.model.model_id, **model_kwargs)
            except ValueError:
                model = AutoModelForCausalLM.from_pretrained(self.model.model_id, **model_kwargs)
            model.to(device)
            model.eval()
            self._loaded[self.model.model_id] = (tokenizer, model, device)

        tokenizer, model, device = self._loaded[self.model.model_id]
        max_length = (self.model.extra or {}).get("max_length", 128)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs.hidden_states[-1]
        attention_mask = encoded["attention_mask"]

        pooling = self.model.pooling_strategy or "mean"
        if pooling == "mean":
            vector = self._mean_pool(hidden, attention_mask)
        elif pooling == "cls":
            vector = hidden[:, 0, :]
        elif pooling == "last_token":
            vector = self._last_token_pool(hidden, attention_mask)
        else:
            raise ValueError(f"Nieznana strategia poolingu Bielika: {pooling}")
        return vector.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _hf_token() -> str | None:
        import os

        return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def _resolve_dtype(dtype: str, device: str):
        if dtype == "auto":
            return torch.float16 if device.startswith("cuda") else None
        if dtype in {"float16", "fp16"}:
            return torch.float16
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype in {"float32", "fp32"}:
            return torch.float32
        if dtype in {None, "none"}:
            return None
        raise ValueError(f"Nieznany torch_dtype dla Bielika: {dtype}")

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        summed = torch.sum(hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @staticmethod
    def _last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_indices, lengths]
