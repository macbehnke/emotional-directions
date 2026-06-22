from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "cache"
OUTPUT_DIR = PROJECT_DIR / "outputs"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.yaml"


EMOTION_REPRESENTATION_STRATEGIES = [
    "single_word",
    "feeling_phrase",
    "centroid_words",
    "centroid_phrases",
]

NEUTRAL_STRATEGIES = [
    "single_neutral",
    "neutral_centroid",
]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    backend: str
    model_id: str
    enabled: bool
    embedding_dimension: int | None = None
    pooling_strategy: str | None = None
    normalize: bool = True
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    models: dict[str, ModelConfig]
    languages: list[str]
    categories: dict[str, dict[str, str]]
    emotions: dict[str, dict[str, dict[str, Any]]]
    neutral_terms: dict[str, list[str]]
    emotion_strategies: list[str]
    neutral_strategies: list[str]
    candidate_root: Path
    search_scope: str
    candidate_unit: str
    top_k: int
    cache_path: Path
    output_dir: Path
    control_conditions: list[str]
    bootstrap_iterations: int
    random_seed: int
    discrete_emotion_lexicon_paths: dict[str, Path]
    exclude_query_terms_from_candidates: bool


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    path = Path(path)
    data = _load_yaml_with_inheritance(path)
    base_dir = path.resolve().parent

    models = {
        name: ModelConfig(
            name=name,
            backend=cfg["backend"],
            model_id=cfg["model_id"],
            enabled=bool(cfg.get("enabled", False)),
            embedding_dimension=cfg.get("embedding_dimension"),
            pooling_strategy=cfg.get("pooling_strategy"),
            normalize=bool(cfg.get("normalize", True)),
            extra={k: v for k, v in cfg.items() if k not in {
                "backend",
                "model_id",
                "enabled",
                "embedding_dimension",
                "pooling_strategy",
                "normalize",
            }},
        )
        for name, cfg in data["models"].items()
    }

    return ExperimentConfig(
        models=models,
        languages=list(data["languages"]),
        categories=data["categories"],
        emotions=data["emotions"],
        neutral_terms=data["neutral_terms"],
        emotion_strategies=list(data.get(
            "emotion_representation_strategies",
            EMOTION_REPRESENTATION_STRATEGIES,
        )),
        neutral_strategies=list(data.get("neutral_strategies", NEUTRAL_STRATEGIES)),
        candidate_root=_resolve_path(base_dir, data.get("candidate_root", "data/candidates")),
        search_scope=data.get("search_scope", "category"),
        candidate_unit=data.get("candidate_unit", "word"),
        top_k=int(data.get("top_k", 10)),
        cache_path=_resolve_path(base_dir, data.get("cache_path", "cache/embeddings_cache.jsonl")),
        output_dir=_resolve_path(base_dir, data.get("output_dir", "outputs")),
        control_conditions=list(data.get("control_conditions", ["emotion"])),
        bootstrap_iterations=int(data.get("bootstrap_iterations", 0)),
        random_seed=int(data.get("random_seed", 42)),
        discrete_emotion_lexicon_paths={
            language: _resolve_path(base_dir, path)
            for language, path in data.get("discrete_emotion_lexicon_paths", {}).items()
        },
        exclude_query_terms_from_candidates=bool(
            data.get("exclude_query_terms_from_candidates", True)
        ),
    )


def make_test_config(config: ExperimentConfig, top_k: int = 5) -> ExperimentConfig:
    """Small deterministic run: 1 model, 1 language, 2 categories, 2 emotions."""
    test_model = ModelConfig(
        name="hash_test",
        backend="hash",
        model_id="deterministic-hash-embedding-v2",
        enabled=True,
        embedding_dimension=128,
        pooling_strategy=None,
        normalize=True,
    )
    categories = {
        key: config.categories[key]
        for key in list(config.categories.keys())[:2]
    }
    emotions = {
        key: config.emotions[key]
        for key in list(config.emotions.keys())[:2]
    }
    return ExperimentConfig(
        models={"hash_test": test_model},
        languages=[config.languages[0]],
        categories=categories,
        emotions=emotions,
        neutral_terms=config.neutral_terms,
        emotion_strategies=config.emotion_strategies,
        neutral_strategies=config.neutral_strategies,
        candidate_root=config.candidate_root,
        search_scope="category",
        candidate_unit=config.candidate_unit,
        top_k=top_k,
        cache_path=config.cache_path,
        output_dir=config.output_dir,
        control_conditions=config.control_conditions,
        bootstrap_iterations=min(config.bootstrap_iterations, 10),
        random_seed=config.random_seed,
        discrete_emotion_lexicon_paths=config.discrete_emotion_lexicon_paths,
        exclude_query_terms_from_candidates=config.exclude_query_terms_from_candidates,
    )


def enabled_models(config: ExperimentConfig) -> dict[str, ModelConfig]:
    return {name: model for name, model in config.models.items() if model.enabled}


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _load_yaml_with_inheritance(path: Path) -> dict[str, Any]:
    """Load YAML and optionally merge it over another config.

    Small final-run configs can set `inherit_from: base.yaml` and override only
    the fields that differ from the base experiment.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    inherited = data.pop("inherit_from", None)
    if not inherited:
        return data

    parent_path = Path(inherited)
    if not parent_path.is_absolute():
        parent_path = path.resolve().parent / parent_path
    parent = _load_yaml_with_inheritance(parent_path)
    return _deep_merge(parent, data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
