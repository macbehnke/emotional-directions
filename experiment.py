from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from config import ExperimentConfig, enabled_models, load_config, make_test_config
from embeddings import EmbeddingCache, get_embedder
from utils import contains_any, load_candidates, mean_vector, validate_same_dimension


def get_emotion_terms(
    config: ExperimentConfig,
    emotion: str,
    language: str,
    strategy: str,
) -> list[str]:
    try:
        value = config.emotions[emotion][language][strategy]
    except KeyError:
        warnings.warn(f"Brak emocji/strategii: {emotion}/{language}/{strategy}")
        return []
    return value if isinstance(value, list) else [value]


def get_emotion_vector(embedder, terms: list[str], language: str) -> np.ndarray:
    if not terms:
        raise ValueError("Brak terminow emocji.")
    return mean_vector([embedder.embed(term, language) for term in terms])


def get_neutral_vector(
    embedder,
    neutral_terms: list[str],
    language: str,
    neutral_strategy: str,
) -> tuple[np.ndarray, list[str]]:
    if not neutral_terms:
        raise ValueError(f"Brak neutral_terms dla jezyka {language}.")
    if neutral_strategy == "single_neutral":
        selected = [neutral_terms[0]]
    elif neutral_strategy == "neutral_centroid":
        selected = neutral_terms
    else:
        raise ValueError(f"Nieznana neutral_strategy: {neutral_strategy}")
    return mean_vector([embedder.embed(term, language) for term in selected]), selected


def nearest_neighbors(
    query_vector: np.ndarray,
    candidate_records: list[dict[str, str]],
    embedder,
    language: str,
    top_k: int,
) -> list[dict[str, Any]]:
    if not candidate_records:
        raise ValueError("Lista kandydatow jest pusta.")
    vectors = [embedder.embed(record["candidate"], language) for record in candidate_records]
    validate_same_dimension([query_vector, *vectors])

    query_vector = query_vector / np.linalg.norm(query_vector)
    rows: list[dict[str, Any]] = []
    for record, vector in zip(candidate_records, vectors):
        vector = vector / np.linalg.norm(vector)
        rows.append({
            **record,
            "cosine_similarity": float(np.dot(query_vector, vector)),
        })
    rows.sort(key=lambda item: item["cosine_similarity"], reverse=True)
    for rank, row in enumerate(rows[:top_k], start=1):
        row["rank"] = rank
    return rows[:top_k]


def run_experiment(config: ExperimentConfig) -> list[dict[str, Any]]:
    models = enabled_models(config)
    if not models:
        raise ValueError("Brak wlaczonych modeli w config.yaml.")

    cache = EmbeddingCache(config.cache_path)
    rows: list[dict[str, Any]] = []

    for model_name, model in models.items():
        embedder = get_embedder(model, cache)
        for language in config.languages:
            for category_id, translations in config.categories.items():
                category_text = translations.get(language)
                if not category_text:
                    warnings.warn(f"Brak tlumaczenia kategorii: {category_id}/{language}")
                    continue

                candidate_records = load_candidates(
                    config.candidate_root,
                    language,
                    category_id,
                    config.search_scope,
                )
                if not candidate_records:
                    warnings.warn(f"Pomijam pusta liste kandydatow: {language}/{category_id}")
                    continue

                category_vector = embedder.embed(category_text, language)
                for emotion_id in config.emotions:
                    for emotion_strategy in config.emotion_strategies:
                        emotion_terms = get_emotion_terms(
                            config,
                            emotion_id,
                            language,
                            emotion_strategy,
                        )
                        if not emotion_terms:
                            continue
                        emotion_vector = get_emotion_vector(embedder, emotion_terms, language)

                        for neutral_strategy in config.neutral_strategies:
                            neutral_vector, selected_neutral = get_neutral_vector(
                                embedder,
                                config.neutral_terms.get(language, []),
                                language,
                                neutral_strategy,
                            )
                            validate_same_dimension([category_vector, emotion_vector, neutral_vector])
                            query_vector = category_vector + (emotion_vector - neutral_vector)
                            query_vector = query_vector / np.linalg.norm(query_vector)

                            top_rows = nearest_neighbors(
                                query_vector,
                                candidate_records,
                                embedder,
                                language,
                                config.top_k,
                            )
                            avg_similarity = float(np.mean([
                                row["cosine_similarity"] for row in top_rows
                            ]))
                            retention = float(np.mean([
                                row["candidate_category"] == category_id for row in top_rows
                            ]))

                            for row in top_rows:
                                candidate = row["candidate"]
                                identical_category = candidate.strip().lower() == category_text.strip().lower()
                                contains_emotion = contains_any(candidate, emotion_terms)
                                if identical_category:
                                    warnings.warn(
                                        f"Top-{row['rank']} zawiera sama kategorie: {candidate!r}"
                                    )
                                if contains_emotion:
                                    warnings.warn(
                                        f"Top-{row['rank']} zawiera slowo/fraze emocji: {candidate!r}"
                                    )

                                rows.append({
                                    "model": model_name,
                                    "model_id": model.model_id,
                                    "pooling_strategy": model.pooling_strategy or "",
                                    "language": language,
                                    "category": category_id,
                                    "category_text": category_text,
                                    "emotion": emotion_id,
                                    "emotion_strategy": emotion_strategy,
                                    "emotion_terms": "; ".join(emotion_terms),
                                    "neutral_strategy": neutral_strategy,
                                    "neutral_terms": "; ".join(selected_neutral),
                                    "candidate": candidate,
                                    "candidate_category": row["candidate_category"],
                                    "rank": row["rank"],
                                    "cosine_similarity": row["cosine_similarity"],
                                    "same_category": row["candidate_category"] == category_id,
                                    "candidate_identical_to_category": identical_category,
                                    "candidate_contains_emotion": contains_emotion,
                                    "average_similarity_top_k": avg_similarity,
                                    "category_retention_at_k": retention,
                                })
    return rows


def export_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        raise ValueError("Brak wynikow do eksportu.")
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    results_csv = output_dir / "results_full.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    import pandas as pd

    frame = pd.DataFrame(rows)
    frame.to_excel(output_dir / "results_full.xlsx", index=False)
    write_summaries(frame, output_dir)
    create_manual_rating_template(frame, output_dir / "manual_rating_template.xlsx")


def write_summaries(frame, output_dir: Path) -> None:
    top10 = frame[frame["rank"] <= 10].copy()
    by_strategy = (
        top10.groupby(["emotion_strategy", "neutral_strategy"], dropna=False)
        .agg(
            mean_cosine_similarity_top_10=("cosine_similarity", "mean"),
            category_retention_at_10=("same_category", "mean"),
            unique_candidates=("candidate", "nunique"),
        )
        .reset_index()
    )
    by_model = (
        top10.groupby(["model", "pooling_strategy"], dropna=False)
        .agg(
            mean_cosine_similarity_top_10=("cosine_similarity", "mean"),
            category_retention_at_10=("same_category", "mean"),
            unique_candidates=("candidate", "nunique"),
        )
        .reset_index()
    )
    by_strategy.to_csv(output_dir / "summary_by_strategy.csv", index=False)
    by_model.to_csv(output_dir / "summary_by_model.csv", index=False)


def create_manual_rating_template(frame, path: Path) -> None:
    columns = [
        "model",
        "language",
        "category",
        "emotion",
        "emotion_strategy",
        "neutral_strategy",
        "candidate",
        "rank",
        "cosine_similarity",
        "category_fit_rating",
        "emotion_fit_rating",
        "interpretability_rating",
        "notes",
    ]
    template = frame[[
        "model",
        "language",
        "category",
        "emotion",
        "emotion_strategy",
        "neutral_strategy",
        "candidate",
        "rank",
        "cosine_similarity",
    ]].copy()
    for column in columns:
        if column not in template:
            template[column] = ""
    template[columns].to_excel(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emotional directions experiment")
    parser.add_argument("--config", default="config.yaml", help="Sciezka do config.yaml")
    parser.add_argument("--top_k", type=int, default=None, help="Nadpisuje top_k z configu")
    parser.add_argument(
        "--models",
        default=None,
        help="Opcjonalna lista modeli po przecinku, np. bielik_1_5b_v3,gemini,qwen3_embedding_0_6b",
    )
    parser.add_argument("--test", action="store_true", help="Maly test: 1 model, 1 jezyk, 2 kategorie, 2 emocje, top_k=5")
    parser.add_argument(
        "--search_scope",
        choices=["category", "language"],
        default=None,
        help="category: kandydaci kategorii; language: caly slownik danego jezyka",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.top_k is not None:
        config = config.__class__(**{**config.__dict__, "top_k": args.top_k})
    if args.search_scope is not None:
        config = config.__class__(**{**config.__dict__, "search_scope": args.search_scope})
    if args.models:
        selected = {name.strip() for name in args.models.split(",") if name.strip()}
        models = {
            name: model.__class__(**{**model.__dict__, "enabled": name in selected})
            for name, model in config.models.items()
        }
        missing = selected - set(models)
        if missing:
            raise ValueError(f"Nieznane modele w --models: {sorted(missing)}")
        config = config.__class__(**{**config.__dict__, "models": models})
    if args.test:
        config = make_test_config(config, top_k=5)

    rows = run_experiment(config)
    export_results(rows, config.output_dir)
    print(f"Zapisano {len(rows)} wierszy do {config.output_dir}")


if __name__ == "__main__":
    main()
