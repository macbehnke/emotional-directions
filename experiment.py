from __future__ import annotations

import argparse
import csv
import hashlib
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


def get_shuffled_emotion_id(config: ExperimentConfig, emotion_id: str) -> str:
    emotion_ids = list(config.emotions)
    if len(emotion_ids) < 2:
        raise ValueError("shuffled_emotion wymaga co najmniej dwoch emocji w konfiguracji.")
    index = emotion_ids.index(emotion_id)
    return emotion_ids[(index + 1) % len(emotion_ids)]


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
    category_vector: np.ndarray,
    emotion_direction: np.ndarray,
) -> list[dict[str, Any]]:
    if not candidate_records:
        raise ValueError("Lista kandydatow jest pusta.")
    vectors = [embedder.embed(record["candidate"], language) for record in candidate_records]
    validate_same_dimension([query_vector, *vectors])

    query_vector = query_vector / np.linalg.norm(query_vector)
    emotion_direction_norm = np.linalg.norm(emotion_direction)
    if emotion_direction_norm == 0.0:
        normalized_direction = emotion_direction
    else:
        normalized_direction = emotion_direction / emotion_direction_norm
    rows: list[dict[str, Any]] = []
    for record, vector in zip(candidate_records, vectors):
        vector = vector / np.linalg.norm(vector)
        candidate_delta = vector - category_vector
        rows.append({
            **record,
            "cosine_similarity": float(np.dot(query_vector, vector)),
            "projection_on_emotion_direction": float(np.dot(candidate_delta, normalized_direction)),
        })
    rows.sort(key=lambda item: item["cosine_similarity"], reverse=True)
    for rank, row in enumerate(rows[:top_k], start=1):
        row["rank"] = rank
    return rows[:top_k]


def filter_candidate_records(
    candidate_records: list[dict[str, str]],
    blocked_terms: list[str],
) -> list[dict[str, str]]:
    blocked = {term.strip().lower() for term in blocked_terms if term and term.strip()}
    if not blocked:
        return candidate_records
    return [
        record for record in candidate_records
        if record["candidate"].strip().lower() not in blocked
    ]


def deterministic_random_direction(
    dimension: int,
    seed: int,
    *parts: str,
) -> np.ndarray:
    raw = "|".join([str(seed), *parts])
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    rng_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(rng_seed)
    vector = rng.normal(size=dimension).astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def build_query_for_condition(
    condition: str,
    category_vector: np.ndarray,
    emotion_vector: np.ndarray,
    neutral_vector: np.ndarray,
    random_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if condition in {"emotion", "shuffled_emotion"}:
        direction = emotion_vector - neutral_vector
    elif condition == "identity":
        direction = neutral_vector - neutral_vector
    elif condition == "random":
        direction = random_direction
    else:
        raise ValueError(f"Nieznany warunek kontrolny: {condition}")
    query = category_vector + direction
    query_norm = np.linalg.norm(query)
    if query_norm == 0.0:
        raise ValueError("Query vector ma norme 0.")
    return query / query_norm, direction


def bootstrap_stability(
    *,
    config: ExperimentConfig,
    embedder,
    model_name: str,
    language: str,
    category_id: str,
    category_vector: np.ndarray,
    candidate_records: list[dict[str, str]],
    emotion_id: str,
    emotion_strategy: str,
    emotion_terms: list[str],
    neutral_strategy: str,
    neutral_terms: list[str],
    reference_direction: np.ndarray,
    reference_candidates: set[str],
) -> list[dict[str, Any]]:
    if config.bootstrap_iterations <= 0:
        return []
    if len(emotion_terms) < 2 and len(neutral_terms) < 2:
        return []

    records: list[dict[str, Any]] = []
    for iteration in range(1, config.bootstrap_iterations + 1):
        rng = np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(
                    f"{config.random_seed}|{model_name}|{language}|{category_id}|"
                    f"{emotion_id}|{emotion_strategy}|{neutral_strategy}|{iteration}".encode("utf-8")
                ).digest()[:8],
                byteorder="little",
                signed=False,
            )
        )
        sampled_emotion = [
            emotion_terms[int(rng.integers(0, len(emotion_terms)))]
            for _ in range(max(1, len(emotion_terms)))
        ]
        sampled_neutral = [
            neutral_terms[int(rng.integers(0, len(neutral_terms)))]
            for _ in range(max(1, len(neutral_terms)))
        ]
        boot_emotion = get_emotion_vector(embedder, sampled_emotion, language)
        boot_neutral = get_emotion_vector(embedder, sampled_neutral, language)
        boot_direction = boot_emotion - boot_neutral
        boot_query = category_vector + boot_direction
        boot_query = boot_query / np.linalg.norm(boot_query)
        boot_rows = nearest_neighbors(
            boot_query,
            candidate_records,
            embedder,
            language,
            config.top_k,
            category_vector,
            reference_direction,
        )
        boot_candidates = {row["candidate"] for row in boot_rows}
        union = reference_candidates | boot_candidates
        reference_norm = np.linalg.norm(reference_direction)
        boot_norm = np.linalg.norm(boot_direction)
        direction_cosine = float(
            np.dot(reference_direction, boot_direction) / (reference_norm * boot_norm)
        ) if reference_norm and boot_norm else 0.0
        records.append({
            "model": model_name,
            "language": language,
            "category": category_id,
            "emotion": emotion_id,
            "emotion_strategy": emotion_strategy,
            "neutral_strategy": neutral_strategy,
            "bootstrap_iteration": iteration,
            "sampled_emotion_terms": "; ".join(sampled_emotion),
            "sampled_neutral_terms": "; ".join(sampled_neutral),
            "direction_cosine_to_reference": direction_cosine,
            "top_k_jaccard_to_reference": len(reference_candidates & boot_candidates) / len(union) if union else 0.0,
            "average_similarity_top_k": float(np.mean([row["cosine_similarity"] for row in boot_rows])),
        })
    return records


def run_experiment(config: ExperimentConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = enabled_models(config)
    if not models:
        raise ValueError("Brak wlaczonych modeli w config.yaml.")

    cache = EmbeddingCache(config.cache_path)
    rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []

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
                        if config.exclude_query_terms_from_candidates:
                            candidate_records_for_query = filter_candidate_records(
                                candidate_records,
                                [category_text, *emotion_terms],
                            )
                        else:
                            candidate_records_for_query = candidate_records
                        if not candidate_records_for_query:
                            warnings.warn(
                                f"Po filtrowaniu brak kandydatow: {language}/{category_id}/{emotion_id}"
                            )
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
                            reference_direction = emotion_vector - neutral_vector
                            random_direction = deterministic_random_direction(
                                len(category_vector),
                                config.random_seed,
                                model_name,
                                language,
                                category_id,
                                emotion_id,
                            )

                            reference_candidates: set[str] | None = None
                            for condition in config.control_conditions:
                                condition_emotion_id = emotion_id
                                condition_emotion_terms = emotion_terms
                                condition_emotion_vector = emotion_vector
                                if condition == "shuffled_emotion":
                                    condition_emotion_id = get_shuffled_emotion_id(config, emotion_id)
                                    condition_emotion_terms = get_emotion_terms(
                                        config,
                                        condition_emotion_id,
                                        language,
                                        emotion_strategy,
                                    )
                                    if not condition_emotion_terms:
                                        continue
                                    condition_emotion_vector = get_emotion_vector(
                                        embedder,
                                        condition_emotion_terms,
                                        language,
                                    )

                                query_vector, direction = build_query_for_condition(
                                    condition,
                                    category_vector,
                                    condition_emotion_vector,
                                    neutral_vector,
                                    random_direction,
                                )

                                top_rows = nearest_neighbors(
                                    query_vector,
                                    candidate_records_for_query,
                                    embedder,
                                    language,
                                    config.top_k,
                                    category_vector,
                                    reference_direction,
                                )
                                avg_similarity = float(np.mean([
                                    row["cosine_similarity"] for row in top_rows
                                ]))
                                avg_projection = float(np.mean([
                                    row["projection_on_emotion_direction"] for row in top_rows
                                ]))
                                retention = float(np.mean([
                                    row["candidate_category"] == category_id for row in top_rows
                                ]))

                                if condition == "emotion":
                                    reference_candidates = {row["candidate"] for row in top_rows}

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
                                        "condition": condition,
                                        "condition_emotion": condition_emotion_id,
                                        "emotion_strategy": emotion_strategy,
                                        "emotion_terms": "; ".join(emotion_terms),
                                        "condition_emotion_terms": "; ".join(condition_emotion_terms),
                                        "neutral_strategy": neutral_strategy,
                                        "neutral_terms": "; ".join(selected_neutral),
                                        "candidate": candidate,
                                        "candidate_category": row["candidate_category"],
                                        "rank": row["rank"],
                                        "cosine_similarity": row["cosine_similarity"],
                                        "projection_on_emotion_direction": row["projection_on_emotion_direction"],
                                        "same_category": row["candidate_category"] == category_id,
                                        "candidate_identical_to_category": identical_category,
                                        "candidate_contains_emotion": contains_emotion,
                                        "average_similarity_top_k": avg_similarity,
                                        "average_projection_top_k": avg_projection,
                                        "category_retention_at_k": retention,
                                        "direction_norm": float(np.linalg.norm(direction)),
                                        "query_norm": float(np.linalg.norm(query_vector)),
                                    })

                            if reference_candidates is not None:
                                stability_rows.extend(bootstrap_stability(
                                    config=config,
                                    embedder=embedder,
                                    model_name=model_name,
                                    language=language,
                                    category_id=category_id,
                                    category_vector=category_vector,
                                    candidate_records=candidate_records_for_query,
                                    emotion_id=emotion_id,
                                    emotion_strategy=emotion_strategy,
                                    emotion_terms=emotion_terms,
                                    neutral_strategy=neutral_strategy,
                                    neutral_terms=selected_neutral,
                                    reference_direction=reference_direction,
                                    reference_candidates=reference_candidates,
                                ))
    return rows, stability_rows


def export_results(
    rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
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
    if stability_rows:
        stability = pd.DataFrame(stability_rows)
        stability.to_csv(output_dir / "bootstrap_stability.csv", index=False)
        stability.to_excel(output_dir / "bootstrap_stability.xlsx", index=False)


def write_summaries(frame, output_dir: Path) -> None:
    top10 = frame[frame["rank"] <= 10].copy()
    by_strategy = (
        top10.groupby(["condition", "emotion_strategy", "neutral_strategy"], dropna=False)
        .agg(
            mean_cosine_similarity_top_10=("cosine_similarity", "mean"),
            mean_projection_top_10=("projection_on_emotion_direction", "mean"),
            category_retention_at_10=("same_category", "mean"),
            unique_candidates=("candidate", "nunique"),
        )
        .reset_index()
    )
    by_model = (
        top10.groupby(["model", "pooling_strategy", "condition"], dropna=False)
        .agg(
            mean_cosine_similarity_top_10=("cosine_similarity", "mean"),
            mean_projection_top_10=("projection_on_emotion_direction", "mean"),
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
        "condition",
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
        "condition",
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
        help=(
            "Opcjonalna lista modeli po przecinku, np. "
            "bielik_1_5b_v3,gemini,qwen3_embedding_8b,arctic_embed_l_v2"
        ),
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

    rows, stability_rows = run_experiment(config)
    export_results(rows, stability_rows, config.output_dir)
    print(f"Zapisano {len(rows)} wierszy do {config.output_dir}")
    if stability_rows:
        print(f"Zapisano {len(stability_rows)} wierszy bootstrap stability")


if __name__ == "__main__":
    main()
