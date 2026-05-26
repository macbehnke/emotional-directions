from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_by_strategy(frame: pd.DataFrame) -> pd.DataFrame:
    top10 = frame[frame["rank"] <= 10].copy()
    group_cols = ["emotion_strategy", "neutral_strategy"]
    if "condition" in top10.columns:
        group_cols = ["condition", *group_cols]
    agg_spec = {
        "mean_cosine_similarity_top_10": ("cosine_similarity", "mean"),
        "category_retention_at_10": ("same_category", "mean"),
        "unique_candidates": ("candidate", "nunique"),
    }
    if "projection_on_emotion_direction" in top10.columns:
        agg_spec["mean_projection_top_10"] = ("projection_on_emotion_direction", "mean")
    return (
        top10.groupby(group_cols, dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )


def summarize_by_model(frame: pd.DataFrame) -> pd.DataFrame:
    top10 = frame[frame["rank"] <= 10].copy()
    group_cols = ["model", "pooling_strategy"]
    if "condition" in top10.columns:
        group_cols.append("condition")
    agg_spec = {
        "mean_cosine_similarity_top_10": ("cosine_similarity", "mean"),
        "category_retention_at_10": ("same_category", "mean"),
        "unique_candidates": ("candidate", "nunique"),
    }
    if "projection_on_emotion_direction" in top10.columns:
        agg_spec["mean_projection_top_10"] = ("projection_on_emotion_direction", "mean")
    return (
        top10.groupby(group_cols, dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )


def strategy_overlap(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    filtered = frame[frame["rank"] <= 10].copy()
    if "condition" in filtered.columns:
        filtered = filtered[filtered["condition"] == "emotion"]
    keys = ["model", "language", "category", "emotion", "neutral_strategy"]
    for key_values, group in filtered.groupby(keys, dropna=False):
        strategy_sets = {
            strategy: set(items["candidate"])
            for strategy, items in group.groupby("emotion_strategy")
        }
        for left, right in combinations(sorted(strategy_sets), 2):
            union = strategy_sets[left] | strategy_sets[right]
            jaccard = len(strategy_sets[left] & strategy_sets[right]) / len(union) if union else 0.0
            records.append({
                **dict(zip(keys, key_values)),
                "strategy_a": left,
                "strategy_b": right,
                "jaccard_top_10": jaccard,
            })
    return pd.DataFrame(records)


def model_output_similarity(frame: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Compare model outputs as ranked top-k candidate lists.

    This intentionally compares retrieved lexical outputs, not raw embedding
    vectors, because different models can have different dimensionalities and
    incompatible coordinate systems.
    """
    columns = [
        "language",
        "category",
        "emotion",
        "condition",
        "emotion_strategy",
        "neutral_strategy",
        "model_a",
        "model_b",
        "top_k",
        "overlap_count",
        "jaccard_top_k",
        "overlap_coefficient_top_k",
        "rank_correlation_union",
        "mean_abs_rank_delta_shared",
        "shared_candidates",
    ]
    records = []
    filtered = frame[frame["rank"] <= top_k].copy()
    keys = [
        "language",
        "category",
        "emotion",
        "condition",
        "emotion_strategy",
        "neutral_strategy",
    ]
    for key_values, group in filtered.groupby(keys, dropna=False):
        model_lists = {
            model: (
                items.sort_values("rank")
                .drop_duplicates("candidate", keep="first")[["candidate", "rank"]]
            )
            for model, items in group.groupby("model")
        }
        for left, right in combinations(sorted(model_lists), 2):
            left_ranks = dict(zip(model_lists[left]["candidate"], model_lists[left]["rank"]))
            right_ranks = dict(zip(model_lists[right]["candidate"], model_lists[right]["rank"]))
            left_set = set(left_ranks)
            right_set = set(right_ranks)
            shared = left_set & right_set
            union = left_set | right_set
            left_rank_vector = np.array([left_ranks.get(item, top_k + 1) for item in sorted(union)])
            right_rank_vector = np.array([right_ranks.get(item, top_k + 1) for item in sorted(union)])
            if len(union) > 1 and np.std(left_rank_vector) and np.std(right_rank_vector):
                rank_correlation_union = float(np.corrcoef(left_rank_vector, right_rank_vector)[0, 1])
            else:
                rank_correlation_union = np.nan
            records.append({
                **dict(zip(keys, key_values)),
                "model_a": left,
                "model_b": right,
                "top_k": top_k,
                "overlap_count": len(shared),
                "jaccard_top_k": len(shared) / len(union) if union else 0.0,
                "overlap_coefficient_top_k": (
                    len(shared) / min(len(left_set), len(right_set))
                    if left_set and right_set else 0.0
                ),
                "rank_correlation_union": rank_correlation_union,
                "mean_abs_rank_delta_shared": (
                    float(np.mean([abs(left_ranks[item] - right_ranks[item]) for item in shared]))
                    if shared else np.nan
                ),
                "shared_candidates": "; ".join(sorted(shared)),
            })
    return pd.DataFrame(records, columns=columns)


def run_analysis(results_path: Path) -> None:
    frame = pd.read_csv(results_path)
    output_dir = results_path.parent
    summarize_by_strategy(frame).to_csv(output_dir / "summary_by_strategy.csv", index=False)
    summarize_by_model(frame).to_csv(output_dir / "summary_by_model.csv", index=False)
    strategy_overlap(frame).to_csv(output_dir / "strategy_overlap_top10.csv", index=False)
    model_output_similarity(frame, top_k=10).to_csv(
        output_dir / "model_output_similarity_top10.csv",
        index=False,
    )
    if "condition" in frame.columns:
        summarize_controls(frame).to_csv(output_dir / "summary_by_condition.csv", index=False)
        summarize_control_deltas(frame).to_csv(output_dir / "summary_control_deltas.csv", index=False)
    print(f"Zapisano analizy w {output_dir}")


def summarize_controls(frame: pd.DataFrame) -> pd.DataFrame:
    top10 = frame[frame["rank"] <= 10].copy()
    return (
        top10.groupby(["model", "language", "category", "emotion", "condition"], dropna=False)
        .agg(
            mean_cosine_similarity_top_10=("cosine_similarity", "mean"),
            mean_projection_top_10=("projection_on_emotion_direction", "mean"),
            category_retention_at_10=("same_category", "mean"),
            unique_candidates=("candidate", "nunique"),
        )
        .reset_index()
    )


def summarize_control_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    top10 = frame[frame["rank"] <= 10].copy()
    keys = ["model", "language", "category", "emotion", "emotion_strategy", "neutral_strategy"]
    grouped = (
        top10.groupby([*keys, "condition"], dropna=False)
        .agg(
            mean_projection=("projection_on_emotion_direction", "mean"),
            mean_cosine=("cosine_similarity", "mean"),
            unique_candidates=("candidate", "nunique"),
        )
        .reset_index()
    )
    grouped["condition_projection"] = grouped["condition"].map({
        "emotion": "projection_emotion",
        "identity": "projection_identity",
        "random": "projection_random",
        "shuffled_emotion": "projection_shuffled_emotion",
    }).fillna(grouped["condition"])
    projection = grouped.pivot_table(
        index=keys,
        columns="condition_projection",
        values="mean_projection",
    ).reset_index()
    for column in [
        "projection_emotion",
        "projection_identity",
        "projection_random",
        "projection_shuffled_emotion",
    ]:
        if column not in projection:
            projection[column] = pd.NA
    projection["delta_emotion_minus_identity"] = (
        projection["projection_emotion"] - projection["projection_identity"]
    )
    projection["delta_emotion_minus_random"] = (
        projection["projection_emotion"] - projection["projection_random"]
    )
    projection["delta_emotion_minus_shuffled"] = (
        projection["projection_emotion"] - projection["projection_shuffled_emotion"]
    )
    return projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare emotion representation strategies")
    parser.add_argument("--results", default="outputs/results_full.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(Path(args.results))
