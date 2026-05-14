from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


def summarize_by_strategy(frame: pd.DataFrame) -> pd.DataFrame:
    top10 = frame[frame["rank"] <= 10].copy()
    return (
        top10.groupby(["emotion_strategy", "neutral_strategy"], dropna=False)
        .agg(
            mean_cosine_similarity_top_10=("cosine_similarity", "mean"),
            category_retention_at_10=("same_category", "mean"),
            unique_candidates=("candidate", "nunique"),
        )
        .reset_index()
    )


def summarize_by_model(frame: pd.DataFrame) -> pd.DataFrame:
    top10 = frame[frame["rank"] <= 10].copy()
    return (
        top10.groupby(["model", "pooling_strategy"], dropna=False)
        .agg(
            mean_cosine_similarity_top_10=("cosine_similarity", "mean"),
            category_retention_at_10=("same_category", "mean"),
            unique_candidates=("candidate", "nunique"),
        )
        .reset_index()
    )


def strategy_overlap(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    keys = ["model", "language", "category", "emotion", "neutral_strategy"]
    for key_values, group in frame[frame["rank"] <= 10].groupby(keys, dropna=False):
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


def run_analysis(results_path: Path) -> None:
    frame = pd.read_csv(results_path)
    output_dir = results_path.parent
    summarize_by_strategy(frame).to_csv(output_dir / "summary_by_strategy.csv", index=False)
    summarize_by_model(frame).to_csv(output_dir / "summary_by_model.csv", index=False)
    strategy_overlap(frame).to_csv(output_dir / "strategy_overlap_top10.csv", index=False)
    print(f"Zapisano analizy w {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare emotion representation strategies")
    parser.add_argument("--results", default="outputs/results_full.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(Path(args.results))
