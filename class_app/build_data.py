from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FLOAT_COLUMNS = {
    "cosine_similarity",
    "projection_on_emotion_direction",
    "mean_cosine_similarity_top_10",
    "mean_projection_top_10",
    "category_retention_at_10",
    "delta_emotion_minus_identity",
    "delta_emotion_minus_random",
    "delta_emotion_minus_shuffled",
    "final_rank_score",
    "mean_direction_cosine",
    "mean_top_k_jaccard",
    "delta_identity",
    "delta_random",
    "delta_shuffled",
    "projection",
}


def records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.where(pd.notna(frame), None)
    output = clean.to_dict(orient="records")
    for row in output:
        for key, value in list(row.items()):
            if key in FLOAT_COLUMNS and value is not None:
                row[key] = round(float(value), 4)
    return output


def build_term_map(examples: pd.DataFrame) -> dict:
    term_map: dict[str, dict[str, dict[str, str]]] = {}
    for row in examples[["emotion", "language", "emotion_terms", "neutral_terms"]].drop_duplicates().itertuples():
        term_map.setdefault(row.emotion, {})[row.language] = {
            "emotion_terms": row.emotion_terms,
            "neutral_terms": row.neutral_terms,
        }
    return term_map


def build_payload(results_path: Path) -> dict:
    output_dir = results_path.parent
    frame = pd.read_csv(results_path)
    examples = (
        frame[(frame["condition"] == "emotion") & (frame["rank"] <= 10)]
        .sort_values(["model", "language", "category", "emotion", "rank"])
        .copy()
    )
    example_cols = [
        "model",
        "language",
        "category",
        "emotion",
        "candidate",
        "rank",
        "cosine_similarity",
        "projection_on_emotion_direction",
        "emotion_terms",
        "neutral_terms",
    ]

    model_report = pd.read_csv(output_dir / "final_model_report.csv")
    strategy_report = pd.read_csv(output_dir / "final_strategy_report.csv")
    control_deltas = pd.read_csv(output_dir / "summary_control_deltas.csv")

    emotion_summary = (
        control_deltas.groupby("emotion", dropna=False)
        .agg(
            delta_identity=("delta_emotion_minus_identity", "mean"),
            delta_random=("delta_emotion_minus_random", "mean"),
            delta_shuffled=("delta_emotion_minus_shuffled", "mean"),
            projection=("projection_emotion", "mean"),
        )
        .reset_index()
        .sort_values("emotion")
    )
    model_language_summary = (
        control_deltas.groupby(["model", "language"], dropna=False)
        .agg(
            delta_identity=("delta_emotion_minus_identity", "mean"),
            delta_random=("delta_emotion_minus_random", "mean"),
            delta_shuffled=("delta_emotion_minus_shuffled", "mean"),
        )
        .reset_index()
        .sort_values(["model", "language"])
    )

    return {
        "meta": {
            "models": sorted(frame["model"].dropna().unique().tolist()),
            "languages": sorted(frame["language"].dropna().unique().tolist()),
            "categories": sorted(frame["category"].dropna().unique().tolist()),
            "emotions": sorted(frame["emotion"].dropna().unique().tolist()),
            "row_count_final": int(len(frame)),
            "row_count_examples": int(len(examples)),
            "source_results": str(results_path).replace("\\", "/"),
        },
        "examples": records(examples[example_cols]),
        "model_report": records(model_report),
        "strategy_report": records(strategy_report),
        "emotion_summary": records(emotion_summary),
        "model_language_summary": records(model_language_summary),
        "term_map": build_term_map(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static class app data.js from experiment outputs.")
    parser.add_argument("--results", default="outputs/three_models/results_full.csv")
    parser.add_argument("--out", default="class_app/data.js")
    args = parser.parse_args()

    payload = build_payload(Path(args.results))
    js = "window.EMO_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n"
    Path(args.out).write_text(js, encoding="utf-8")
    print(f"Wrote {args.out} from {args.results}")


if __name__ == "__main__":
    main()
