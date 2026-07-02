from __future__ import annotations

import argparse
import re
from pathlib import Path


def build_wordfreq_vocabulary(language: str, limit: int, min_length: int) -> list[str]:
    try:
        from wordfreq import top_n_list
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: wordfreq. Install it with `pip install wordfreq` "
            "or `pip install -r requirements.txt`."
        ) from exc

    raw_words = top_n_list(language, limit)
    words: list[str] = []
    seen: set[str] = set()
    for word in raw_words:
        normalized = word.strip().lower()
        if len(normalized) < min_length:
            continue
        if not re.fullmatch(r"[a-z]+", normalized):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        words.append(normalized)
    return words


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a clean one-word-per-line vocabulary for nearest-neighbor search."
    )
    parser.add_argument("--language", default="en", help="wordfreq language code, e.g. en")
    parser.add_argument("--limit", type=int, default=50000, help="How many frequent entries to request")
    parser.add_argument("--min_length", type=int, default=2, help="Minimum word length")
    parser.add_argument("--out", default="data/vocabulary/en.txt", help="Output vocabulary path")
    args = parser.parse_args()

    words = build_wordfreq_vocabulary(args.language, args.limit, args.min_length)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"Wrote {len(words)} words to {out}")


if __name__ == "__main__":
    main()
