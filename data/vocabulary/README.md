# Vocabulary search space

Put one word per line in:

```text
data/vocabulary/en.txt
data/vocabulary/pl.txt
data/vocabulary/zh.txt
```

When `search_scope: "vocabulary"`, the experiment ranks words from these files
instead of category candidate strings.

If a language file is missing, the code falls back to extracting individual
words from `data/candidates/{language}/*.txt`. That fallback is only for smoke
tests and backwards compatibility. For the cleaner experiment, replace it with a
large external word list.

Recommended sources:

- a frequency list such as wordfreq/wordfreq-top words,
- a corpus-derived vocabulary from the language being tested,
- a curated word list filtered to nouns/adjectives if the analysis should avoid
function words.

Keep the list finite and documented, because nearest-neighbor search always
depends on the chosen search vocabulary.
