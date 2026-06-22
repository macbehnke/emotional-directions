# Technical note: tokens, phrases, and what the experiment actually ranks

This project does **not** ask an embedding model to generate new words.

The experiment uses a fixed candidate list stored in:

```text
data/candidates/{language}/{category}.txt
```

The current default is:

```yaml
candidate_unit: "word"
```

In this mode, each line is split into word-like candidates before ranking. For
example, `hospital food` becomes two candidates: `hospital` and `food`.

Some input lines are already single words, for example:

```text
bread
apple
park
```

Some input lines are short phrases, for example:

```text
hospital food
comfort food
dangerous street
being betrayed
```

In `candidate_unit: "word"` mode, those phrases are not ranked as full phrases.
Only the extracted words are embedded and ranked:

```python
embedder.embed("hospital", "en")
embedder.embed("food", "en")
```

The older `candidate_unit: "text"` mode ranks the whole line as one candidate:

```python
embedder.embed("hospital food", "en")
```

This is why earlier results could contain `hospital food`: the old pipeline
ranked candidate lines as full text strings. The corrected default ranks
single-word candidates instead.

## Why this is valid

Modern text embedding models usually accept arbitrary text spans, not only
single vocabulary tokens. Internally, the model tokenizes the text into
subword/token pieces, contextualizes those pieces, and pools them into one
fixed-size vector. The API returns only the final vector for the whole input
string.

For dedicated embedding models such as Gemini Embedding, Qwen3-Embedding, and
Arctic Embed, this phrase-level embedding is the intended use case.

For Bielik, the project uses an experimental approximation: it passes the whole
string through the model and mean-pools hidden states. That is useful as a
baseline, but it should not be interpreted as a dedicated embedding model.

## What the experiment can claim

The experiment can claim:

- Given a fixed candidate list, vector arithmetic can rank candidate strings in
  ways that are often interpretable.
- Some emotion directions produce robust top candidates across models.
- Different embedding backends behave differently.

The experiment cannot claim:

- The models generated the candidate phrases.
- The results are nearest vocabulary tokens.
- The models understand emotion in a human sense.
- The phrase vectors equal a simple sum or average of individual word vectors.

## Minimal pipeline

For one language, category, and emotion, the pipeline is:

1. Embed the category text, e.g. `food`.
2. Embed emotion phrases, e.g. `feeling disgusted`, `feeling revolted`.
3. Embed neutral phrases, e.g. `neutral`, `ordinary`, `plain`, `average`.
4. Build an emotion direction:

```text
emotion_direction = mean(emotion_phrase_vectors) - mean(neutral_phrase_vectors)
```

5. Build the query:

```text
query = category_vector + emotion_direction
```

6. Embed every candidate word extracted from the candidate file.
7. Rank candidates by cosine similarity to the query.

With `candidate_unit: "word"`, the results should be single words rather than
phrases from the candidate files.
