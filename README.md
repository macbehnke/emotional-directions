# Emotional Directions in Embedding Spaces

Projekt bada, czy operacje w przestrzeni embeddingowej:

```text
category + emotion - neutral
```

daja interpretowalne przesuniecia semantyczne, np.:

```text
food + disgust - neutral -> worms / mold / rotten food
```

## Strategie emocji

Pipeline porownuje cztery warianty reprezentowania emocji:

- `single_word` - pojedyncze slowo emocji, np. `sadness`.
- `feeling_phrase` - fraza emocjonalna, np. `feeling sad`.
- `centroid_words` - centroid kilku slow emocji.
- `centroid_phrases` - centroid kilku fraz emocjonalnych.

Neutralnosc tez ma ablation:

- `single_neutral` - pierwsze slowo z listy neutralnej.
- `neutral_centroid` - centroid wszystkich neutralnych terminow dla jezyka.

## Modele

Konfiguracja jest w `config.yaml`.

- `gemini` - Google Gemini Embedding, domyslnie `models/gemini-embedding-001`, do 3072 wymiarow.
- `qwen3_embedding_0_6b` - lekki Qwen przez Hugging Face / sentence-transformers, do 1024 wymiarow.
- `qwen3_embedding_8b` - wiekszy Qwen do glownego porownania, `Qwen/Qwen3-Embedding-8B`, do 4096 wymiarow.
- `bielik_1_5b_v3` - Bielik przez hidden states i pooling. To wariant eksperymentalny, bo Bielik nie jest klasycznym modelem embeddingowym. Wynikow nie nalezy bezposrednio porownywac z wyspecjalizowanymi modelami embeddingowymi.
- `stella_pl` - `sdadas/stella-pl`, rekomendowany polsko-angielski baseline embeddingowy.
- `text_embedding_3_large` - placeholder pod opcjonalny model porownawczy.

## Instalacja

```powershell
cd C:\Users\macbe\OneDrive\Informatyka_WMI_Class2027\deel_learning\emotional_directions
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Do szybkiego testu bez prawdziwych modeli wystarcza tryb `--test`, ktory uzywa deterministycznego modelu `hash_test`.

## Gemini API key

```powershell
$env:GEMINI_API_KEY="TWOJ_KLUCZ_API"
```

Backend akceptuje tez `GOOGLE_API_KEY`, jesli takiej nazwy uzywasz w swoim srodowisku.

Na klastrze Linuksowym najbezpieczniej ustawic klucze jako zmienne srodowiskowe w skrypcie joba albo w prywatnym pliku z uprawnieniami `600`, np.:

```bash
export HF_TOKEN="hf_..."
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export GEMINI_API_KEY="..."
```

Nie wpisuj tokenow do `config.yaml` i nie commituj ich do repozytorium.

Nastepnie w `config.yaml` ustaw:

```yaml
models:
  gemini:
    enabled: true
```

## Struktura

```text
emotional_directions/
  config.yaml
  config.py
  embeddings/
    base.py
    gemini_embedder.py
    qwen_embedder.py
    bielik_embedder.py
    stella_embedder.py
  data/
    candidates/
      en/
      pl/
      zh/
  cache/
  outputs/
  experiment.py
  analysis.py
  utils.py
  README.md
  requirements.txt
```

## Kategorie, emocje i kandydaci

Kategorie, emocje, strategie i neutralne terminy edytuj w `config.yaml`.

Kandydatow wpisuj po jednym na linie w:

```text
data/candidates/{language}/{category}.txt
```

Domyslnie nearest neighbors szuka w pliku dla tej samej kategorii i jezyka. Aby szukac w calym slowniku danego jezyka:

```powershell
python experiment.py --config config.yaml --top_k 10 --search_scope language
```

## Test

```powershell
python experiment.py --config config.yaml --test
python analysis.py --results outputs/results_full.csv
```

Test uruchamia:

- 1 model,
- 1 jezyk,
- 2 kategorie,
- 2 emocje,
- `top_k = 5`,
- male listy kandydatow.

## Pelny eksperyment

1. Ustaw `enabled: true` dla wybranych modeli w `config.yaml`.
2. Uzupelnij pelna liste emocji i kandydatow.
3. Uruchom:

```powershell
python experiment.py --config config.yaml --top_k 10
python analysis.py --results outputs/results_full.csv
```

## Porownanie trzech modeli: Bielik, Gemini, Qwen

Gotowa konfiguracja jest w `config_3models.yaml`. Wlaczone sa tylko:

- `bielik_1_5b_v3`
- `gemini`
- `qwen3_embedding_8b`

Domyslnie uzywa jednej glownej strategii:

```text
centroid_phrases + neutral_centroid
```

Uruchomienie:

```bash
source ~/.secrets/hf_gemini.env
python experiment.py --config config_3models.yaml --top_k 10
python analysis.py --results outputs/three_models/results_full.csv
```

Mozesz tez wymusic modele z CLI:

```bash
python experiment.py --config config_3models.yaml --models bielik_1_5b_v3,gemini,qwen3_embedding_8b
```

## Pliki wynikowe

Eksperyment zapisuje:

- `outputs/results_full.csv`
- `outputs/results_full.xlsx`
- `outputs/manual_rating_template.xlsx`

Analiza zapisuje:

- `outputs/summary_by_strategy.csv`
- `outputs/summary_by_model.csv`
- `outputs/strategy_overlap_top10.csv`

Wyniki zawieraja m.in. `rank`, `cosine_similarity`, `same_category`, `candidate_identical_to_category`, `candidate_contains_emotion`, `average_similarity_top_k` i `category_retention_at_k`.
