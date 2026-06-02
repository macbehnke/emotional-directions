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
- `mmlw_roberta_large` - `sdadas/mmlw-roberta-large`, polski model embeddingowy porownywalny z wyspecjalizowanymi encoderami.
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

## Porownanie modeli: Bielik, Gemini, Qwen, MMLW-RoBERTa

Gotowa konfiguracja jest w `config_3models.yaml`. Wlaczone sa tylko:

- `bielik_1_5b_v3`
- `gemini`
- `qwen3_embedding_8b`
- `mmlw_roberta_large`

Domyslnie uzywa jednej glownej strategii:

```text
centroid_phrases + neutral_centroid
```

Projekt koncentruje sie na emocjach i afektywnych etykietach takich jak
`sadness`, `disgust`, `anger`, `fear`, `joy`, `amusement`, `excitement`,
`love`, `positive` i `negative`, a nie na wymiarach VAD jako glownym celu.
`positive` i `negative` sa szerokimi biegunami afektywnymi, wiec warto
raportowac je osobno od bardziej dyskretnych emocji.
Jesli dodasz zewnetrzne leksykony emocji, powinny miec etykiety dyskretnych
emocji, np. `candidate, emotion_label, emotion_intensity`.

Konfiguracja zawiera tez warunki kontrolne:

- `emotion` - wlasciwy kierunek emocji,
- `identity` - kontrola `category + neutral - neutral`,
- `random` - losowy, deterministyczny kierunek,
- `shuffled_emotion` - kierunek innej emocji przypisany do tej samej kategorii.

Wyniki zawieraja dodatkowo `projection_on_emotion_direction`, czyli projekcje
kandydata na badany kierunek emocji. Bootstrap stabilnosci zapisuje sie do:

```text
outputs/three_models/bootstrap_stability.csv
outputs/three_models/bootstrap_stability.xlsx
```

Uruchomienie:

```bash
source ~/.secrets/hf_gemini.env
python experiment.py --config config_3models.yaml --top_k 10
python analysis.py --results outputs/three_models/results_full.csv
```

Mozesz tez wymusic modele z CLI:

```bash
python experiment.py --config config_3models.yaml --models bielik_1_5b_v3,gemini,qwen3_embedding_8b,mmlw_roberta_large
```

## Final analysis run

Po pilocie glowna konfiguracja finalna jest w `config_final.yaml`.
Uzywa:

- `centroid_phrases + neutral_centroid`,
- warunkow kontrolnych `emotion`, `identity`, `random`, `shuffled_emotion`,
- filtrowania kandydatow identycznych z kategoria albo seedami emocji,
- modeli `gemini`, `qwen3_embedding_8b`, `bielik_1_5b_v3` i `mmlw_roberta_large`.

Uruchomienie na klastrze:

```bash
cd /projects/laigai/emo_dir/emotional-directions
source .venv/bin/activate

export HF_HOME=/projects/laigai/hf_cache
export HUGGINGFACE_HUB_CACHE=/projects/laigai/hf_cache/hub
export TRANSFORMERS_CACHE=/projects/laigai/hf_cache/transformers
export TORCH_HOME=/work/s152265/torch_cache
export TMPDIR=/work/s152265/tmp
export TEMP=/work/s152265/tmp
export TMP=/work/s152265/tmp
export PYTHONUTF8=1
export HF_HUB_DISABLE_XET=1

source ~/.secrets/hf_gemini.env

python experiment.py --config config_final.yaml --top_k 10
python analysis.py --results outputs/final_run/results_full.csv
```

Jesli chcesz ograniczyc finalny bieg do trzech modeli bez MMLW-RoBERTa:

```bash
python experiment.py --config config_final.yaml --top_k 10 --models bielik_1_5b_v3,gemini,qwen3_embedding_8b
python analysis.py --results outputs/final_run/results_full.csv
```

Dodatkowy check strategii dyskretnych jest w
`config_supplement_discrete_check.yaml`. Porownuje finalne
`centroid_phrases` z `single_word`:

```bash
python experiment.py --config config_supplement_discrete_check.yaml --top_k 10
python analysis.py --results outputs/supplement_discrete_check/results_full.csv
```

Analiza zapisuje dodatkowe pliki:

- `final_model_report.csv` - zwiezle porownanie modeli dla finalnej strategii,
- `final_strategy_report.csv` - zwiezle porownanie strategii, szczegolnie dla supplementary run,
- `summary_control_deltas.csv` - roznice `emotion` wzgledem kontroli.

## Materialy na prezentacje w klasie

Gotowe materialy sa w:

```text
presentation/emotional_directions_class_presentation.pptx
class_app/index.html
```

Uruchomienie lokalnej aplikacji:

```bash
cd class_app
python -m http.server 8765 --bind 127.0.0.1
```

Nastepnie otworz:

```text
http://127.0.0.1:8765/index.html
```

Aplikacja jest statyczna i korzysta z `class_app/data.js`, wiec nie uruchamia
modeli i nie wymaga tokenow API.

## Pliki wynikowe

Eksperyment zapisuje:

- `outputs/results_full.csv`
- `outputs/results_full.xlsx`
- `outputs/manual_rating_template.xlsx`

Analiza zapisuje:

- `outputs/summary_by_strategy.csv`
- `outputs/summary_by_model.csv`
- `outputs/strategy_overlap_top10.csv`
- `outputs/model_output_similarity_top10.csv`

Wyniki zawieraja m.in. `rank`, `cosine_similarity`, `same_category`, `candidate_identical_to_category`, `candidate_contains_emotion`, `average_similarity_top_k` i `category_retention_at_k`.
