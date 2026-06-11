# NotebookLM source pack

Upload this folder to NotebookLM for a simple presentation about the project.

Recommended upload priority:

1. `README.md`
2. `config_final.yaml`
3. `config_3models.yaml`
4. `experiment.py`
5. `analysis.py`
6. `embedders/gemini_embedder.py`
7. `embedders/qwen_embedder.py`
8. `embedders/bielik_embedder.py`
9. Files in `results_final_run/`, especially:
   - `final_model_report.csv`
   - `summary_by_model.csv`
   - `model_output_similarity_top10.csv`
   - `results_full_sample_300_rows.csv`

Presentation prompt:

```text
Create a simple 8-slide presentation about my project "Emotional Directions in Embedding Spaces".

Audience: university class / NLP and deep learning course.
Tone: clear, simple, not overly technical.
Goal: explain what the project does, how the experiment works, what models were compared, and what the main findings mean.

Use this slide structure:

1. Title: project title and one-sentence research question.
2. Motivation: why emotions in embedding spaces are interesting and why comparing languages/models matters.
3. Main idea: emotional direction = emotion vector minus neutral vector; candidates are ranked in embedding space.
4. Data and inputs: languages, emotions, neutral terms, and candidate categories.
5. Models: Gemini embedding model, Qwen3-Embedding-8B, and Bielik hidden-state pooling; state that Bielik is not a dedicated embedding model.
6. Method: embed terms and candidates, build emotion direction, rank candidates, compare top-k outputs.
7. Results: summarize the strongest findings from the CSV results and include 2-3 concrete top-candidate examples.
8. Limitations and conclusion: small candidate lists, hidden-state pooling is experimental, similarity is not proof of human emotion understanding, final takeaway.

Please write concise slide bullets, speaker notes for each slide, and suggest one simple visual per slide.
Avoid unsupported claims; if the data does not prove something, phrase it cautiously.
```
