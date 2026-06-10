# Emotional Directions Explorer

Static classroom app for exploring the final run.

Run locally:

```bash
cd class_app
python -m http.server 8765 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8765/index.html
```

The app uses `data.js`, generated from `outputs/three_models/results_full.csv`
and the final summary reports. It does not call any model or API.

Regenerate after a new result run:

```bash
python analysis.py --results outputs/three_models/results_full.csv
python class_app/build_data.py --results outputs/three_models/results_full.csv
```

Main views:

- EN vs PL comparison for the selected model.
- Model comparison within the selected language.
- Exact candidate overlaps that appear across at least two model top-10 lists.
