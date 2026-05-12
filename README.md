# Stock Movement Prediction Web App

Predicts short-term Up/Down for stocks (AI-Lab Project | UCP Spring 2026)

- Neural Network (MLP) and K-Nearest Neighbors models
- Streamlit Web UI
- Interactive history + DB logging
- Visual results, confusion matrix

## How to run
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Project Structure

- `app/streamlit_app.py` — Streamlit code (UI)
- `src/data.py`          — Data fetch utilities
- `src/features.py`      — Feature engineering/labels
- `src/models/`          — Model code (KNN, MLP)
- `src/evaluate.py`      — Evaluation, confusion matrix
- `src/db.py`            — SQLite run logger

## Term Project Submission (Spring 2026)
- Report, dataset, models, guide, code: see `/reports`, `/data`, `/assets/screenshots`

---
Project by: Malik-Faisal-Awan1 | UCP IT
