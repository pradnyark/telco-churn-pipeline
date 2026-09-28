# Telco Churn Data Pipeline (Group 9)

- `preprocess_eda.py` – preprocessing + EDA pipeline
- `.github/workflows/pipeline.yml` – GitHub Actions schedule (runs the pipeline every ~2 minutes, commits results to `output/`)
- `streamlit_app.py` – cloud dashboard (Streamlit Community Cloud)
- `app.py` – Flask API (run locally, test with Postman)
- `data/` – put `WA_Fn-UseC_-Telco-Customer-Churn.csv` here
