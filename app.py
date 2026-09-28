"""
app.py - Flask API for the Telco Churn Data Pipeline

Exposes application/pipeline details as REST endpoints, satisfying
Sub-Objective 2 (API Access). Run this locally, then test each
endpoint in Postman.

Run with:
    pip install flask
    python app.py

Then open http://127.0.0.1:5000/ or test endpoints in Postman at
that same base URL.
"""

import os
import json
import glob
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)

# Folder where preprocess_eda.py writes its output (adjust if different)
OUTPUT_DIR = "output"


def get_latest_run_log():
    log_files = glob.glob(os.path.join(OUTPUT_DIR, "run_log_*.json"))
    if not log_files:
        return None
    latest = max(log_files, key=os.path.getmtime)
    with open(latest) as f:
        return json.load(f)


def get_all_run_logs():
    log_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "run_log_*.json")))
    logs = []
    for lf in log_files:
        with open(lf) as f:
            logs.append(json.load(f))
    return logs


@app.route("/")
def home():
    return jsonify({
        "app": "Telco Churn Pipeline API",
        "status": "running",
        "endpoints": ["/health", "/latest-run", "/run-history", "/dataset-info"]
    })


@app.route("/health")
def health():
    """Detail 1: Application health/deployment status"""
    return jsonify({
        "status": "healthy",
        "checked_at": datetime.utcnow().isoformat(),
        "output_dir_exists": os.path.isdir(OUTPUT_DIR),
    })


@app.route("/latest-run")
def latest_run():
    """Detail 2: Most recent pipeline run details"""
    log = get_latest_run_log()
    if log is None:
        return jsonify({"error": "No runs found yet. Run preprocess_eda.py first."}), 404
    return jsonify(log)


@app.route("/run-history")
def run_history():
    """Detail 3: Full history of pipeline runs (proves the 2-minute scheduling)"""
    logs = get_all_run_logs()
    return jsonify({
        "total_runs": len(logs),
        "runs": logs
    })


@app.route("/dataset-info")
def dataset_info():
    """Detail 4: Static info about the dataset/business problem"""
    return jsonify({
        "dataset": "Telco Customer Churn",
        "source": "Kaggle - blastchar/telco-customer-churn",
        "target_column": "Churn",
        "business_problem": "Predict which telecom customers are likely to churn",
        "rows_expected": 7043
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
