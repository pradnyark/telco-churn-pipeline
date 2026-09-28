"""
preprocess_eda.py
------------------
Covers Assignment sub-activities:
  1.3 Data Pre-processing
  1.4 Exploratory Data Analysis (EDA)

Designed to run either:
  - Locally for testing, OR
  - As an AWS Glue Job / Lambda function (see notes at bottom)

Usage (local):
    python preprocess_eda.py --input data/telco_churn.csv --target Churn --outdir output/

Dependencies:
    pip install pandas numpy scikit-learn matplotlib seaborn boto3
"""

import argparse
import logging
import os
import sys
import json
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for Lambda/Glue
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging setup (CloudWatch will capture stdout/stderr automatically when
# this runs inside Lambda or Glue -- no extra config needed there)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pipeline")


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded shape: {df.shape}")
    return df


def summary_statistics(df: pd.DataFrame) -> dict:
    logger.info("Computing summary statistics")
    desc = df.describe(include="all").to_dict()
    dtypes = df.dtypes.astype(str).to_dict()
    logger.info(f"Data types: {dtypes}")
    return {"describe": desc, "dtypes": dtypes}


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    logger.info(f"Missing value counts:\n{missing}")
    return missing


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Imputed '{col}' missing values with median={median_val}")
    return df


def normalize_numeric(df: pd.DataFrame, exclude: list) -> pd.DataFrame:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if numeric_cols:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        logger.info(f"Normalized columns: {numeric_cols}")
    return df


def encode_categoricals(df: pd.DataFrame, exclude: list) -> pd.DataFrame:
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in exclude]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = list(le.classes_)
        logger.info(f"Encoded '{col}' -> {len(le.classes_)} classes")
    return df, encoders


def bin_numeric_column(df: pd.DataFrame, col: str, bins: int = 4) -> pd.DataFrame:
    if col in df.columns:
        new_col = f"{col}_binned"
        df[new_col] = pd.cut(df[col], bins=bins, labels=False)
        logger.info(f"Created binned column '{new_col}' from '{col}'")
    return df


def correlation_analysis(df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    path = os.path.join(outdir, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved correlation heatmap to {path}")
    return corr


def feature_importance(df: pd.DataFrame, target: str, outdir: str):
    X = df.drop(columns=[target])
    y = df[target]
    X = X.select_dtypes(include=[np.number])  # assumes categoricals already encoded

    if y.dtype == "object" or y.nunique() <= 10:
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        y_enc = y
        model = RandomForestRegressor(n_estimators=200, random_state=42)

    model.fit(X, y_enc)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    importances.head(15).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importances")
    plt.tight_layout()
    path = os.path.join(outdir, "feature_importance.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved feature importance chart to {path}")
    return importances


def univariate_bivariate_plots(df: pd.DataFrame, target: str, outdir: str):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target][:4]  # limit for brevity

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"univariate_{col}.png"))
        plt.close()

    if target in df.columns:
        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=df[target], y=df[col])
            plt.title(f"{col} vs {target}")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"bivariate_{col}_vs_{target}.png"))
            plt.close()

    logger.info(f"Saved univariate/bivariate plots for columns: {numeric_cols}")


def push_to_cloud(outdir: str, run_log: dict, s3_bucket: str = None, push_metrics: bool = False):
    """
    Optional: sync local outputs to S3 and push a CloudWatch custom metric.
    Runs only if boto3 is installed AND AWS credentials are available locally
    (e.g. from `aws configure` using temporary SSO keys). Fails quietly with
    a log warning if credentials aren't set up -- never breaks the local run.
    """
    if not BOTO3_AVAILABLE:
        logger.warning("boto3 not installed -- skipping cloud sync (pip install boto3 to enable)")
        return

    try:
        if s3_bucket:
            s3 = boto3.client("s3")
            for fname in os.listdir(outdir):
                local_path = os.path.join(outdir, fname)
                if os.path.isfile(local_path):
                    s3.upload_file(local_path, s3_bucket, f"output/{fname}")
            logger.info(f"Synced {outdir} contents to s3://{s3_bucket}/output/")

        if push_metrics:
            cw = boto3.client("cloudwatch")
            cw.put_metric_data(
                Namespace="DataPipeline",
                MetricData=[
                    {"MetricName": "RowsProcessed", "Value": float(run_log["input_rows"])},
                    {"MetricName": "MissingValueColumns", "Value": float(len(run_log["missing_value_columns"]))},
                ],
            )
            logger.info("Pushed custom metrics to CloudWatch namespace 'DataPipeline'")

    except Exception as e:
        logger.warning(f"Cloud sync skipped (credentials not configured or expired): {e}")


def run_pipeline(input_path: str, target: str, outdir: str, s3_bucket: str = None, push_metrics: bool = False):
    os.makedirs(outdir, exist_ok=True)
    run_start = datetime.utcnow().isoformat()
    logger.info(f"=== Pipeline run started at {run_start} ===")

    df = load_data(input_path)
    # Drop ID-like columns before analysis -- they carry no analytical signal
    id_cols = [c for c in df.columns if "id" in c.lower() and df[c].nunique() == len(df)]
    if id_cols:
        logger.info(f"Dropping ID-like columns before analysis: {id_cols}")
        df = df.drop(columns=id_cols)
    stats = summary_statistics(df)
    missing = check_missing_values(df)
    df = impute_missing(df)
    df, encoders = encode_categoricals(df, exclude=[target])
    df = normalize_numeric(df, exclude=[target])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df = bin_numeric_column(df, numeric_cols[0])

    corr = correlation_analysis(df, outdir)
    importances = feature_importance(df, target, outdir)
    univariate_bivariate_plots(df, target, outdir)

    processed_path = os.path.join(outdir, "processed_data.csv")
    df.to_csv(processed_path, index=False)

    run_log = {
        "run_start": run_start,
        "run_end": datetime.utcnow().isoformat(),
        "input_rows": len(df),
        "missing_value_columns": missing.to_dict(),
        "top_features": importances.head(5).to_dict(),
        "processed_file": processed_path,
    }
    log_path = os.path.join(outdir, f"run_log_{run_start.replace(':', '-')}.json")
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2, default=str)

    logger.info(f"=== Pipeline run complete. Log saved to {log_path} ===")

    push_to_cloud(outdir, run_log, s3_bucket=s3_bucket, push_metrics=push_metrics)

    return run_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV (local path or S3-downloaded file)")
    parser.add_argument("--target", required=True, help="Name of the target/label column")
    parser.add_argument("--outdir", default="output", help="Directory to write outputs/plots/logs")
    parser.add_argument("--s3-bucket", default=None, help="Optional: S3 bucket to sync outputs to (requires AWS credentials configured locally)")
    parser.add_argument("--push-metrics", action="store_true", help="Optional: push run stats to CloudWatch as custom metrics (requires AWS credentials)")
    args = parser.parse_args()

    try:
        run_pipeline(args.input, args.target, args.outdir, s3_bucket=args.s3_bucket, push_metrics=args.push_metrics)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Notes for deploying on AWS:
#
# AWS Lambda:
#   - Zip this script + dependencies (or use a Lambda layer for pandas/sklearn).
#   - Read input CSV from S3 using boto3 (s3.download_file(bucket, key, "/tmp/input.csv"))
#     since Lambda's only writable directory is /tmp.
#   - Write outputs back to S3 (s3.upload_file(...)) instead of local outdir.
#
# AWS Glue:
#   - Glue has pandas/sklearn available via a Python shell job or a Glue job
#     with a custom wheel/dependency bundle.
#   - Point --input at the S3 path directly (Glue jobs can read s3:// paths).
#
# Scheduling every 2 minutes:
#   - Create an EventBridge (CloudWatch Events) rule with schedule
#     expression: rate(2 minutes)
#   - Target: your Lambda function ARN or a Step Functions state machine
#     that chains this script with any downstream steps.
#
# Logging:
#   - Lambda/Glue automatically stream stdout/stderr (i.e. these logger
#     calls) to CloudWatch Logs -- no extra setup required.
# ---------------------------------------------------------------------------
