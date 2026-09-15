"""Reproducible training entry point with MLflow tracking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data import load_tweet_eval, validate_dataset
from src.model import build_model

MODEL_NAME = "sentiment-classifier"


def train(limit: int | None = None) -> dict[str, float]:
    train_df = load_tweet_eval("train", limit)
    test_df = load_tweet_eval("test", limit)
    validate_dataset(train_df)
    validate_dataset(test_df)
    model = build_model()
    model.fit(train_df["text"], train_df["label"])
    pred = model.predict(test_df["text"])
    metrics = {
        "accuracy": float(accuracy_score(test_df["label"], pred)),
        "macro_f1": float(f1_score(test_df["label"], pred, average="macro")),
        "macro_precision": float(precision_score(test_df["label"], pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(test_df["label"], pred, average="macro", zero_division=0)),
    }
    mlflow.set_experiment("sentiment-intelligence")
    with mlflow.start_run(run_name="tfidf-logistic-regression") as run:
        mlflow.log_params({
            "model": "logistic_regression",
            "features": "tfidf",
            "ngram_range": "1-2",
            "train_limit": limit or "full",
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model", registered_model_name=MODEL_NAME)
        mlflow.set_tag("selection_metric", "macro_f1")
        mlflow.set_tag("run_id_for_registry", run.info.run_id)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Optional sample size for local development")
    args = parser.parse_args()
    print(json.dumps(train(args.limit), indent=2))


if __name__ == "__main__":
    main()
