"""Dataset loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset

LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def load_tweet_eval(split: str = "train", limit: int | None = None) -> pd.DataFrame:
    dataset = load_dataset("tweet_eval", "sentiment", split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    frame = dataset.to_pandas()[["text", "label"]]
    frame["label_name"] = frame["label"].map(LABELS)
    return frame


def validate_dataset(frame: pd.DataFrame) -> None:
    required = {"text", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame["text"].isna().any() or frame["label"].isna().any():
        raise ValueError("Dataset contains null text or labels")
    if not set(frame["label"].unique()).issubset(LABELS):
        raise ValueError("Unexpected sentiment label")


def write_validation_marker(path: str = "reports/data_validation.marker") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sample = load_tweet_eval("train", 1000)
    validate_dataset(sample)
    Path(path).write_text("validated=true\nrows=1000\n", encoding="utf-8")


if __name__ == "__main__":
    write_validation_marker()
