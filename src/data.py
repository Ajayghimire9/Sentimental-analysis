"""Dataset loading and schema validation."""
from __future__ import annotations

from datasets import load_dataset
import pandas as pd

LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def load_tweet_eval(split: str = "train", limit: int | None = None) -> pd.DataFrame:
    """Load the public TweetEval sentiment benchmark without committing data to Git."""
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
    if frame["text"].isna().any():
        raise ValueError("Text contains null values")
    if frame["label"].isna().any():
        raise ValueError("Labels contain null values")
