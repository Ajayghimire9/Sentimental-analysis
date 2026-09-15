"""Trainable TF-IDF sentiment classifier."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.text import normalize_text


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=normalize_text,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=100_000,
        )),
        ("classifier", LogisticRegression(
            C=2.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
