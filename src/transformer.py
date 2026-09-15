"""Optional transformer benchmark interface.

Kept separate from the lightweight production baseline so CPU-only development
remains fast. Requires: transformers, torch.
"""

from __future__ import annotations


def build_transformer(model_name: str = "distilbert-base-uncased"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    return tokenizer, model
