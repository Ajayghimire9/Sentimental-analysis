"""Deterministic text normalization used by both training and inference."""
from __future__ import annotations

import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize a social-media message without leaking label information."""
    text = str(text)
    text = _URL.sub(" URL ", text)
    text = _MENTION.sub(" USER ", text)
    text = text.replace("#", " ")
    text = re.sub(r"[^\w\s!?]", " ", text, flags=re.UNICODE)
    text = _WHITESPACE.sub(" ", text).strip().lower()
    return text
