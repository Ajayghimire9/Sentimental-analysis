"""Lightweight population-stability monitoring for text length and sentiment labels."""

from __future__ import annotations

import numpy as np


def population_stability_index(
    reference: list[float], current: list[float], bins: int = 10
) -> float:
    """Compute PSI; values above ~0.2 can be investigated as meaningful drift."""
    if not reference or not current:
        raise ValueError("Both populations must contain observations")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref = np.maximum(ref_hist / len(reference), 1e-6)
    cur = np.maximum(cur_hist / len(current), 1e-6)
    return float(np.sum((cur - ref) * np.log(cur / ref)))
