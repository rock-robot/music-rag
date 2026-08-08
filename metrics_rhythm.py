"""metrics_rhythm.py — duration histogram for the Week 4 style comparison.

METRIC_EDGES were chosen from CORPUS percentiles only, before any A'/B
comparison was computed, and are frozen. See decisions log.
"""
import numpy as np
from collections import Counter
from features import DURATION_EDGES
from metrics_core import aggregate_pooled

METRIC_EDGES = [0.05, 0.10, 0.15, 0.30, 0.45, 0.70, 1.20, 2.50]
N_BINS = len(METRIC_EDGES) + 1                      # 9

assert set(DURATION_EDGES) <= set(METRIC_EDGES), \
    "metric edges must REFINE the retrieval edges, never move them"


def _representatives(edges):
    """One point strictly inside each bin -- never on an edge (ties are ambiguous)."""
    return [edges[0] / 2] + [(a + b) / 2 for a, b in zip(edges, edges[1:])] + [edges[-1] * 2]


# fine bin index -> coarse bin index. Derived, not hand-typed, so it cannot drift.
COARSE_OF = [int(np.searchsorted(DURATION_EDGES, p)) for p in _representatives(METRIC_EDGES)]


def duration_counts(notes, edges=METRIC_EDGES):
    """Note durations bucketed by `edges`. Empty input -> empty Counter (no raise)."""
    c = Counter()
    for n in notes:
        c[int(np.searchsorted(edges, n.end - n.start))] += 1
    return c


def to_coarse(hist_fine):
    """9-bin histogram -> the frozen 7-bin retrieval scheme. Exact, by construction."""
    out = np.zeros(len(DURATION_EDGES) + 1)
    for i, v in enumerate(hist_fine):
        out[COARSE_OF[i]] += v
    return out


def bin_labels(edges=METRIC_EDGES):
    return ([f"<{edges[0]:.2f}"]
            + [f"{a:.2f}-{b:.2f}" for a, b in zip(edges, edges[1:])]
            + [f">{edges[-1]:.2f}"])
