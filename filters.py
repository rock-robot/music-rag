"""filters.py — the collapse criterion, and a distance immune to it.

Two degeneracy modes, caught by a UNION:
  static repetition  -> zero_frac    (misses two-pitch loops)
  two-pitch loop     -> distinct < 3 (misses 96%-repeat runs among 4-6 pitches)

Because the filter uses zero_frac, the interval-0 bin is CONDITIONED. The
headline distance therefore drops bin 0 and renormalizes: P(iv | iv != 0).
"""
import numpy as np
from features import INTERVAL_BINS

MAX_NOTES, MAX_ZERO_FRAC, MIN_DISTINCT = 200, 0.40, 3
ZERO_IDX = INTERVAL_BINS.index(0)


def is_collapsed(s):
    """s = stats dict from reclassify.stats(). True if EITHER mode fires."""
    return (s["n"] > MAX_NOTES
            or s["zero_frac"] > MAX_ZERO_FRAC     # static repetition
            or s["distinct"] < MIN_DISTINCT)      # two-pitch oscillation


def drop_zero(hist):
    """Interval histogram -> conditional distribution excluding bin 0."""
    h = np.delete(np.asarray(hist, dtype=float), ZERO_IDX)
    total = h.sum()
    return h / total if total > 0 else h


def tv_nonzero(p, q):
    """TV distance over non-zero intervals only. Immune to the filter."""
    a, b = drop_zero(p), drop_zero(q)
    return 0.5 * float(np.abs(a - b).sum())

def stats(notes):
    """Everything a collapse criterion needs, computed once per generation."""
    p = [n.pitch for n in notes]
    zeros = sum(1 for a, b in zip(p, p[1:]) if a == b)
    return {"n": len(p),
            "distinct": len(set(p)),
            "zero_frac": zeros / (len(p) - 1) if len(p) > 1 else 0.0}