"""metrics_core.py — Week 4 measurement machinery.

Features live in features.py (single source of truth). This module adds only:
  fail-soft wrapping, normalization to fixed bins, and the two aggregation policies.
"""
import numpy as np
from collections import Counter
import features
from features import load_notes


def pitch_class_counts(notes):
    """Tonal content: pitch mod 12. The one feature features.py doesn't have."""
    return Counter(n.pitch % 12 for n in notes)


def safe(counts_fn):
    """features.py raises on degenerate input (right for retrieval: stop the program).
    Metrics must not die on 1 bad file in 226 -- return an empty Counter instead."""
    def wrapped(notes):
        try:
            return counts_fn(notes)
        except ValueError:
            return Counter()
    wrapped.__name__ = f"safe_{counts_fn.__name__}"
    return wrapped


def to_hist(counts, bins):
    """Counter -> normalized array over an explicit bin order. Empty -> zeros."""
    v = np.array([counts.get(b, 0) for b in bins], dtype=float)
    total = v.sum()
    return v / total if total > 0 else v


def tv_distance(p, q):
    """Total variation distance in [0,1]. 0 = identical, 1 = disjoint support."""
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def aggregate_pooled(paths, counts_fn, bins, loader=load_notes):
    """Pool COUNTS, normalize once -> note-weighted. For the CORPUS reference."""
    total, used, skipped = Counter(), 0, []
    for p in paths:
        try:
            notes = loader(p)
        except Exception as e:
            skipped.append((str(p), repr(e)))
            continue
        c = counts_fn(notes)
        if not c:
            skipped.append((str(p), "no features"))
            continue
        total.update(c)
        used += 1
    return to_hist(total, bins), used, skipped


def aggregate_per_seed(recs, counts_fn, bins, loader=load_notes):
    """Normalize each generation, mean within seed, mean across seeds.
    Every SEED gets one vote regardless of note count or surviving samples."""
    by_seed, dropped = {}, []
    for r in recs:
        try:
            notes = loader(r["path"])
        except Exception as e:
            dropped.append((str(r["path"]), repr(e)))
            continue
        h = to_hist(counts_fn(notes), bins)
        if h.sum() == 0:
            dropped.append((str(r["path"]), "no features"))
            continue
        by_seed.setdefault(r["seed"], []).append(h)
    seed_means = {s: np.mean(hs, axis=0) for s, hs in by_seed.items()}
    grand = (np.mean(list(seed_means.values()), axis=0) if seed_means
             else np.zeros(len(bins)))
    return grand, seed_means, dropped