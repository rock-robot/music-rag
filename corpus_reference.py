"""corpus_reference.py — length- and aggregation-matched corpus references.

Two axes are mismatched between corpus and generations:
  LENGTH       whole melodies (minutes) vs 10s clips
  AGGREGATION  pooled note-weighted vs per-unit averaged
This builds the reference all four ways so the axes can be separated.
"""
import numpy as np
from pathlib import Path
from collections import Counter

import metrics_rhythm
from features import load_notes, INTERVAL_BINS
from metrics_core import to_hist, tv_distance, safe
import features

EXCERPT_SECONDS   = 10.0
MIN_EXCERPT_NOTES = 2          # matches the generations' own floor (<2 = "empty")


def excerpts(notes, length=EXCERPT_SECONDS, hop=None, min_notes=MIN_EXCERPT_NOTES):
    """Tile a melody into consecutive `length`-second excerpts.

    Membership is by ONSET only; durations are NEVER clipped. This deliberately
    does not use window(): window() caps sustains because a prompt's time
    footprint is a hard constraint. Here the goal is to mirror a generation, and
    generations' durations aren't clipped -- a note starting at 9.5s keeps its
    full length. Same-shaped operation, opposite correct choice.
    """
    hop = length if hop is None else hop
    if not notes:
        return []
    t, last, out = notes[0].start, max(n.start for n in notes), []
    while t <= last:
        sel = [n for n in notes if t <= n.start < t + length]
        if len(sel) >= min_notes:
            out.append(sel)
        t += hop
    return out


def pooled(units, counts_fn, bins):
    """Note-weighted: pool counts, normalize once. (What the corpus uses now.)"""
    total = Counter()
    for notes in units:
        total.update(counts_fn(notes))
    return to_hist(total, bins)


def averaged(units, counts_fn, bins):
    """Unit-weighted: normalize each unit, then average.
    Mirrors aggregate_per_seed -- how the generations are aggregated."""
    hs = [h for notes in units
          if (h := to_hist(counts_fn(notes), bins)).sum() > 0]
    return np.mean(hs, axis=0) if hs else np.zeros(len(bins))


def build_references(counts_fn, bins, corpus_dir="corpus"):
    melodies = [load_notes(p) for p in sorted(Path(corpus_dir).glob("*.mid"))]
    exc = [e for m in melodies for e in excerpts(m)]

    # correctness check: exhaustive tiling must preserve the note multiset exactly
    exc_all = [e for m in melodies for e in excerpts(m, min_notes=1)]
    assert sum(len(e) for e in exc_all) == sum(len(m) for m in melodies), \
        "tiling lost or duplicated notes"
    assert np.allclose(pooled(melodies, counts_fn, bins),
                       pooled(exc_all, counts_fn, bins)), \
        "pooled histogram changed under exhaustive tiling -- excerpting is wrong"

    print(f"  {len(melodies)} melodies -> {len(exc)} excerpts "
          f"(>={MIN_EXCERPT_NOTES} notes, {EXCERPT_SECONDS:.0f}s, non-overlapping)")
    return {"whole_pooled":   pooled(melodies, counts_fn, bins),
            "excerpt_pooled": pooled(exc, counts_fn, bins),
            "whole_avg":      averaged(melodies, counts_fn, bins),
            "excerpt_avg":    averaged(exc, counts_fn, bins)}


if __name__ == "__main__":
    bins = list(range(metrics_rhythm.N_BINS))
    labels = metrics_rhythm.bin_labels()
    refs = build_references(metrics_rhythm.duration_counts, bins)

    print(f"\n  {'bin':>10} " + " ".join(f"{k:>14s}" for k in refs))
    for i, lab in enumerate(labels):
        print(f"  {lab:>10} " + " ".join(f"{refs[k][i]:14.3f}" for k in refs))

    print("\n  TV distance between references:")
    keys = list(refs)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            print(f"    {a:15s} vs {b:15s} {tv_distance(refs[a], refs[b]):.4f}")