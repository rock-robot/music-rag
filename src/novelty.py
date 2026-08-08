"""novelty.py — transposition-aware interval n-gram overlap. STAGE 1: calibration.

Feature = sequences of consecutive melodic INTERVALS. Transposition-invariant by
construction: transposing a phrase leaves every interval unchanged, so a
transposed copy scores as a copy. (This is the same principle the Week 1 de-dupe
gate and the augmentation rationale both rest on.)
"""
import random
import numpy as np
from pathlib import Path
from features import load_notes, window, piece_name

MAX_GAP = 2.0        # seconds; a longer silence means the two notes aren't adjacent
NS      = (3, 4, 5, 6, 7, 8)


def interval_runs(notes, max_gap=MAX_GAP):
    """Notes -> list of interval sequences, BROKEN at rests longer than max_gap.

    Without the break, a flute resting 20 bars would emit one spurious interval
    spanning the rest (our corpus IOI 99th percentile is 23s -- this is real)."""
    runs, cur = [], []
    for a, b in zip(notes, notes[1:]):
        if b.start - a.start > max_gap:
            if cur:
                runs.append(tuple(cur))
            cur = []
            continue
        cur.append(b.pitch - a.pitch)          # NOT clamped -- see deconstruction
    if cur:
        runs.append(tuple(cur))
    return runs


def load_runs(path, seconds=None):
    notes = load_notes(path)
    if seconds is not None:
        notes = window(notes, seconds)
    return interval_runs(notes)


def ngrams(runs, n):
    return [run[i:i + n] for run in runs for i in range(len(run) - n + 1)]


def shuffle_runs(runs, rng):
    """Keep the interval multiset, destroy the order. The chance floor."""
    flat = [iv for run in runs for iv in run]
    rng.shuffle(flat)
    out, i = [], 0
    for run in runs:                            # preserve run-length structure
        out.append(tuple(flat[i:i + len(run)]))
        i += len(run)
    return out


def index_paths(paths, ns=NS, loader=load_runs, key=piece_name):
    """Generalizes build_reference: any path set, any loader, any provenance key."""
    idx = {n: {} for n in ns}
    for p in paths:
        runs, src = loader(p), key(p)
        for n in ns:
            for g in ngrams(runs, n):
                idx[n].setdefault(g, set()).add(src)
    return idx


def coverage(runs, idx, n):
    """Fraction of this melody's n-grams that appear ANYWHERE in the reference.
    Returns None if there are no n-grams (too short) -- excluded, not scored 0."""
    grams = ngrams(runs, n)
    if not grams:
        return None
    return sum(1 for g in grams if g in idx[n]) / len(grams)


def curve(list_of_runs, idx, ns=NS, label=""):
    """Mean coverage at each n, over a population of melodies."""
    out = {}
    for n in ns:
        vals = [c for runs in list_of_runs if (c := coverage(runs, idx, n)) is not None]
        out[n] = (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)
    if label:
        print(f"  {label:22s} " +
              " ".join(f"n={n}:{out[n][0]:.3f}" for n in ns) +
              f"   ({out[ns[0]][1]} items)")
    return out


if __name__ == "__main__":
    train = sorted(Path("data/train").glob("*_t+00.mid"))
    val   = sorted(Path("data/val").glob("*_t+00.mid"))
    print(f"reference: {len(train)} train chunks | baseline: {len(val)} val chunks")

    idx = build_reference(train)
    for n in NS:
        print(f"  n={n}: {len(idx[n]):>7d} distinct n-grams")

    val_runs = [load_runs(p, seconds=10.0) for p in val]     # length-matched to generations
    rng = random.Random(42)

    print("\ncoverage of the TRAIN set (mean fraction of n-grams found):")
    curve(val_runs, idx, label="val (real, novel)")
    curve([shuffle_runs(r, rng) for r in val_runs], idx, label="val shuffled (floor)")