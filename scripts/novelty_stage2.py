"""novelty_stage2.py — two-sided novelty for A / A' / B.

Side 1 (memorization): generation n-grams found in the TRAIN set.
Side 2 (regurgitation): generation n-grams found in ITS OWN prompt phrases.
Both measured in the same direction (fraction of the GENERATION that is matched),
so the numbers are comparable.
"""
import random
import numpy as np
from pathlib import Path
from collections import Counter

from features import load_notes, window, piece_name
from condition import truncate
from novelty import interval_runs, ngrams, coverage, index_paths, load_runs, shuffle_runs
from results import load_results, SYSTEMS
from reclassify import stats
from filters import is_collapsed

NS       = (5, 8)
OUTLIER  = "el_mar_idx04_melody_chunk05_t+00.mid"
TRAIN    = Path("data/train")


def clean(sysname):
    """Generations surviving the UNION collapse filter, with runs attached."""
    out = []
    for r in load_results(sysname, ok_only=False):
        notes = load_notes(r["path"])
        if len(notes) < 2 or is_collapsed(stats(notes)):
            continue
        r["runs"] = interval_runs(notes)
        out.append(r)
    return out


def prompt_runs(rec):
    """The retrieved/random phrases for one generation, truncated as the model saw
    them. Alignment is skipped: it is octave-only, and intervals are unchanged by
    any transposition -- the metric's own invariance makes the step unnecessary."""
    return {r["chunk"]: interval_runs(truncate(load_notes(TRAIN / r["chunk"])))
            for r in rec["retrieved"]}


def per_seed(recs, score_fn, n):
    """Mean within seed, then across seeds. Returns {seed: mean}."""
    by_seed = {}
    for r in recs:
        v = score_fn(r, n)
        if v is not None:
            by_seed.setdefault(r["seed"], []).append(v)
    return {s: float(np.mean(v)) for s, v in by_seed.items()}


def summarize(d, label, n):
    vals = list(d.values())
    print(f"    {label:26s} n={n}  {np.mean(vals):.3f}  "
          f"(sd {np.std(vals):.3f}, {len(vals)} seeds)")
    return float(np.mean(vals))


if __name__ == "__main__":
    rng = random.Random(42)
    train_paths = sorted(TRAIN.glob("*_t+00.mid"))
    train_idx = index_paths(train_paths)
    data = {s: clean(s) for s in SYSTEMS}

    # --- baselines, length-matched to the generations ---
    val_runs = [load_runs(p, seconds=10.0)
                for p in sorted(Path("data/val").glob("*_t+00.mid"))]

    for n in NS:
        print(f"\n{'='*64}\nSIDE 1 — overlap with the TRAIN set   (n={n})")
        ceil = [c for r in val_runs if (c := coverage(r, train_idx, n)) is not None]
        floor = [c for r in (shuffle_runs(x, rng) for x in val_runs)
                 if (c := coverage(r, train_idx, n)) is not None]
        print(f"    {'val (real, novel) CEILING':26s} n={n}  {np.mean(ceil):.3f}"
              f"   ({len(ceil)} chunks)")
        print(f"    {'shuffled FLOOR':26s} n={n}  {np.mean(floor):.3f}")

        for s in SYSTEMS:
            recs = data[s]
            summarize(per_seed(recs, lambda r, n: coverage(r["runs"], train_idx, n), n),
                      f"{s} generations", n)
            # per-system chance floor: controls for each system's own interval vocabulary
            summarize(per_seed(recs, lambda r, n: coverage(
                          shuffle_runs(r["runs"], rng), train_idx, n), n),
                      f"{s} shuffled", n)

        print(f"\nSIDE 2 — overlap with OWN PROMPT phrases   (n={n})")
        for s in ("Aprime", "B"):
            def score(r, n):
                idx = index_paths([], ns=(n,))          # empty shell
                for chunk, runs in prompt_runs(r).items():
                    for g in ngrams(runs, n):
                        idx[n].setdefault(g, set()).add(chunk)
                return coverage(r["runs"], idx, n)
            summarize(per_seed(data[s], score, n), f"{s} vs own prompt", n)

    # --- provenance: which train pieces does B match most? ---
    print(f"\n{'='*64}\nB: top matched source pieces (n=8)")
    src = Counter()
    for r in data["B"]:
        for g in ngrams(r["runs"], 8):              # n=8: distinctive by construction
            pieces = train_idx[8].get(g, ())
            if len(pieces) == 1:                    # unique to one piece = real fingerprint
                src[next(iter(pieces))] += 1
    total = sum(src.values()) or 1
    for piece, c in src.most_common(8):
        print(f"    {piece:34s} {c:6d}  ({100*c/total:4.1f}%)")

    