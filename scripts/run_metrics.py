"""run_metrics.py — Week 4 objective metrics with a reference-sensitivity table.

Absolute distance-from-corpus depends heavily on how "the corpus" is constructed
(up to 0.28 TV between constructions, ~10x the effect size). So every comparison
is reported under all five references. The PAIRED DELTAS are the result; the
absolute numbers are not portable.
"""
import numpy as np
from pathlib import Path

import features, metrics_rhythm
from features import INTERVAL_BINS, load_notes
from metrics_core import (pitch_class_counts, safe, to_hist, tv_distance,
                          aggregate_per_seed)
from corpus_reference import excerpts, pooled, averaged
from filters import is_collapsed, tv_nonzero, stats
from results import load_results, SYSTEMS

CORPUS  = sorted(Path("corpus").glob("*.mid"))
CHUNKS  = (sorted(Path("data/train").glob("*_t+00.mid")) +
           sorted(Path("data/val").glob("*_t+00.mid")))
OUTLIER = "el_mar_idx04_melody_chunk05_t+00.mid"

SPECS = {
    "pitch_class": (pitch_class_counts, list(range(12)),
                    ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"],
                    tv_distance),
    "interval":    (safe(features.interval_counts), INTERVAL_BINS,
                    [f"{i:+d}" for i in INTERVAL_BINS],
                    tv_nonzero),          # bin 0 is conditioned by the filter
    "duration":    (metrics_rhythm.duration_counts,
                    list(range(metrics_rhythm.N_BINS)),
                    metrics_rhythm.bin_labels(), tv_distance),
}


# ---------- populations: loaded once, reused by every metric ----------

def load_populations():
    melodies = [load_notes(p) for p in CORPUS]
    exc      = [e for m in melodies for e in excerpts(m)]
    units    = [e[0] for p in CHUNKS if (e := excerpts(load_notes(p)))]
    print(f"populations: {len(melodies)} melodies | {len(exc)} excerpts | "
          f"{len(units)} chunk units "
          f"(median notes: {np.median([len(u) for u in units]):.0f})")
    return melodies, exc, units


def references(pop, counts_fn, bins):
    """The same corpus, constructed five ways. Order: length axis, then aggregation."""
    melodies, exc, units = pop
    return {"whole_pooled":   pooled(melodies, counts_fn, bins),
            "excerpt_pooled": pooled(exc,      counts_fn, bins),
            "whole_avg":      averaged(melodies, counts_fn, bins),
            "excerpt_avg":    averaged(exc,      counts_fn, bins),
            "chunk_avg":      averaged(units,    counts_fn, bins)}


# ---------- generations ----------

def clean(sysname):
    """Generations surviving the UNION collapse filter (both degeneracy modes)."""
    out = []
    for r in load_results(sysname, ok_only=False):
        notes = load_notes(r["path"])
        if len(notes) >= 2 and not is_collapsed(stats(notes)):
            out.append(r)
    return out


def paired(dist_a, dist_b, drop=()):
    """Compare two systems only on seeds BOTH produced. Negative delta favours b."""
    shared = sorted((set(dist_a) & set(dist_b)) - set(drop))
    if not shared:
        return 0, float("nan"), 0
    diffs = [dist_b[s] - dist_a[s] for s in shared]
    return len(shared), float(np.mean(diffs)), sum(d < 0 for d in diffs)


# ---------- report ----------

def report(name, counts_fn, bins, labels, dist_fn, data, pop):
    refs = references(pop, counts_fn, bins)

    # seed_means are reference-independent: compute ONCE per system.
    seed_means = {}
    for s in SYSTEMS:
        g, sm, dropped = aggregate_per_seed(data[s], counts_fn, bins)
        seed_means[s] = sm
        if dropped:
            print(f"  {s}: dropped {len(dropped)}")

    print(f"\n{'='*78}\n{name}")
    print(f"\n  distribution vs the primary reference (chunk_avg)")
    print(f"  {'bin':>10} {'chunk_avg':>10} " + " ".join(f"{s:>9s}" for s in SYSTEMS))
    ref0 = refs["chunk_avg"]
    grand = {s: (np.mean(list(sm.values()), axis=0) if sm else np.zeros(len(bins)))
             for s, sm in seed_means.items()}
    for i, lab in enumerate(labels):
        if ref0[i] < 0.005 and all(grand[s][i] < 0.005 for s in SYSTEMS):
            continue
        print(f"  {lab:>10} {ref0[i]:10.3f} " +
              " ".join(f"{grand[s][i]:9.3f}" for s in SYSTEMS))

    print(f"\n  reference sensitivity  (delta<0 favours the second system)")
    print(f"  {'reference':>15} {'A':>7} {'A-prime':>8} {'B':>7} "
          f"| {'A->A-prime':>17} | {'A-prime->B':>17} | {'ex-outlier':>12}")
    for key, ref in refs.items():
        d = {s: {seed: dist_fn(h, ref) for seed, h in sm.items()}
             for s, sm in seed_means.items()}
        means = " ".join(f"{np.mean(list(d[s].values())):7.3f}" for s in SYSTEMS)
        n1, m1, f1 = paired(d["A"], d["Aprime"])
        n2, m2, f2 = paired(d["Aprime"], d["B"])
        n3, m3, f3 = paired(d["Aprime"], d["B"], drop=(OUTLIER,))
        print(f"  {key:>15} {means} | {m1:+8.4f} {f1:2d}/{n1:2d} seeds "
              f"| {m2:+8.4f} {f2:2d}/{n2:2d} seeds | {m3:+8.4f} {f3:2d}/{n3:2d}")

    return refs, seed_means


if __name__ == "__main__":
    pop  = load_populations()
    data = {s: clean(s) for s in SYSTEMS}
    print("after union filter: " +
          ", ".join(f"{s} {len(data[s])}/80" for s in SYSTEMS))
    out = {name: report(name, *spec, data, pop) for name, spec in SPECS.items()}