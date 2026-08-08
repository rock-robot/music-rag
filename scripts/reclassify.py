"""reclassify.py — re-verdict the saved generations under filters that do NOT
reference interval-0, then re-run the interval metric under each.

Why: the original classifier filtered on interval-0 fraction and the metric
reports interval-0. Conditioning on the statistic you report truncates it --
unequally across systems (B lost 7 generations, A' lost 2).
"""
import numpy as np
from collections import Counter

import features
from features import INTERVAL_BINS, load_notes
from metrics_core import safe, to_hist, tv_distance, aggregate_pooled, aggregate_per_seed
from results import load_results, SYSTEMS
from run_metrics import CORPUS, paired

MAX_NOTES     = 200
MAX_ZERO_FRAC = 0.40
MIN_DISTINCT  = 3


from filters import stats, MAX_NOTES, MAX_ZERO_FRAC, MIN_DISTINCT


def old(s):        # what generate_ab.py used -- contaminated
    return s["n"] > MAX_NOTES or s["zero_frac"] > MAX_ZERO_FRAC

def hybrid(s):     # note-count + distinct-pitch: no explicit interval-0 term
    return s["n"] > MAX_NOTES or s["distinct"] < MIN_DISTINCT

def distinct_only(s):   # fully independent of BOTH interval-0 and density
    return s["distinct"] < MIN_DISTINCT

CRITERIA = {"old": old, "hybrid": hybrid, "distinct_only": distinct_only}


def verdicts():
    """Load ALL generations (including previously-collapsed) and stat each once."""
    out = {}
    for sysname in SYSTEMS:
        recs = load_results(sysname, ok_only=False)
        for r in recs:
            notes = load_notes(r["path"])
            r["stats"] = stats(notes) if len(notes) >= 2 else None
        out[sysname] = recs
    return out


def survivors(recs, crit):
    """Records passing `crit`. Empty (<2 notes) is always excluded."""
    return [r for r in recs if r["stats"] is not None and not crit(r["stats"])]


def report_filters(data):
    print(f"\n{'='*70}\nfilter comparison (of 80 generations per system)")
    print(f"  {'system':8s} " + " ".join(f"{k:>14s}" for k in CRITERIA))
    for sysname, recs in data.items():
        row = [f"{len(survivors(recs, c)):>7d} kept" for c in CRITERIA.values()]
        print(f"  {sysname:8s} " + " ".join(f"{x:>14s}" for x in row))

    print("\n  disagreements (old vs hybrid):")
    for sysname, recs in data.items():
        for r in recs:
            if r["stats"] is None:
                continue
            o, h = old(r["stats"]), hybrid(r["stats"])
            if o != h:
                tag = "old dropped, hybrid KEEPS" if o else "old kept, hybrid DROPS"
                print(f"    {sysname:7s} {r['seed']}#{r['sample']}  {tag}  "
                      f"(n={r['stats']['n']}, distinct={r['stats']['distinct']}, "
                      f"zero_frac={r['stats']['zero_frac']:.2f})")


def rerun_interval(data, crit_name):
    """Interval histogram + A'->B headline under one filter."""
    crit = CRITERIA[crit_name]
    counts_fn = safe(features.interval_counts)
    ref, _, _ = aggregate_pooled(CORPUS, counts_fn, INTERVAL_BINS)

    grand, dists, kept = {}, {}, {}
    for sysname, recs in data.items():
        surv = survivors(recs, crit)
        kept[sysname] = len(surv)
        g, seed_means, _ = aggregate_per_seed(surv, counts_fn, INTERVAL_BINS)
        grand[sysname], dists[sysname] = g, {s: tv_distance(h, ref)
                                             for s, h in seed_means.items()}

    z = INTERVAL_BINS.index(0)
    print(f"\n  [{crit_name}]  kept " +
          ", ".join(f"{s} {kept[s]}" for s in SYSTEMS))
    print(f"    interval-0   corpus {ref[z]:.3f} | " +
          " ".join(f"{s} {grand[s][z]:.3f}" for s in SYSTEMS))
    for leg, (a, b) in (("A ->A'", ("A", "Aprime")), ("A'->B ", ("Aprime", "B"))):
        n, md, nb = paired(dists[a], dists[b])
        print(f"    {leg} paired n={n}  mean delta {md:+.4f}  ({nb}/{n} favour {b})")


if __name__ == "__main__":
    data = verdicts()
    report_filters(data)
    print(f"\n{'='*70}\ninterval metric under each filter")
    for name in CRITERIA:
        rerun_interval(data, name)