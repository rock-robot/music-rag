"""density_and_reference.py -- measure the density gap, build the training-matched reference."""
import numpy as np
from pathlib import Path

import metrics_rhythm
from features import load_notes
from metrics_core import tv_distance
from corpus_reference import excerpts, pooled, averaged, build_references
from results import load_all

BINS   = list(range(metrics_rhythm.N_BINS))
LABELS = metrics_rhythm.bin_labels()

# --- 1. how dense is each population, per 10s? ---
corpus_exc = [e for p in sorted(Path("corpus").glob("*.mid"))
                for e in excerpts(load_notes(p))]
print(f"corpus 10s excerpts: n={len(corpus_exc)}  "
      f"median {np.median([len(e) for e in corpus_exc]):.0f} notes  "
      f"mean {np.mean([len(e) for e in corpus_exc]):.1f}")

for s, recs in load_all(ok_only=True).items():
    ns = [r["n_notes"] for r in recs]
    print(f"  {s:8s} generations: n={len(ns)}  median {np.median(ns):.0f}  "
          f"mean {np.mean(ns):.1f}  range {min(ns)}-{max(ns)}")

# --- 2. the training-distribution reference: chunks the model actually saw ---
chunks = sorted(Path("data/train").glob("*_t+00.mid")) + \
         sorted(Path("data/val").glob("*_t+00.mid"))
chunk_units = [e[0] for p in chunks if (e := excerpts(load_notes(p)))]
print(f"\nchunk-derived 10s units: {len(chunk_units)}  "
      f"median {np.median([len(e) for e in chunk_units]):.0f} notes")

refs = build_references(metrics_rhythm.duration_counts, BINS)
refs["chunk_avg"] = averaged(chunk_units, metrics_rhythm.duration_counts, BINS)

print(f"\n  {'bin':>10} " + " ".join(f"{k:>13s}" for k in refs))
for i, lab in enumerate(LABELS):
    print(f"  {lab:>10} " + " ".join(f"{refs[k][i]:13.3f}" for k in refs))

print("\n  chunk_avg vs others:")
for k in refs:
    if k != "chunk_avg":
        print(f"    {k:15s} {tv_distance(refs['chunk_avg'], refs[k]):.4f}")