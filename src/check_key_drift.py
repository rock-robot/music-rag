# check_key_drift.py -- does the retrieved phrase's key pull the continuation?
import numpy as np
from pathlib import Path
from features import load_notes
from metrics_core import pitch_class_counts, to_hist
from results import load_results

PC = list(range(12))

def best_rotation(p, q):
    """Semitone rotation of q that best matches p, and the cosine at that rotation."""
    scores = [float(np.dot(p, np.roll(q, r))) for r in range(12)]
    r = int(np.argmax(scores))
    return (r if r <= 6 else r - 12), scores[r]

rows = []
for rec in load_results("B"):
    gen  = to_hist(pitch_class_counts(load_notes(rec["path"])), PC)
    for r in rec["retrieved"]:
        ret = to_hist(pitch_class_counts(
                  load_notes(Path("data/train") / r["chunk"])), PC)
        shift, _ = best_rotation(gen, ret)
        rows.append({"seed": rec["seed"], "piece": r["piece"],
                     "sim": r["sim"], "key_shift": shift})

shifts = [abs(r["key_shift"]) for r in rows]
print(f"{len(rows)} (generation, retrieved-phrase) pairs")
print(f"  |key offset| mean {np.mean(shifts):.2f} semitones, "
      f"median {np.median(shifts):.1f}")
print(f"  in-key (offset 0): {sum(s == 0 for s in shifts)/len(shifts):.1%}")