# inspect_durations.py -- look at the corpus ONLY, to choose bin edges honestly.
import numpy as np
from pathlib import Path
from features import load_notes

durs, iois = [], []
for f in sorted(Path("corpus").glob("*.mid")):
    notes = load_notes(f)
    durs += [n.end - n.start for n in notes]
    iois += [b.start - a.start for a, b in zip(notes, notes[1:])
             if b.start - a.start > 1e-4]          # drop simultaneities

for name, vals in (("duration", durs), ("IOI", iois)):
    v = np.array(vals)
    print(f"\n{name}: n={len(v)}  min={v.min():.3f}  max={v.max():.3f}  "
          f"median={np.median(v):.3f}")
    print("  percentiles:",
          {p: round(float(np.percentile(v, p)), 3)
           for p in (1, 5, 25, 50, 75, 95, 99)})
    print("  10 most common (rounded to 10ms):",
          [f"{val:.2f}s x{cnt}" for val, cnt in
           sorted(((float(x), int(c)) for x, c in
                   zip(*np.unique(np.round(v, 2), return_counts=True))),
                  key=lambda t: -t[1])[:10]])