import json
import numpy as np
from features import load_notes, feature_vector, cosine
from pathlib import Path

INDEX_DIR = Path("data/train")

index_paths = sorted(INDEX_DIR.glob("*_t+00.mid"))
print(f"found {len(index_paths)} identity-transposition chunks")

vectors = []
kept_paths = []
failures = []

for p in index_paths:
    try:
        notes = load_notes(p)
        vectors.append(feature_vector(notes))
        kept_paths.append(p.name)          # .name — filename only, no path (remember!)
    except Exception as e:
        failures.append((p.name, repr(e)))

B = np.vstack(vectors)

# --- reconciliation: the numbers must add up ---
print(f"vectors: {len(vectors)}   paths: {len(kept_paths)}   failed: {len(failures)}")
assert len(vectors) == len(kept_paths), "paths/vectors desynced"
assert len(vectors) + len(failures) == len(index_paths), "files went missing"
print("B.shape:", B.shape)

for name, err in failures[:10]:
    print("  !!", name, err)

np.save("index_vectors.npy", B)
Path("index_paths.json").write_text(json.dumps(kept_paths))

sims = cosine(B[0], B)
print("self-similarity (must be 1.0):", sims[0])
print("mean similarity across corpus:", sims.mean())
print("min / max:", sims.min(), sims.max())

