import json
import numpy as np
from collections import Counter
from pathlib import Path

from features import feature_vector, cosine, piece_name, load_notes


def load_index(vec_path="index_vectors.npy", path_path="index_paths.json"):
    """Load the index matrix and its filenames — TOGETHER, always."""
    B = np.load(vec_path)
    paths = json.loads(Path(path_path).read_text())
    assert len(paths) == B.shape[0], f"desync: {len(paths)} paths vs {B.shape[0]} rows"
    return B, paths


def retrieve(seed_notes, B, paths, k=3, max_per_piece=1, exclude_piece=None):
    """Rank the index against a seed melody; return top-k with a per-piece cap.

    seed_notes:    list of Notes — the query phrase
    B:             (N, 35) index matrix
    paths:         list of N filenames, SAME ORDER as B's rows
    k:             how many phrases to retrieve
    max_per_piece: diversity cap — how many chunks one piece may contribute
    exclude_piece: skip all chunks from this piece (for self-query sanity tests)

    Returns: list of (filename, similarity), best first, length <= k
    """
    q = feature_vector(seed_notes)          # SAME featurizer as the index. Non-negotiable.
    sims = cosine(q, B)                     # (N,) — every chunk scored at once
    order = np.argsort(-sims)               # indices, best first

    taken = Counter()
    results = []

    for i in order:
        name = paths[i]
        piece = piece_name(name)

        if piece == exclude_piece:
            continue
        if taken[piece] >= max_per_piece:
            continue

        results.append((name, float(sims[i])))
        taken[piece] += 1

        if len(results) == k:
            break

    return results


if __name__ == "__main__":
    import json
    import numpy as np
    from pathlib import Path

    B = np.load("index_vectors.npy")
    paths = json.loads(Path("index_paths.json").read_text())

    seed_file = Path("data/val/spanish_sunrise_idx01_melody_chunk05_t+00.mid")
    seed = load_notes(seed_file)

    print("QUERY:", seed_file.name)
    print("\n-- no cap (max_per_piece=99) --")
    for name, s in retrieve(seed, B, paths, k=5, max_per_piece=99,
                            exclude_piece=piece_name(seed_file)):
        print(f"  {s:.3f}  {name}")

    print("\n-- diversity cap (max_per_piece=1) --")
    for name, s in retrieve(seed, B, paths, k=5, max_per_piece=1,
                            exclude_piece=piece_name(seed_file)):
        print(f"  {s:.3f}  {name}")