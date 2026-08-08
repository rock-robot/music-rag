"""seeds.py — build and freeze the fixed seed set for the A/B experiment."""

import json
import random
from pathlib import Path

from features import load_notes, piece_name

VAL_DIR      = Path("data/val")
SEED_FILE    = Path("seeds.json")
N_SEEDS      = 20
RANDOM_SEED  = 42                 # same seed as the train/val split. One number, one project.
SEED_SECONDS = 6.0                # how much melody to prompt with

MIN_SEED_NOTES = 6



def is_viable(path, seconds=SEED_SECONDS, min_notes=MIN_SEED_NOTES):
    notes = load_notes(path)
    return sum(1 for n in notes if n.start < seconds) >= min_notes


def build_seed_set(n=N_SEEDS, seed=RANDOM_SEED):
    """Choose n chunks from the val split, stratified across the 7 val pieces."""
    candidates = [p for p in sorted(VAL_DIR.glob("*_t+00.mid")) if is_viable(p)]
    print(f"{len(candidates)} viable candidates (>= {MIN_SEED_NOTES} notes in {SEED_SECONDS}s)")
    if not candidates:
        raise FileNotFoundError(f"no *_t+00.mid in {VAL_DIR}")

    # group by piece so every val piece is represented
    by_piece = {}
    for p in candidates:
        by_piece.setdefault(piece_name(p), []).append(p)

    rng = random.Random(seed)
    chosen = []
    pieces = sorted(by_piece)                             # sorted -> deterministic
    i = 0
    while len(chosen) < n:
        piece = pieces[i % len(pieces)]                   # round-robin across pieces
        pool = by_piece[piece]
        pick = rng.choice(pool)
        if pick not in chosen:
            chosen.append(pick)
        i += 1
        if i > 1000:
            raise RuntimeError("could not fill the seed set — too few candidates")

    return [p.name for p in chosen]

def save_seed_set(names, path=SEED_FILE):
    """Save names WITH their note counts — the density stratifier for Week 4."""
    records = [{"name": x, "notes": len(load_seed(x))} for x in names]
    path.write_text(json.dumps(records, indent=2))


def load_seed_set(path=SEED_FILE):
    """Load the FROZEN seed set. Never regenerate — read what was frozen."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run `python seeds.py` ONCE to freeze the seed set, "
            "then never again — Week 4 must evaluate the same seeds as Week 3."
        )
    return json.loads(path.read_text())

from features import load_notes, window

def load_seed(name, seconds=SEED_SECONDS):
    notes = window(load_notes(VAL_DIR / name), seconds)
    if len(notes) < 4:
        raise ValueError(f"{name}: only {len(notes)} notes in first {seconds}s")
    return notes

if __name__ == "__main__":
    from collections import Counter

    names = build_seed_set()
    save_seed_set(names)

    print("\nseeds by piece:")
    for piece, n in Counter(piece_name(x) for x in names).most_common():
        print(f"  {n}  {piece}")

    print("\nnote counts in the first 6s:")
    for x in names:
        try:
            print(f"  {len(load_seed(x)):>3}  {x}")
        except ValueError as e:
            print(f"  !!  {e}")