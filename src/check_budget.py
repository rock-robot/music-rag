"""check_budget.py — inspect System B prompt lengths before running the full batch.
Throwaway diagnostic. Measures; generates nothing."""

from pathlib import Path
import pretty_midi

from features import load_notes
from retrieve import load_index, retrieve
from seeds import load_seed_set, load_seed
from condition import align_to_seed, splice

K = 2
RETRIEVED_SECONDS = 2.5
GAP = 0.25


def truncate(notes, seconds):
    kept = window(notes, seconds)
    return kept if len(kept) >= 3 else notes[:3]

def window(notes, seconds):
    """Keep notes starting before `seconds`; cap any sustain at the window edge."""
    out = []
    for n in notes:
        if n.start >= seconds:
            continue
        out.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                    start=n.start, end=min(n.end, seconds)))
    return out


if __name__ == "__main__":
    B, paths = load_index()
    seeds = load_seed_set()

    print(f"k={K}  retrieved={RETRIEVED_SECONDS}s  gap={GAP}s\n")
    over = 0
    for rec in seeds:
        name = rec["name"]
        seed_notes = load_seed(name)

        hits = retrieve(seed_notes, B, paths, k=K, max_per_piece=1)
        aligned = []
        for hit_name, _ in hits:
            r = load_notes(Path("data/train") / hit_name)
            a, _ = align_to_seed(truncate(r, RETRIEVED_SECONDS), seed_notes)
            aligned.append(a)

        notes, start_time = splice(aligned, seed_notes, gap=GAP)
        flag = "  <-- OVER 15s" if start_time > 15 else ""
        over += start_time > 15
        print(f"  {start_time:5.1f}s  ({len(notes):3d} notes, {len(notes)*3:4d} tok)  {name}{flag}")

    print(f"\n{over}/{len(seeds)} prompts exceed the ~15s adapter training range")