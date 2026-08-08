"""select_clips.py — choose ONE generation per seed, matched across systems.

The sample index must be SHARED by A / A' / B: generate_ab.py used
torch.manual_seed(i) matched across systems, so sample i of A' and sample i of B
differ only in their prompt. Picking different indices per system would discard
that pairing and put sampling noise back into the headline comparison.
"""
import json
from pathlib import Path

from features import load_notes
from filters import is_collapsed, stats
from results import load_results, SYSTEMS


def usable(system):
    """{(seed, sample) : record} for generations passing the union filter."""
    out = {}
    for r in load_results(system, ok_only=False):
        notes = load_notes(r["path"])
        if len(notes) >= 2 and not is_collapsed(stats(notes)):
            out[(r["seed"], r["sample"])] = r
    return out


def choose(n_samples=5):
    tables = {s: usable(s) for s in SYSTEMS}
    seeds = sorted({seed for t in tables.values() for seed, _ in t})

    chosen, dropped = [], []
    for seed in seeds:
        # lowest index that survives in ALL systems -- fixed rule, no judgement
        idx = next((i for i in range(n_samples)
                    if all((seed, i) in tables[s] for s in SYSTEMS)), None)
        if idx is None:
            dropped.append(seed)
            continue
        chosen.append({"seed": seed, "sample": idx,
                       "paths": {s: str(tables[s][(seed, idx)]["path"]) for s in SYSTEMS},
                       "seed_notes": tables["B"][(seed, idx)].get("seed_notes")})

    print(f"{len(chosen)} seeds usable, {len(dropped)} dropped: {dropped or 'none'}")
    for c in chosen:
        print(f"  {c['seed']:44s} sample {c['sample']}")
    Path("selected.json").write_text(json.dumps(chosen, indent=2))
    return chosen


if __name__ == "__main__":
    choose()