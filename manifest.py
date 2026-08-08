"""manifest.py — rendered clips -> blinded per-participant sessions.

Two comparisons: A'-vs-B is the headline (does retrieval help?), A-vs-A' is the
secondary check on the context effect. Both are blinded and side-randomised; the
mapping never reaches the browser.
"""
import json, random
from pathlib import Path

TIERS = {"quick":   {"AprimeB": 8,  "AAprime": 0},
         "standard":{"AprimeB": 15, "AAprime": 0},
         "full":    {"AprimeB": 15, "AAprime": 8}}
N_PARTICIPANTS = 60


def pair(t, left, right, rng):
    """One blinded trial. Returns (shown, key_entry)."""
    flip = rng.random() < .5
    l, r = (right, left) if flip else (left, right)
    return ({"seed": t["seed"], "instrument": t["instrument"],
             "clip_a": t["clips"][l], "clip_b": t["clips"][r]},
            {"seed": t["seed"], "comparison": f"{left}v{right}", "a": l, "b": r})


def build(pid, tier, rendered, rng):
    all_t = rendered["trials"]
    off = pid % len(all_t)                      # rotate: even seed coverage
    rot = all_t[off:] + all_t[:off]
    spec = TIERS[tier]

    trials, key = [], []
    for t in rot[:min(spec["AprimeB"], len(rot))]:
        s, k = pair(t, "Aprime", "B", rng); trials.append(s); key.append(k)
    for t in rot[:min(spec["AAprime"], len(rot))]:
        s, k = pair(t, "A", "Aprime", rng); trials.append(s); key.append(k)

    order = list(range(len(trials)))
    rng.shuffle(order)                          # interleave the two comparison types
    return ({"participant": str(pid), "tier": tier,
             "calibration": rendered["calibration"],
             "trials": [trials[i] for i in order]},
            {"participant": str(pid), "tier": tier,
             "key": [key[i] for i in order]})


def main():
    rendered = json.loads(Path("trials_rendered.json").read_text())
    missing = [t["seed"] for t in rendered["trials"]
               if not {"A","Aprime","B"} <= set(t["clips"])]
    if missing:
        raise SystemExit(f"seeds missing a system: {missing[:3]}")

    sessions, keys = [], []
    for pid in range(N_PARTICIPANTS):
        for tier in TIERS:
            s, k = build(pid, tier, rendered, random.Random(f"{pid}-{tier}"))
            sessions.append(s); keys.append(k)

    Path("sessions.json").write_text(json.dumps(sessions, separators=(",",":")))
    Path("MANIFEST_KEY.json").write_text(json.dumps(keys, indent=2))
    print(f"{len(sessions)} sessions -> sessions.json")
    print("MANIFEST_KEY.json is PRIVATE — keep it in music-rag.")


if __name__ == "__main__":
    main()