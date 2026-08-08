"""manifest_paired.py — rename rendered clips so an app can pair them by filename.

Output naming:  s07_1.mp3 / s07_2.mp3   (seed index, slot)
  - seed index groups the pair            -> the app pairs on the shared "sNN_" prefix
  - slot (1/2) is which side plays         -> NOT tied to system; assigned per seed
                                              by a coin flip so slot order can't be
                                              read as system identity.

The slot -> system mapping is the blinding. It lives ONLY in PAIRING_KEY.json, which
stays in music-rag and is NEVER deployed. The app never needs it; you join on it when
the results come back.

Run from music-rag after render_audio.py.  Reads trials_rendered.json, copies audio
into paired/ under the new names, writes manifest_paired.json + PAIRING_KEY.json.
"""
import json, random, shutil, hashlib
from pathlib import Path

SITE      = Path("site")           # render_audio.py output
OUT       = Path("paired")         # renamed copies for the app
SEED      = 20260724               # fixes the slot coin-flips; change to reshuffle
COMPARISON = ("Aprime", "B")       # the two systems that form a trial pair


def src_path(clip):
    """A clip dict from trials_rendered.json -> its file under site/."""
    return SITE / clip["url"].split("audio/", 1)[-1]


def main():
    data = json.loads(Path("trials_rendered.json").read_text())
    trials_in = data["trials"]
    rng = random.Random(SEED)

    OUT.mkdir(exist_ok=True)
    (OUT / "audio").mkdir(exist_ok=True)
    (OUT / "calib").mkdir(exist_ok=True)

    manifest, key = [], []
    for i, t in enumerate(trials_in):
        clips = t.get("clips") or {k: t[k] for k in ("clip_a", "clip_b") if k in t}
        missing = [s for s in COMPARISON if s not in clips]
        if missing:
            raise SystemExit(f"{t['seed']}: missing {missing} — re-render with those systems")

        sid = f"s{i:02d}"
        # Coin-flip which system is slot 1. This is the side-randomisation; without
        # it, "slot 1 is always Aprime" would make the filename reveal the system.
        first, second = COMPARISON if rng.random() < .5 else COMPARISON[::-1]
        slot_system = {1: first, 2: second}

        pair = {}
        for slot, system in slot_system.items():
            c = clips[system]
            name = f"{sid}_{slot}.mp3"
            shutil.copy(src_path(c), OUT / "audio" / name)
            pair[f"clip_{slot}"] = {"file": f"audio/{name}",
                                    "boundary": c["boundary"],
                                    "duration": c["duration"]}
            key.append({"pair": sid, "seed": t["seed"], "slot": slot,
                        "system": system, "instrument": t["instrument"]})

        manifest.append({"pair": sid, "seed_hash": _hash(t["seed"]),
                         "instrument": t["instrument"], **pair})

    # calibration passes through unchanged; it isn't paired and has no system
    calib = []
    for j, c in enumerate(data["calibration"]):
        name = f"{j+1:02d}.mp3"
        shutil.copy(src_path(c), OUT / "calib" / name)
        calib.append({"file": f"calib/{name}", "boundary": c["boundary"],
                      "duration": c["duration"]})

    (OUT / "manifest_paired.json").write_text(json.dumps(
        {"calibration": calib, "pairs": manifest}, indent=2))
    Path("PAIRING_KEY.json").write_text(json.dumps(key, indent=2))

    print(f"{len(manifest)} pairs -> {OUT}/audio/  (sNN_1.mp3 / sNN_2.mp3)")
    print(f"{len(calib)} calibration clips -> {OUT}/calib/")
    print("manifest_paired.json + audio/  are safe to deploy.")
    print("PAIRING_KEY.json is PRIVATE — it maps slot->system. Keep it in music-rag.")


def _hash(seed_name):
    """Opaque per-seed id for the public manifest — lets you sanity-check pairs
    without exposing the real seed filename."""
    return hashlib.sha1(seed_name.encode()).hexdigest()[:8]


if __name__ == "__main__":
    main()