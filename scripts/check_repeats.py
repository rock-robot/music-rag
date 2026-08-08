import pretty_midi
from pathlib import Path

GEN_DIR = "gen_lora"   # <-- the folder with your 20 LoRA generations

files = sorted(Path(GEN_DIR).glob("*.mid"))
print(f"found {len(files)} files in {GEN_DIR}/")

for f in files[:3]:
    pm = pretty_midi.PrettyMIDI(str(f))
    notes = sorted(pm.instruments[0].notes, key=lambda n: n.start)
    print(f"\n{f.name}: {len(notes)} notes")

    repeated = 0
    for a, b in zip(notes, notes[1:]):
        if a.pitch == b.pitch:                 # consecutive notes, same pitch = interval 0
            gap = b.start - a.end
            print(f"  pitch {a.pitch:3d}   "
                  f"{a.start:5.2f}-{a.end:5.2f}  ->  {b.start:5.2f}-{b.end:5.2f}"
                  f"   gap {gap:+.2f}")
            repeated += 1
    print(f"  ({repeated} repeated-pitch pairs)")