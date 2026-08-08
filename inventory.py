import glob, os
from collections import Counter
from music21 import converter, tempo

FOLDER = "midi"  # <-- change to your folder path

files = sorted(glob.glob(os.path.join(FOLDER, "*.mid")) +
               glob.glob(os.path.join(FOLDER, "*.midi")))
print(f"Found {len(files)} MIDI files\n")

instrument_counter = Counter()   # how many pieces contain each instrument
part_counts = Counter()          # how many pieces have N parts
key_counter = Counter()          # estimated key per piece
tempos = []
rows = []

for path in files:
    name = os.path.basename(path)
    try:
        score = converter.parse(path)
    except Exception as e:
        print(f"  !! could not parse {name}: {e}")
        continue

    n_parts = len(score.parts)
    part_counts[n_parts] += 1

    instr_names = set()
    for part in score.parts:
        inst = part.getInstrument(returnDefault=True)
        instr_names.add(inst.instrumentName or "Unknown")
    for n in instr_names:
        instrument_counter[n] += 1

    mm = score.flatten().getElementsByClass(tempo.MetronomeMark)
    bpm = round(mm[0].number) if len(mm) else None
    if bpm:
        tempos.append(bpm)

    try:
        k = str(score.analyze('key'))
    except Exception:
        k = "?"
    key_counter[k] += 1

    rows.append((name, n_parts, bpm, k))

print("\n=== Parts per piece ===")
for n, c in sorted(part_counts.items()):
    print(f"  {n} part(s): {c} pieces")

print("\n=== Instruments (pieces containing each) ===")
for inst, c in instrument_counter.most_common():
    print(f"  {inst}: {c}")

print("\n=== Estimated keys ===")
for k, c in key_counter.most_common():
    print(f"  {k}: {c}")

if tempos:
    print(f"\n=== Tempo (BPM) ===")
    print(f"  min {min(tempos)}, max {max(tempos)}, "
          f"mean {sum(tempos)//len(tempos)}, pieces with a tempo mark: {len(tempos)}")

print("\n=== Per-piece ===")
for name, n_parts, bpm, k in rows:
    print(f"  {name:40s} parts={n_parts}  bpm={bpm}  key={k}")