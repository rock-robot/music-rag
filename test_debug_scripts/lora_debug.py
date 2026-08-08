import pretty_midi

pm = pretty_midi.PrettyMIDI("generated/Gen2(tp tests)/out_lora_tp98.mid")
notes = sorted(pm.instruments[0].notes, key=lambda n: n.start)

print(f"{len(notes)} notes")
overlaps = 0
for a, b in zip(notes, notes[1:]):
    gap = b.start - a.end        # negative = b starts before a ends = overlap
    flag = "  <-- OVERLAP" if gap < -0.01 else ""
    print(f"  pitch {a.pitch:3d}  start {a.start:6.2f}  end {a.end:6.2f}  "
          f"next_gap {gap:+.2f}{flag}")
    if gap < -0.01:
        overlaps += 1
print(f"\n{overlaps} overlapping note pairs out of {len(notes)-1}")

# also: are there ever multiple notes at the SAME start time? (= a true chord)
from collections import Counter
starts = Counter(round(n.start, 2) for n in notes)
chords = {t: c for t, c in starts.items() if c > 1}
print(f"timestamps with multiple simultaneous note-onsets (true chords): {len(chords)}")
if chords:
    print(" ", chords)

