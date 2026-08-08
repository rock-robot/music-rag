import pretty_midi
from collections import Counter

PIECE = '/home/whamel/music-rag/midi/A Letter.mid'
IDX   = 2

pm = pretty_midi.PrettyMIDI(PIECE)
inst = pm.instruments[IDX]
notes = sorted(inst.notes, key=lambda n: n.start)

# For each note, count how many *other* notes overlap it, and log the interval
overlaps = 0
intervals = Counter()
for i, n in enumerate(notes):
    for m in notes:
        if m is n:
            continue
        # do n and m sound at the same time?
        if m.start < n.end and n.start < m.end:
            overlaps += 1
            intervals[abs(n.pitch - m.pitch)] += 1
            break  # count each note once

print(f"track {IDX}: {len(notes)} notes, {overlaps} of them overlap another note "
      f"({100*overlaps/len(notes):.0f}%)")
print("\nmost common intervals (semitones) between overlapping notes:")
for semis, count in intervals.most_common(8):
    print(f"  {semis:>2} semitones  ({count} times)")