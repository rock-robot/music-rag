import pretty_midi

PIECE = '/home/whamel/music-rag/midi/A Letter.mid'
IDX   = 2
OUT   = '/home/whamel/music-rag/midi/candidates/aletter_02_skyline.mid'

pm = pretty_midi.PrettyMIDI(PIECE)
notes = [n for n in pm.instruments[IDX].notes if n.end > n.start]

# 1. Every instant the texture can change is a note start or a note end
boundaries = sorted({n.start for n in notes} | {n.end for n in notes})

# 2. In each time-slice, the highest sounding note wins ("the skyline")
skyline = []                      # list of (start, end, pitch)
for a, b in zip(boundaries, boundaries[1:]):
    sounding = [n for n in notes if n.start <= a and n.end >= b]
    if not sounding:
        continue
    top = max(sounding, key=lambda n: n.pitch)
    # if the top pitch is unchanged and contiguous, extend the held note
    if skyline and skyline[-1][2] == top.pitch and abs(skyline[-1][1] - a) < 1e-6:
        skyline[-1] = (skyline[-1][0], b, top.pitch)
    else:
        skyline.append((a, b, top.pitch))

# 3. Rebuild as a clean, single-voice MIDI
mel  = pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(program=73)     # label the line "flute"
for start, end, pitch in skyline:
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=end))
mel.instruments.append(inst)
mel.write(OUT)

print(f"input notes: {len(notes)}  ->  skyline notes: {len(inst.notes)}")
print(f"wrote {OUT}")