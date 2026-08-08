import pretty_midi

PIECE = '/home/whamel/music-rag/midi/A Letter.mid'   # <-- your file

def avg_polyphony(inst):
    """Average number of notes sounding at once, while anything is playing."""
    events = []
    for n in inst.notes:
        events.append((n.start, 1))    # a note begins
        events.append((n.end,  -1))    # a note ends
    events.sort()
    cur = weighted = sounding = 0.0
    last_t = None
    for t, delta in events:
        if last_t is not None and cur > 0:
            weighted += cur * (t - last_t)
            sounding += (t - last_t)
        cur += delta
        last_t = t
    return (weighted / sounding) if sounding else 0.0

pm = pretty_midi.PrettyMIDI(PIECE)
print(f"{'idx':>3} {'prog':>4} {'name':<22} {'#notes':>6} {'poly':>5} {'low':>4} {'high':>4}")
for i, inst in enumerate(pm.instruments):
    if inst.is_drum or not inst.notes:
        continue
    name = pretty_midi.program_to_instrument_name(inst.program)
    poly = avg_polyphony(inst)
    pitches = [n.pitch for n in inst.notes]
    lo = pretty_midi.note_number_to_name(min(pitches))
    hi = pretty_midi.note_number_to_name(max(pitches))
    print(f"{i:>3} {inst.program:>4} {name:<22} {len(inst.notes):>6} {poly:>5.2f} {lo:>4} {hi:>4}")