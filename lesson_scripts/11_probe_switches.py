import pretty_midi

PIECE = '/home/whamel/music-rag/midi/A Letter.mid'
FLUTE_FAMILY = range(0, 7)   # idx 0–6 were the flute-family tracks in your inventory

pm = pretty_midi.PrettyMIDI(PIECE)

for idx in FLUTE_FAMILY:
    inst = pm.instruments[idx]
    notes = inst.notes
    if not notes:
        print(f"idx {idx:2d}: (empty)")
        continue
    start = min(n.start for n in notes)
    end   = max(n.end   for n in notes)
    name  = pretty_midi.program_to_instrument_name(inst.program)
    print(f"idx {idx:2d}: program {inst.program:3d} ({name:<16}) "
          f"{len(notes):4d} notes   plays {start:6.1f}s -> {end:6.1f}s")