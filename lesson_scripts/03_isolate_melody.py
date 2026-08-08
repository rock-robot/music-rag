import pretty_midi

PIECE = '/home/whamel/music-rag/midi/Avanti.mid'           # <-- your file
OUT   = '/home/whamel/music-rag/midi/avanti_melody.mid'
MELODY_PROGRAMS = [71]   # clarinet (GM program 71) — the voice carrying the tune

pm = pretty_midi.PrettyMIDI(PIECE)
melody = pretty_midi.PrettyMIDI()          # a new, empty MIDI to hold only the melody

kept = []
for inst in pm.instruments:
    if not inst.is_drum and inst.program in MELODY_PROGRAMS:
        melody.instruments.append(inst)
        kept.append(pretty_midi.program_to_instrument_name(inst.program))

print(f"kept {len(melody.instruments)} track(s): {kept}")
print(f"melody notes: {sum(len(i.notes) for i in melody.instruments)}")
melody.write(OUT)
print(f"wrote {OUT}")