import pretty_midi

PIECE = '/home/whamel/music-rag/midi/Avanti.mid'   # <-- pick one with a clear tune

pm = pretty_midi.PrettyMIDI(PIECE)

print(f"{'idx':>3}  {'program':>7}  {'name':<24}  {'#notes':>6}  drums?")
for i, inst in enumerate(pm.instruments):
    name = pretty_midi.program_to_instrument_name(inst.program)
    print(f"{i:>3}  {inst.program:>7}  {name:<24}  {len(inst.notes):>6}  {inst.is_drum}")