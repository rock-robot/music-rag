import os
import pretty_midi

PIECE   = '/home/whamel/music-rag/midi/A Letter.mid'
OUT_DIR = '/home/whamel/music-rag/midi/candidates'
CANDIDATES = [0, 3, 11, 17]   # piccolo, flute, clarinet, soprano sax — the lead shortlist

os.makedirs(OUT_DIR, exist_ok=True)
pm = pretty_midi.PrettyMIDI(PIECE)

for idx in CANDIDATES:
    inst = pm.instruments[idx]
    name = pretty_midi.program_to_instrument_name(inst.program).replace(' ', '')

    solo = pretty_midi.PrettyMIDI()          # fresh empty MIDI
    solo.instruments.append(inst)            # just this one line
    out = os.path.join(OUT_DIR, f'aletter_{idx:02d}_{name}.mid')
    solo.write(out)
    print(f'idx {idx:>2}  {name:<12}  {len(inst.notes):>4} notes  -> {out}')