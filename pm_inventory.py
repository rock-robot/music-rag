import glob, os
from collections import Counter
import pretty_midi

FOLDER = os.path.expanduser("~/music-rag/midi")

files = sorted(glob.glob(os.path.join(FOLDER, "*.mid")) +
               glob.glob(os.path.join(FOLDER, "*.midi")))
print(f"Found {len(files)} MIDI files\n")

# How many pieces contain each instrument (by GM name)
instrument_piece_counter = Counter()
# Total track count by instrument across the whole corpus
instrument_track_counter = Counter()
# How many pieces have drums
drum_pieces = 0
rows = []

for path in files:
    name = os.path.basename(path)
    try:
        pm = pretty_midi.PrettyMIDI(path)
    except Exception as e:
        print(f"  !! could not parse {name}: {e}")
        continue

    names_in_piece = set()
    has_drums = False
    for inst in pm.instruments:
        if inst.is_drum:
            label = "Drums/Percussion"
            has_drums = True
        else:
            # Convert GM program number -> human-readable instrument name
            label = pretty_midi.program_to_instrument_name(inst.program)
        names_in_piece.add(label)
        instrument_track_counter[label] += 1

    if has_drums:
        drum_pieces += 1
    for n in names_in_piece:
        instrument_piece_counter[n] += 1

    rows.append((name, len(pm.instruments), sorted(names_in_piece)))

print("=== Instruments: number of PIECES containing each ===")
for inst, c in instrument_piece_counter.most_common():
    print(f"  {inst:28s} {c} pieces")

print("\n=== Instruments: total TRACK count across corpus ===")
for inst, c in instrument_track_counter.most_common():
    print(f"  {inst:28s} {c} tracks")

print(f"\n=== Pieces containing drums/percussion: {drum_pieces} ===")

print("\n=== Per-piece instruments ===")
for name, n_inst, names in rows:
    print(f"\n  {name}  ({n_inst} tracks)")
    print(f"     {', '.join(names)}")