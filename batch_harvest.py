import os
import glob
from harvest import harvest_piece, write_melodies   # clean import, thanks to the rename

MIDI_DIR = '/home/whamel/music-rag/midi'

# Files sitting in midi/ that are NOT training-corpus pieces.
# Matched against the cleaned piece_name(). Keep this commented — future-you
# needs to know WHY each is here.
EXCLUDE = {
    'avanti_melody',   # working file: isolated clarinet line from an early lesson
    'flutesolo1',      # working file: continuation-test seed
    'ex_machina',      # dropped in Week 0: electronic, no usable melodic line
}

def piece_name(path):
    """'/path/A Letter.mid' -> clean corpus prefix 'a_letter'."""
    base = os.path.splitext(os.path.basename(path))[0]
    return base.lower().replace(' ', '_').replace('__', '_')

import pretty_midi

if __name__ == '__main__':
    all_mids = sorted(glob.glob(os.path.join(MIDI_DIR, '*.mid')))
    pieces   = [p for p in all_mids if piece_name(p) not in EXCLUDE]

    total_files = 0
    results  = []      # (name, n_files) for successes
    problems = []      # (name, reason) for flagged pieces

    for path in pieces:
        name = piece_name(path)
        try:
            pm = pretty_midi.PrettyMIDI(path)
            survivors = harvest_piece(pm, name)   # was: harvest_piece(pm)

            if not survivors:                       # no melody found — flag, don't write
                problems.append((name, "no melody tracks survived the gates"))
                continue

            written = write_melodies(survivors, name)
            total_files += len(written)
            results.append((name, len(written)))

        except Exception as e:                       # any failure: flag it, keep going
            problems.append((name, f"ERROR: {e}"))

    # --- the end-of-run report ---
    print(f"\n{'='*50}")
    print(f"harvested {len(results)} pieces -> {total_files} melody files")
    print(f"flagged {len(problems)} pieces\n")
    for name, n in results:
        print(f"  {name:<32} {n:>3} files")
    if problems:
        print("\nFLAGGED:")
        for name, reason in problems:
            print(f"  {name:<32} {reason}")