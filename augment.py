import pretty_midi

# A sane melodic pitch range. Notes outside this after transposing signal that
# the shift pushed the melody somewhere implausible.
PITCH_MIN = 21     # A0, bottom of a piano
PITCH_MAX = 108    # C8, top of a piano

def transpose_notes(notes, semitones):
    """Shift every note's pitch by `semitones`. Returns (new_notes, ok).
    ok=False if any note would land outside the sane range."""
    shifted = []
    for n in notes:
        new_pitch = n.pitch + semitones
        if not (PITCH_MIN <= new_pitch <= PITCH_MAX):
            return None, False          # this transposition is out of range — reject it
    # only build the notes if ALL of them are in range
    for n in notes:
        shifted.append(pretty_midi.Note(
            velocity=n.velocity,
            pitch=n.pitch + semitones,
            start=n.start,
            end=n.end,
        ))
    return shifted, True

def all_transpositions(notes):
    """Produce every in-range transposition of a chunk across all 12 keys
    (centered shifts -6..+5). Returns list of (semitones, shifted_notes)."""
    out = []
    for semi in range(-6, 6):              # -6, -5, ..., +4, +5  = 12 shifts
        shifted, ok = transpose_notes(notes, semi)
        if ok:
            out.append((semi, shifted))
    return out    

def transposition_tag(semitones):
    """Format a shift as a filename-safe tag: +5 -> 't+05', -6 -> 't-06'."""
    sign = '+' if semitones >= 0 else '-'
    return f"t{sign}{abs(semitones):02d}"

import glob, os

CHUNK_DIR = '/home/whamel/music-rag/chunks'
AUG_DIR   = '/home/whamel/music-rag/augmented'

def augment_chunk_file(path):
    """Read one chunk, write all its in-range transpositions. Returns count written."""
    pm = pretty_midi.PrettyMIDI(path)
    notes = pm.instruments[0].notes
    program = pm.instruments[0].program
    base = os.path.splitext(os.path.basename(path))[0]     # e.g. a_letter_idx00_melody_chunk03

    written = 0
    for semi, shifted in all_transpositions(notes):
        mid  = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=program)
        inst.notes = shifted
        mid.instruments.append(inst)
        out = os.path.join(AUG_DIR, f"{base}_{transposition_tag(semi)}.mid")
        mid.write(out)
        written += 1
    return written

if __name__ == '__main__':
    os.makedirs(AUG_DIR, exist_ok=True)
    chunks = sorted(glob.glob(os.path.join(CHUNK_DIR, '*.mid')))
    print(f"found {len(chunks)} chunks -> up to {len(chunks)*12} augmented files\n")

    total = 0
    problems = []
    for i, path in enumerate(chunks):
        try:
            total += augment_chunk_file(path)
        except Exception as e:
            problems.append((os.path.basename(path), str(e)))
        if (i + 1) % 500 == 0:                       # progress ping every 500 chunks
            print(f"  ...{i+1}/{len(chunks)} chunks done, {total} files so far")

    print(f"\n{'='*50}")
    print(f"augmented {len(chunks)-len(problems)} chunks -> {total} files")
    print(f"flagged {len(problems)} chunks")
    for name, reason in problems[:20]:
        print(f"  {name}: {reason}")