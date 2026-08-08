import pretty_midi

WINDOW = 15.0     # chunk length in seconds
STEP   = 7.5      # advance per chunk (50% overlap)

MIN_CHUNK_NOTES = 8      # a chunk needs at least this many notes to be a usable phrase

def chunk_melody(inst, total_end):
    """Slice into overlapping windows; keep only chunks with enough notes."""
    chunks = []
    start = 0.0
    while start < total_end:
        end = start + WINDOW
        window_notes = [n for n in inst.notes if start <= n.start < end]
        if len(window_notes) >= MIN_CHUNK_NOTES:      # the new gate
            chunks.append((start, end, window_notes))
        start += STEP
    return chunks

import os

def write_chunk(notes, program, out_path):
    """Shift a chunk's notes to start at t=0 and write as a standalone .mid."""
    if not notes:
        return False
    offset = min(n.start for n in notes)          # the chunk's earliest note start

    mid  = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=n.velocity,
            pitch=n.pitch,
            start=n.start - offset,               # shift to zero
            end=n.end   - offset,
        ))
    mid.instruments.append(inst)
    mid.write(out_path)
    return True

if __name__ == '__main__':
    PIECE = '/home/whamel/music-rag/corpus/aletter_idx00_melody.mid'
    pm = pretty_midi.PrettyMIDI(PIECE)
    inst = pm.instruments[0]
    total_end = max(n.end for n in inst.notes)

    os.makedirs('/home/whamel/music-rag/chunks', exist_ok=True)
    chunks = chunk_melody(inst, total_end)

    for i, (start, end, notes) in enumerate(chunks[:3]):    # just first 3 as a test
        path = f'/home/whamel/music-rag/chunks/test_chunk{i:02d}.mid'
        write_chunk(notes, inst.program, path)
        # verify: re-read and check the first note starts near 0
        check = pretty_midi.PrettyMIDI(path)
        first = min(n.start for n in check.instruments[0].notes)
        print(f"chunk {i}: was {start:.1f}s, first note now starts at {first:.2f}s")
