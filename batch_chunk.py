import os
import glob
import pretty_midi
from chunk import chunk_melody, write_chunk   # the per-file tools we built

CORPUS_DIR = '/home/whamel/music-rag/corpus'   # the 462 melody files
CHUNK_DIR  = '/home/whamel/music-rag/chunks'   # where chunks go

if __name__ == '__main__':
    os.makedirs(CHUNK_DIR, exist_ok=True)
    melodies = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.mid')))
    print(f"found {len(melodies)} melody files to chunk\n")
    total_chunks = 0
    results  = []      # (melody_name, n_chunks)
    problems = []      # (melody_name, reason)

    for path in melodies:
        name = os.path.splitext(os.path.basename(path))[0]   # e.g. 'a_letter_idx00_melody'
        try:
            pm = pretty_midi.PrettyMIDI(path)
            inst = pm.instruments[0]                          # melody files are single-track
            if not inst.notes:
                problems.append((name, "empty melody file"))
                continue

            total_end = max(n.end for n in inst.notes)
            chunks = chunk_melody(inst, total_end)

            if not chunks:                                    # nothing cleared the note-floor
                problems.append((name, "no chunks passed the note-floor"))
                continue

            for i, (start, end, notes) in enumerate(chunks):
                out = os.path.join(CHUNK_DIR, f"{name}_chunk{i:02d}.mid")
                write_chunk(notes, inst.program, out)
                total_chunks += 1
            results.append((name, len(chunks)))

        except Exception as e:
            problems.append((name, f"ERROR: {e}"))

    # --- report ---
    print(f"\n{'='*50}")
    print(f"chunked {len(results)} melodies -> {total_chunks} chunks")
    print(f"flagged {len(problems)} melodies")
    if problems:
        print("\nFLAGGED:")
        for name, reason in problems[:20]:      # cap the list if long
            print(f"  {name:<38} {reason}")
        if len(problems) > 20:
            print(f"  ... and {len(problems)-20} more")
