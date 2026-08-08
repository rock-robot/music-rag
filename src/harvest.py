# ==========================================================================
# 15_harvest.py — turn one piece's melody tracks into clean monophonic
# corpus files, through three gates: length -> skyline -> de-dupe.
# ==========================================================================

# --- 1. IMPORTS -----------------------------------------------------------
import os
import pretty_midi
from importlib import import_module

from router import route_track

# --- 2. CONSTANTS ---------------------------------------------------------
MIN_NOTES      = 25       # a reduced line shorter than this isn't a usable melody
DUPE_THRESHOLD = 0.80     # PROVISIONAL: only the "independent" end is validated
                          # (A Letter topped ~28%); never yet tested against a
                          # confirmed doubling. Revisit when a real one appears.
CORPUS_DIR     = '/home/whamel/music-rag/corpus'

# --- 3. FUNCTIONS ---------------------------------------------------------

def melody_tracks(pm, piece_name=None):
    """(idx, inst) for every track the router tags 'melody'."""
    out = []
    for idx, inst in enumerate(pm.instruments):
        if inst.is_drum or not inst.notes:
            continue
        if route_track(inst, piece_name)['role'] == 'melody':   # forward the name
            out.append((idx, inst))
    return out


def skyline(inst):
    """Reduce a track to its top voice: at each instant keep the highest note.
    Monophonic in -> unchanged; polyphonic in -> top line only."""
    notes = [n for n in inst.notes if n.end > n.start]
    boundaries = sorted({n.start for n in notes} | {n.end for n in notes})

    top_line = []                                   # (start, end, pitch)
    for a, b in zip(boundaries, boundaries[1:]):
        sounding = [n for n in notes if n.start <= a and n.end >= b]
        if not sounding:
            continue
        top = max(sounding, key=lambda n: n.pitch)
        if top_line and top_line[-1][2] == top.pitch and abs(top_line[-1][1] - a) < 1e-6:
            top_line[-1] = (top_line[-1][0], b, top.pitch)   # extend held note
        else:
            top_line.append((a, b, top.pitch))

    out = pretty_midi.Instrument(program=inst.program)
    for start, end, pitch in top_line:
        out.notes.append(pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=end))
    return out


def intervals(inst):
    """Transposition-invariant signature: semitone jumps between consecutive notes."""
    notes = sorted(inst.notes, key=lambda n: n.start)
    pitches = [n.pitch for n in notes]
    return [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]


def best_similarity(a, b, max_shift=8):
    """Best match fraction between two signatures, sliding b against a by up to
    ±max_shift positions (catches doublings offset by a pickup or staggered start)."""
    best = 0.0
    for shift in range(-max_shift, max_shift + 1):
        matches = total = 0
        for i in range(len(a)):
            j = i - shift
            if 0 <= j < len(b):
                total += 1
                if a[i] == b[j]:
                    matches += 1
        if total:
            best = max(best, matches / total)
    return best


def harvest_piece(pm, piece_name=None):
    """Run the three gates in order; return survivors as (idx, reduced_inst).
    Order matters: skyline BEFORE length, so length judges the final line."""
    # 1. melody tracks -> 2. skyline each to a single line
    reduced = [(idx, skyline(inst)) for idx, inst in melody_tracks(pm, piece_name)]

    # 3. length gate on the REDUCED line
    long_enough = [(idx, inst) for idx, inst in reduced
                   if len(inst.notes) >= MIN_NOTES]

    # 4. de-dupe: keep a track only if it isn't a near-copy of one already kept
    survivors = []                                  # (idx, inst)
    for idx, inst in long_enough:
        sig = intervals(inst)
        if any(best_similarity(sig, intervals(kept_inst)) >= DUPE_THRESHOLD
               for _, kept_inst in survivors):
            continue
        survivors.append((idx, inst))
    return survivors


def write_melodies(survivors, piece_name, out_dir=CORPUS_DIR):
    """Write each surviving melody as its own .mid into the corpus folder."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for idx, inst in survivors:
        mel = pretty_midi.PrettyMIDI()
        mel.instruments.append(inst)
        path = os.path.join(out_dir, f"{piece_name}_idx{idx:02d}_melody.mid")
        mel.write(path)
        written.append(path)
    return written


# --- 4. RUNNER ------------------------------------------------------------
if __name__ == '__main__':
    PIECE = '/home/whamel/music-rag/midi/A Letter.mid'
    pm = pretty_midi.PrettyMIDI(PIECE)

    survivors = harvest_piece(pm)
    written   = write_melodies(survivors, 'aletter')

    print(f"wrote {len(written)} melody files to {CORPUS_DIR}:")
    for p in written:
        print(f"  {os.path.basename(p)}")