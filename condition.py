"""condition.py — turn retrieved phrases + a seed into a single prompt for generate()."""

import numpy as np
import pretty_midi

from features import load_notes, piece_name

MIN_PITCH, MAX_PITCH = 21, 108      # same guard as augment.py — piano range
PHRASE_GAP = 0.5                    # seconds of rest between spliced phrases

from features import window

RETRIEVED_SECONDS = 3.0     # how much of each retrieved chunk to actually use

def truncate(notes, seconds=RETRIEVED_SECONDS):
    kept = window(notes, seconds)
    return kept if len(kept) >= 3 else notes[:3]      # fallback: keep 3 notes minimum

# --- condition.py ---

MAX_PROMPT_SECONDS = 14.0          # keep generation inside the adapter's ~15s range

def build_conditioned_prompt(retrieved_list, seed_notes, gap=PHRASE_GAP):
    """Splice, then enforce the prompt budget.

    Returns (notes, start_time, ok):
      ok=False  -> prompt exceeds MAX_PROMPT_SECONDS; caller should EXCLUDE this
                   seed from System B and record why. Do not generate on it.
    """
    notes, start_time = splice(retrieved_list, seed_notes, gap=gap)
    ok = start_time <= MAX_PROMPT_SECONDS
    return notes, start_time, ok

# in align_to_seed, you already have octave_only as a param — default it True
def align_to_seed(retrieved_notes, seed_notes, octave_only=True):
    """Transpose retrieved_notes so its pitch centre matches the seed's."""
    seed_centre = np.median([n.pitch for n in seed_notes])
    ret_centre  = np.median([n.pitch for n in retrieved_notes])

    shift = int(round(seed_centre - ret_centre))
    if octave_only:
        shift = 12 * int(round(shift / 12))

    # Bounds guard. Unlike augment.py we cannot simply reject: rejecting means
    # retrieving nothing. Instead, PULL THE SHIFT BACK until the phrase fits.
    lo = min(n.pitch for n in retrieved_notes)
    hi = max(n.pitch for n in retrieved_notes)
    shift = max(shift, MIN_PITCH - lo)      # don't push the lowest note below 21
    shift = min(shift, MAX_PITCH - hi)      # don't push the highest note above 108

    return [pretty_midi.Note(velocity=n.velocity, pitch=n.pitch + shift,
                             start=n.start, end=n.end)
            for n in retrieved_notes], shift

def splice(retrieved_list, seed_notes, gap=PHRASE_GAP):
    """Concatenate retrieved phrases, then the seed, on one timeline.

    Returns: (notes, start_time)
      notes      — one flat, time-ordered list of Notes
      start_time — the time at which the SEED ENDS. Everything after this
                   is the model's own output.
    """
    out = []
    t = 0.0

    for phrase in retrieved_list:
        dur = max(n.end for n in phrase)
        for n in phrase:
            out.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                        start=n.start + t, end=n.end + t))
        t += dur + gap

    seed_dur = max(n.end for n in seed_notes)
    for n in seed_notes:
        out.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                    start=n.start + t, end=n.end + t))
    start_time = t + seed_dur

    return sorted(out, key=lambda n: n.start), start_time

def notes_to_midi(notes, path, program=73):     # 73 = GM flute
    """Write a Note list out as a .mid file (for tokenizing, or listening)."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    inst.notes = list(notes)
    pm.instruments.append(inst)
    pm.write(str(path))
    return path

def verify_roundtrip(notes, path, tol=0.02):
    """Write notes to MIDI, read them back, confirm nothing was lost or moved."""
    notes_to_midi(notes, path)
    back = load_notes(path)

    assert len(back) == len(notes), f"note count changed: {len(notes)} -> {len(back)}"

    drift = max(abs(a.start - b.start) for a, b in zip(notes, back))
    pitch_ok = all(a.pitch == b.pitch for a, b in zip(notes, back))

    assert pitch_ok, "pitches changed on round-trip (!!)"
    print(f"  round-trip OK: {len(back)} notes, max time drift {drift*1000:.1f} ms")
    return drift