"""presentation.py — rebuild seed -> pause -> continuation clips for listening.

The full generated stream was overwritten in Week 3, but it is reconstructible:
seed notes are deterministic (seeds.py) and the clipped continuation is on disk
re-zeroed to t=0. The retrieved/random PROMPT is deliberately NOT included --
System A has none, so playing it would reveal the condition.
"""
import pretty_midi
from pathlib import Path

from features import load_notes
from seeds import load_seed

PAUSE_SECONDS = 1.5
PROGRAM       = 73        # flute; identical across every clip
VELOCITY      = 80        # flattened, so loudness can't cue the boundary
OUT           = Path("listening/midi")

MELODY_PROGRAMS = [(73,"flute",60,96), (68,"oboe",58,91),
                   (71,"clarinet",50,91), (56,"trumpet",55,86)]
MAX_EXCURSION = 4      # semitones a stray note may sit outside the patch range

def instrument_for(seed_name, seed_list, notes=None):
    order = sorted(seed_list)
    start = order.index(seed_name) % len(MELODY_PROGRAMS)
    if not notes:
        prog, name, _, _ = MELODY_PROGRAMS[start]
        return prog, name
    ps = [n.pitch for n in notes]
    for step in range(len(MELODY_PROGRAMS)):
        prog, name, lo, hi = MELODY_PROGRAMS[(start + step) % len(MELODY_PROGRAMS)]
        out = [p for p in ps if p < lo or p > hi]
        # Both tests matter: a few notes slightly outside is inaudible, a few notes
        # far outside is the horn failure -- counting alone can't tell them apart.
        if len(out) / len(ps) <= 0.05 and all(
                min(abs(p - lo), abs(p - hi)) <= MAX_EXCURSION for p in out):
            return prog, name
    return 73, "flute"          # widest top end

def assemble(seed_notes, cont_notes, pause=PAUSE_SECONDS):
    """seed (as written) + silence + continuation. Returns (notes, boundary_time)."""
    seed_end = max(n.end for n in seed_notes)
    boundary = seed_end + pause
    out = [pretty_midi.Note(VELOCITY, n.pitch, n.start, n.end) for n in seed_notes]
    out += [pretty_midi.Note(VELOCITY, n.pitch,
                             n.start + boundary, n.end + boundary)
            for n in cont_notes]
    return out, boundary


def write(notes, path, program=PROGRAM):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    inst.notes = list(notes)
    pm.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))
