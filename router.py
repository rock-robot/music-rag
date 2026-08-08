import statistics
import pretty_midi
from importlib import import_module

from routing import FAMILY_ROLE, ROLE_AMBIGUOUS, PIECE_OVERRIDES

BASS_REGISTER_MAX = 55

def register_median(inst):
    """Median MIDI pitch of a track — robust 'how high does this sit' number."""
    pitches = [n.pitch for n in inst.notes]
    return statistics.median(pitches) if pitches else None

def gm_family(program):
    """Coarse family from a GM program number."""
    if program in (72, 73):   return 'flute'          # piccolo + flutes
    if program == 82:         return 'flute'          # calliope = your lead flute
    if program == 68:         return 'oboe'
    if program == 69:         return 'english horn'
    if program == 70:         return 'bassoon'
    if program == 71:         return 'clarinet'
    if 64 <= program <= 67:   return 'saxophone'
    if program in (56, 59):   return 'trumpet'
    if program == 60:         return 'horn'
    if program == 57:         return 'trombone'
    if program == 58:         return 'tuba'
    if 40 <= program <= 45:   return 'strings'        # violin/viola/cello/bass/tremolo/pizz
    if program == 46:         return 'harp'           # plucked, chordal -> harmony, NOT bowed strings
    if program in (48, 49):   return 'strings'        # string ensemble sections
    return f'program_{program}'                        # catch-all (keyboards/perc)

def route_track(inst, piece_name=None):
    fam = gm_family(inst.program)

    # piece-level override wins if this piece names this family
    if piece_name and piece_name in PIECE_OVERRIDES:
        override = PIECE_OVERRIDES[piece_name].get(fam)
        if override:
            return {'program': inst.program, 'family': fam,
                    'role': override, 'median': register_median(inst)}

    # otherwise, normal routing
    role = FAMILY_ROLE.get(fam, 'unknown')
    med  = register_median(inst)
    if fam in ROLE_AMBIGUOUS and med is not None and med <= BASS_REGISTER_MAX:
        role = 'bass'
    return {'program': inst.program, 'family': fam, 'role': role, 'median': med}

if __name__ == '__main__':
    PIECE = '/home/whamel/music-rag/midi/Paradise Lost_ Mov 1.mid'
    pm = pretty_midi.PrettyMIDI(PIECE)

    print(f"{'idx':>3} {'prog':>4} {'family':<12} {'role':<9} {'median':>6}")
    for idx, inst in enumerate(pm.instruments):
        if inst.is_drum or not inst.notes:
            continue
        r = route_track(inst)
        print(f"{idx:>3} {r['program']:>4} {r['family']:<12} "
              f"{r['role']:<9} {r['median']:>6.1f}")