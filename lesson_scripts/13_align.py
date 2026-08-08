# ==========================================================================
# 13_align.py — align the score roster to MIDI tracks, tag each with a role
# ==========================================================================

# --- 1. IMPORTS -----------------------------------------------------------
import statistics
import pretty_midi
from importlib import import_module

# pull reusable pieces from the earlier lesson-files (names start with a
# digit, so we can't use normal `import` — import_module handles that)
parse = import_module('10_parse_roster')
parse_rows     = parse.parse_rows
extract_roster = parse.extract_roster

route = import_module('12_route_families')
FAMILY_ROLE = route.FAMILY_ROLE

# --- 2. CONSTANTS ---------------------------------------------------------
BASS_REGISTER_MAX = 55          # median pitch <= this = bass role (MIDI 55 ≈ low G)

# families that can be EITHER melody or bass depending on register
ROLE_AMBIGUOUS = {'clarinet', 'saxophone', 'trombone', 'bassoon'}

# --- 3. FUNCTIONS ---------------------------------------------------------

def gm_family(program):
    """Coarse family from a GM program number — a FIRST guess, pre-roster."""
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
    return f'program_{program}'                        # catch-all (keyboards/perc)


def register_median(inst):
    """Median MIDI pitch of a track — robust 'how high does this sit' number."""
    pitches = [n.pitch for n in inst.notes]
    return statistics.median(pitches) if pitches else None


def align(roster, midi_families):
    """
    Walk the ROSTER as the spine; consume MIDI tracks to fill each family run.
    Pointer only advances when the family CHANGES (crosses a block boundary),
    so one roster entry can absorb many MIDI tracks (6 flutes -> 1 flute slot).
    Returns one dict per MIDI track carrying identity + default role.
    """
    result = []
    r = 0
    for m, mfam in enumerate(midi_families):
        # crossed into a new family? skip the rest of the old family's entries
        if r < len(roster) and roster[r]['family'] != mfam:
            while r < len(roster) and roster[r]['family'] != mfam:
                r += 1

        if r < len(roster) and roster[r]['family'] == mfam:
            rec  = roster[r]
            result.append({
                'midi_idx':  m,
                'gm_family': mfam,
                'identity':  rec['family'],
                'solo':      rec.get('solo', False),
                'qualifier': rec.get('qualifier'),
                'role':      FAMILY_ROLE.get(rec['family'], 'unknown'),
            })
        else:
            result.append({
                'midi_idx':  m, 'gm_family': mfam,
                'identity':  None, 'solo': False,
                'qualifier': None, 'role': 'unknown',
            })
    return result


def refine_role(family, default_role, inst):
    """Override role to 'bass' if a role-ambiguous family sits in bass register."""
    if family not in ROLE_AMBIGUOUS:
        return default_role
    med = register_median(inst)
    if med is not None and med <= BASS_REGISTER_MAX:
        return 'bass'
    return default_role


# --- 4. RUNNER ------------------------------------------------------------
if __name__ == '__main__':
    PDF   = '/home/whamel/music-rag/scores/A Letter - Full Score.pdf'
    PIECE = '/home/whamel/music-rag/midi/A Letter.mid'

    # roster from the score, minus unmatched/artifact rows
    roster = [r for r in parse_rows(extract_roster(PDF)) if r['family']]

    # playable MIDI tracks (skip drums + empties), keep original indices
    pm = pretty_midi.PrettyMIDI(PIECE)
    midi_tracks   = [(i, inst) for i, inst in enumerate(pm.instruments)
                     if not inst.is_drum and inst.notes]
    midi_families = [gm_family(inst.program) for _, inst in midi_tracks]

    # align (family + default role), then refine role by register
    tagged = align(roster, midi_families)

    print(f"{'idx':>3} {'prog':>4} {'family':<10} {'role':<9} {'median':>6}  flag")
    for t, (orig_idx, inst) in zip(tagged, midi_tracks):
        default_role = t['role']
        final_role   = refine_role(t['gm_family'], default_role, inst)
        med          = register_median(inst)
        changed      = '<-- reclassified' if final_role != default_role else ''
        print(f"{orig_idx:>3} {inst.program:>4} {t['gm_family']:<10} "
              f"{final_role:<9} {med:>6.1f}  {changed}")