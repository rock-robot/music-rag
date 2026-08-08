# Which cascade stage does each instrument family feed?
# This is the ONE place the melody/harmony/bass decision lives.
# v1 harvests MELODY; later stages flip to HARMONY, then BASS.

FAMILY_ROLE = {
    # --- melody-register winds: v1 harvests these ---
    'piccolo':            'melody',
    'flute':              'melody',
    'oboe':               'melody',
    'english horn':       'melody',
    'clarinet':           'melody',
    'alto clarinet':      'melody',
    'saxophone':          'melody',
    'soprano saxophone':  'melody',
    'alto saxophone':     'melody',
    'tenor saxophone':    'melody',
    'trumpet':            'melody',
    'horn':               'melody',

    # --- bass voices: the cascade's bass stage ---
    'contrabassoon':      'bass',
    'bassoon':            'melody',   # 1st chair carries tenor lines; heuristics prune bass passages
    'bass clarinet':      'bass',
    'contrabass clarinet':'bass',
    'baritone saxophone': 'bass',
    'bass trombone':      'bass',
    'trombone':           'melody',   # same logic — register test rejects the bass ones
    'euphonium':          'bass',
    'tuba':               'bass',
    'double bass':        'bass',

    # --- harmony / keyboard: the harmony stage ---
    'harp':               'harmony',
    'piano':              'harmony',
    'organ':              'harmony',

    # --- percussion: excluded from pitched stages for now ---
    'timpani':            'percussion',
    'vibraphone':         'percussion',
    'xylophone':          'percussion',
    'glockenspiel':       'percussion',
    'crotales':           'percussion',
    'suspended cymbal':   'percussion',
    'tam-tam':            'percussion',
    'strings':  'melody',      # default melody; register flips low ones to bass
    'harp':     'harmony',     # chordal accompaniment — kept out of melody pool
}

# Piece-level role overrides: for specific pieces where an instrument plays a
# role different from its usual one. Keyed by the cleaned piece_name().
# Each entry forces a family to a role, ONLY for that piece.
PIECE_OVERRIDES = {
    'concerto_for_tuba2': {'tuba': 'melody'},   # tuba is the SOLOIST here, not bass
    'concerto_no_1':      {'tuba': 'melody'},   # (confirm the soloist — see below)
}

# assumes you have, from earlier scripts:
#   parse_rows(extract_roster(PDF))  -> ordered roster records (family per staff)
#   pretty_midi track list           -> ordered MIDI tracks
# For now we test the ROLE lookup in isolation, before wiring in alignment.
ROLE_AMBIGUOUS = {'clarinet', 'saxophone', 'trombone', 'bassoon', 'strings'}
def role_for_family(family):
    """Coarse-route a family name to its cascade stage."""
    return FAMILY_ROLE.get(family, 'unknown')

if __name__ == '__main__':
    # quick sanity test on the families we know from A Letter's roster
    test_families = ['piccolo', 'flute', 'bassoon', 'tuba', 'piano',
                     'timpani', 'trumpet', 'double bass', 'harp']
    for fam in test_families:
        print(f"  {fam:<14} -> {role_for_family(fam)}")