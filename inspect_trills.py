# inspect_trills.py -- is the short-note mass ornamental or scalar?
# CORPUS ONLY. Diagnostic, not a pipeline change.
from pathlib import Path
from collections import Counter
from features import load_notes

MAX_TRILL_INTERVAL = 2      # semitone/whole-tone alternation
MAX_RUN_IOI        = 0.20   # notes must be fast to count as a figure
MIN_RUN_NOTES      = 4      # a-b-a-b is the shortest credible trill
SHORT_DURATION     = 0.10   # "short note", from the percentile table


def fast_runs(notes, max_ioi=MAX_RUN_IOI, min_notes=MIN_RUN_NOTES):
    """Maximal spans of consecutive notes whose successive onsets are all fast."""
    spans, i, n = [], 0, len(notes)
    while i < n - 1:
        j = i
        while j < n - 1 and (notes[j + 1].start - notes[j].start) < max_ioi:
            j += 1
        if j - i + 1 >= min_notes:
            spans.append((i, j + 1))
        i = j + 1
    return spans


def classify_run(notes, lo, hi):
    """'trill' | 'tremolo' | 'scalar' | 'other' -- decided by pitch content."""
    p = [n.pitch for n in notes[lo:hi]]
    uniq = sorted(set(p))
    if len(uniq) == 2 and all(a != b for a, b in zip(p, p[1:])):
        return "trill" if uniq[1] - uniq[0] <= MAX_TRILL_INTERVAL else "tremolo"
    steps = [b - a for a, b in zip(p, p[1:])]
    if steps and all(0 < abs(s) <= 2 for s in steps) and len({s > 0 for s in steps}) == 1:
        return "scalar"
    return "other"


kinds = Counter()            # notes accounted for, by figure type
short_by_kind = Counter()    # SHORT notes only, by figure type
run_lengths = Counter()
total_notes = total_short = 0

for f in sorted(Path("corpus").glob("*.mid")):
    notes = load_notes(f)
    total_notes += len(notes)
    covered = {}
    for lo, hi in fast_runs(notes):
        kind = classify_run(notes, lo, hi)
        run_lengths[kind] += hi - lo
        for idx in range(lo, hi):
            covered[idx] = kind
    for idx, n in enumerate(notes):
        kind = covered.get(idx, "isolated")
        kinds[kind] += 1
        if n.end - n.start < SHORT_DURATION:
            total_short += 1
            short_by_kind[kind] += 1

print(f"corpus: {total_notes} notes, {total_short} short (<{SHORT_DURATION}s) "
      f"= {100*total_short/total_notes:.1f}%\n")
print("all notes by figure type:")
for k, c in kinds.most_common():
    print(f"  {k:9s} {c:7d}  ({100*c/total_notes:5.1f}%)")
print(f"\nSHORT notes only -- what explains the {total_short} of them:")
for k, c in short_by_kind.most_common():
    print(f"  {k:9s} {c:7d}  ({100*c/total_short:5.1f}% of short notes)")