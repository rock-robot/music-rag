"""
metric_histogram.py — melodic-interval style comparison (filtered).
Corpus (real melodies) vs. pretrained vs. LoRA generations.
Drops collapsed generations (repetition-trap runs) before aggregating.
"""
import pretty_midi
from pathlib import Path
from collections import Counter

# --- config ---
CORPUS_DIR    = "corpus"          # 462 original melodies (not chunked/transposed)
PRE_DIR       = "gen_pretrained"  # 20 pretrained generations
LORA_DIR      = "gen_lora"        # 20 LoRA generations
PROMPT_SECS   = 5                 # strip the shared seed prompt from generations
MAX_INTERVAL  = 12
# collapse-detection thresholds
MAX_NOTES     = 200               # a 10s monophonic clip can't legitimately have this many
MAX_ZERO_FRAC = 0.40              # >40% repeated-notes = degenerate run


def interval_counts(midi_path, start_after=0.0):
    """Count absolute melodic intervals between consecutive notes in instrument 0."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = sorted((n for n in pm.instruments[0].notes if n.start >= start_after),
                   key=lambda n: n.start)
    pitches = [n.pitch for n in notes]
    c = Counter()
    for a, b in zip(pitches, pitches[1:]):
        c[min(abs(b - a), MAX_INTERVAL)] += 1
    return c


def histogram(counts):
    """Normalize raw counts into a probability distribution (sums to 1)."""
    total = sum(counts.values()) or 1
    return [counts.get(i, 0) / total for i in range(MAX_INTERVAL + 1)]


def is_collapsed(midi_path, start_after=0.0):
    """True if a generation is a degenerate repetition-trap run."""
    c = interval_counts(midi_path, start_after=start_after)
    total = sum(c.values()) or 1
    return (total > MAX_NOTES) or (c.get(0, 0) / total > MAX_ZERO_FRAC)


def aggregate_corpus(folder):
    """Real corpus: whole melodies, no prompt to strip, no collapse filtering."""
    files = sorted(Path(folder).glob("*.mid"))
    total = Counter()
    for f in files:
        total += interval_counts(f)
    print(f"corpus: {len(files)} melodies, {sum(total.values())} intervals")
    return histogram(total)


def aggregate_clean(folder, start_after=PROMPT_SECS):
    """Generations: strip prompt, DROP collapsed runs, aggregate the rest."""
    files = sorted(Path(folder).glob("*.mid"))
    kept_counts, dropped = Counter(), []
    for f in files:
        if is_collapsed(f, start_after=start_after):
            dropped.append(f.name)
        else:
            kept_counts += interval_counts(f, start_after=start_after)
    n_kept = len(files) - len(dropped)
    print(f"{folder}: kept {n_kept}/{len(files)}  "
          f"({sum(kept_counts.values())} intervals)")
    if dropped:
        print(f"    dropped (collapsed): {dropped}")
    return histogram(kept_counts)


# --- run ---
corpus_hist = aggregate_corpus(CORPUS_DIR)
pre_hist    = aggregate_clean(PRE_DIR)
lora_hist   = aggregate_clean(LORA_DIR)

print("\ninterval :  corpus  pretrained  LoRA")
for i in range(MAX_INTERVAL + 1):
    label = f"{i}" if i < MAX_INTERVAL else f"{i}+"
    print(f"  {label:>6} :  {corpus_hist[i]:.3f}    {pre_hist[i]:.3f}     {lora_hist[i]:.3f}")