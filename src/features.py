"""features.py — turn a melody chunk into a fixed-length feature vector."""

from collections import Counter
import numpy as np
import pretty_midi
import glob, os

MAX_INTERVAL = 12          # clamp leaps beyond an octave to +/-12
INTERVAL_BINS = list(range(-MAX_INTERVAL, MAX_INTERVAL + 1))   # 25 slots: -12..+12

# Duration bins, in seconds. Right-open: a note lands in the first bin whose
# upper edge it is strictly below. Roughly: 16th, 8th, dotted-8th, quarter,
# half, whole, longer.
DURATION_EDGES = [0.15, 0.3, 0.45, 0.7, 1.2, 2.5]              # -> 7 slots

CONTOUR_BINS = ["down", "same", "up"]                          # 3 slots

def load_notes(path):
    """MIDI file -> list of pretty_midi Notes, sorted by start time."""
    pm = pretty_midi.PrettyMIDI(str(path))
    assert len(pm.instruments) == 1, f"{path}: expected 1 instrument, got {len(pm.instruments)}"
    return sorted(pm.instruments[0].notes, key=lambda n: n.start)

def load_generated_notes(path, monophonic=True):
    """Read a generated midi, pool instruments, optionally enforce monophony.

    Pooling >1 instrument can create simultaneous notes that aren't melodic
    events. If monophonic, keep the highest-pitched note at each onset (skyline)
    and trim overlaps so downstream interval/rhythm metrics see a real line.
    """
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = sorted((n for inst in pm.instruments for n in inst.notes),
                   key=lambda n: (n.start, -n.pitch))     # earliest first, highest first
    if not monophonic:
        return notes

    out = []
    for n in notes:
        if out and n.start - out[-1].start < 1e-3:        # same onset -> already took the top
            continue
        if out and n.start < out[-1].end:                 # overlap -> trim the previous note
            out[-1] = pretty_midi.Note(velocity=out[-1].velocity, pitch=out[-1].pitch,
                                       start=out[-1].start, end=n.start)
        out.append(n)
    return out

def clip_and_rezero(notes, t):
    """Keep notes starting strictly after t; shift times so the first note is ~t=0.

    Used to strip the prompt (retrieved prefix + seed) off a generation, so that
    System A and System B outputs live on the same timeline.
    """
    kept = [n for n in notes if n.start > t]
    if not kept:
        return []
    offset = kept[0].start
    out = []
    for n in kept:
        out.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                    start=n.start - offset, end=n.end - offset))
    return out

def window(notes, seconds):
    """Keep notes that START before `seconds`; cap any sustain at the window edge.

    The single source of truth for time-windowing a melody. Clips onsets AND
    durations, so a slow sustained note can't drag a windowed phrase past its
    nominal length. (This is the fast-vs-slow bias fix — every onset-only cut
    silently keeps long tails.)
    """
    out = []
    for n in notes:
        if n.start >= seconds:
            continue
        end = min(n.end, seconds)
        if end <= n.start:                 # degenerate: a note starting AT the edge
            continue
        out.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                    start=n.start, end=end))
    return out

def interval_counts(notes):
    """Signed semitone intervals between consecutive notes, clamped to +/-MAX_INTERVAL."""
    if len(notes) < 2:
        raise ValueError(f"need >=2 notes for intervals, got {len(notes)}")
    pitches = [n.pitch for n in notes]
    c = Counter()
    for a, b in zip(pitches, pitches[1:]):
        iv = b - a
        iv = max(-MAX_INTERVAL, min(iv, MAX_INTERVAL))     # two-sided clamp
        c[iv] += 1
    return c


def duration_counts(notes):
    """Note durations, bucketed by DURATION_EDGES."""
    if not notes:
        raise ValueError("need >=1 note for durations")
    c = Counter()
    for n in notes:
        dur = n.end - n.start
        # np.searchsorted finds the index where `dur` would slot into the edges:
        # dur=0.1 -> 0, dur=0.2 -> 1, dur=3.0 -> 6.  Exactly the bin index.
        c[int(np.searchsorted(DURATION_EDGES, dur))] += 1
    return c


def contour_counts(notes):
    """How many intervals go down / stay / go up."""
    if len(notes) < 2:
        raise ValueError(f"need >=2 notes for contour, got {len(notes)}")
    pitches = [n.pitch for n in notes]
    c = Counter()
    for a, b in zip(pitches, pitches[1:]):
        c["up" if b > a else ("down" if b < a else "same")] += 1
    return c

DURATION_BINS = list(range(len(DURATION_EDGES) + 1))   # 7 slots: 0..6

def _project(counter, bins):
    """Counter -> numpy array over a fixed bin order, normalized to sum 1."""
    v = np.array([counter.get(b, 0) for b in bins], dtype=float)
    total = v.sum()
    if total == 0:
        raise ValueError("empty counter — cannot normalize")
    return v / total


def feature_vector(notes, w_interval=1.0, w_duration=1.0, w_contour=0.5):
    """Melody notes -> a single normalized 35-dim feature vector."""
    iv = _project(interval_counts(notes), INTERVAL_BINS) * w_interval
    du = _project(duration_counts(notes), DURATION_BINS) * w_duration
    co = _project(contour_counts(notes), CONTOUR_BINS)  * w_contour
    return np.concatenate([iv, du, co])

def cosine(a, B):
    """Similarity of one vector `a` against every row of matrix `B`.

    a: (35,)      the seed
    B: (N, 35)    the index — one row per corpus chunk
    returns: (N,) similarity in [0, 1], one score per chunk
    """
    a_norm = a / np.linalg.norm(a)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return B_norm @ a_norm

from pathlib import Path

def piece_name(path):
    """Filename -> piece key. THE single source of truth for piece identity.

    'a_breeze_through_the_willows_idx00_melody_chunk00_t+00.mid'
        -> 'a_breeze_through_the_willows'

    Accepts a str or Path. Uses only the FILENAME, never the directory,
    so 'data/train/foo_idx00.mid' and 'foo_idx00.mid' give the same key.
    """
    name = Path(path).name
    head, sep, _ = name.partition("_idx")
    if not sep:
        raise ValueError(f"no '_idx' landmark in filename: {name!r}")
    return head


if __name__ == "__main__":
    # self-test — runs ONLY when you type `python features.py`,
    # never when another file imports this one.
    from pathlib import Path
    f = sorted(Path("data/train").glob("*_t+00.mid"))[0]
    notes = load_notes(f)
    print(f.name, len(notes), "notes")
    print("intervals:", interval_counts(notes))