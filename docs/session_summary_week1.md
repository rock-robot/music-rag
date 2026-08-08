# Session Summary — Week 1 (ML Foundations + The Data Pipeline)

*A narrative of how we got from "I'm in Week 1, what's next?" to a complete, training-ready dataset: 66,122 augmented melody examples, honestly split by piece. Like the Week 0 summary, this focuses on the reasoning behind each decision, not the line-by-line debugging.*

---

## 1. First inference — calibrating the ear before training

We began where the plan pointed: running the pretrained Anticipatory Music Transformer (`stanford-crfm/music-small-800k`) *before* touching training, so we'd have a "before" picture to judge later changes against. Two runs taught the key lesson:

- **Unconditional generation** (from silence) came out chaotic — "the drums and violin got stuck in triplet hell." Two failure modes on display: no *anchor* (generating from nothing is the hardest case), and the *repetition trap* (autoregressive degeneration, where the model feeds its own loop back to itself).
- **Seeded continuation** (feed 5 seconds of a real phrase, ask it to continue) was "much more solid... relatively human." Same model, same sampling — the only change was giving it a real phrase to hold onto.

The takeaway that carried through the whole week: **a good prompt collapses the model into a far more coherent space than starting from silence.** And listening now *calibrated the ear* — we learned what "the raw model" sounds like, so later we can tell "this model being this model" from "something I broke." That was the milestone: *understand what good and bad sound like before training.*

A mental model locked in here: this MIDI model is a **GPT** — a next-token predictor — identical in architecture to a text LLM. It just predicts the next *musical event* instead of the next word-piece. `midi_to_events` is the translator between MIDI and the model's token language; `ops.clip` grabs a seed; `generate(..., inputs=history)` continues it.

## 2. The data pipeline — the shape of the problem

The rest of the week was the plan's data-pipeline task: *MIDI → melody lines → chunks → augmented → split.* The central difficulty is that the catalog is **dense ensemble music** (24–41 parts per piece), but v1 is a **melody model** that trains on single monophonic lines. So the pipeline's job is to reliably pull clean, single-voice melodies out of 40-part scores — and do it automatically across ~50 pieces.

We built this in five stages, testing each on **A Letter** (a 41-staff wind ensemble, our hardest case) before scaling:

1. **Routing** — tag every MIDI track as melody / bass / harmony / unknown.
2. **Harvest** — turn melody-role tracks into clean monophonic files, through three gates.
3. **Batch harvest** — run it across all pieces.
4. **Chunk** — slice melodies into phrase-length examples.
5. **Augment + split** — multiply by transposition, divide honestly.

## 3. Routing — and a major design pivot

The first, biggest sub-problem: *which tracks are melodies?* GM program numbers give a coarse first guess, but they lie — Sibelius collapses four different clarinets (Eb, Bb, alto, bass) all onto program 71, and alto/bass flutes onto program 73. So the program number alone can't tell a melody clarinet from a bass clarinet.

**The roster detour (built, then shelved).** We spent several sessions building a *score-roster extractor*: parsing each score PDF's first system with `pdfplumber` to recover the true instrument names (the score is the only place the Eb-vs-bass-clarinet distinction survives). It worked — we automatically reconstructed A Letter's full 41-instrument roster from the PDF, using an elegant trick: filter left-margin words by x-position, group fragments by their `top` coordinate to reassemble shattered labels, then parse against an instrument vocabulary. We even cross-validated it against the MIDI track order and confirmed they align.

**Then we pivoted.** When we tried to *align* the roster to the MIDI tracks (a forward-walking pointer), an out-of-order muted-trumpet track (idx 28, wedged inside the trombones) *desynced the pointer permanently* — every track after it got mislabeled. The forward-only walk assumed the MIDI's family order was a clean subsequence of the roster's; interleaved brass broke that assumption.

The fix reframed the whole approach. We asked: *do we even need the roster's order?* For a **melody model** that pools all flutes together, we don't need to know "which exact track is the alto flute" — we only need each track's **family → role**. So we switched to **Option B: order-independent, per-track routing** — look up `gm_family(program)` → `FAMILY_ROLE[family]`, refine by register. This is *stateless* (each track routed in isolation), so it's structurally immune to the interleaving bug: a misplaced trumpet just looks up "trumpet → melody" regardless of position.

The roster extractor wasn't wasted — it's **shelved infrastructure for the cascade** (the harmony/bass stages will want the fine identities it recovers). v1 simply doesn't need it.

**Register as the tie-breaker.** Some families play *both* roles depending on pitch (clarinet, sax, trombone, bassoon, strings). We handle these with a `ROLE_AMBIGUOUS` set + a register threshold (`BASS_REGISTER_MAX = 55`): the family *starts* in melody, but any track whose median pitch sits at/below 55 flips to `bass`. On A Letter's six program-71 clarinet tracks, this produced a clean register staircase (A#5 → B1) and correctly flipped the bottom two (bass, contrabass) to bass while keeping the upper four as melody.

**Strings (caught by the composer).** A Letter is wind ensemble, so we'd built and tuned everything with *no violins*. The student caught that orchestral pieces would have their violin melodies **silently dropped** — `gm_family` didn't recognize string programs, so they'd fall through to `unknown` and never harvest. We added strings (programs 40–45, 48, 49 → `strings`, role-ambiguous so register sorts violins-to-melody, cellos-to-bass) and kept harp (46) separate → `harmony` (it's chordal, not a bowed line, and skyline would turn its chords into jagged nonsense). Validated on **Paradise Lost Mov 1**: violins → melody, cellos/basses → bass, harp → harmony. All correct.

**Piece-level overrides (tuba concertos).** Two pieces flagged "no melody." Diagnosis: they're *tuba concertos* — the soloist is a tuba, which routes to `bass` by family. The melody is a tuba. We added `PIECE_OVERRIDES` — a per-piece patch (`concerto_for_tuba2`, `concerto_no_1` → force tuba to melody) — chosen deliberately over a global "tuba can be melody" rule, which would corrupt the 44 pieces where tuba is correctly the bass. The override is a *fact about the piece*, not about tubas.

## 4. The three harvest gates

Routing answers *which tracks*; the harvest turns those tracks into clean corpus files through three gates, tested exhaustively on A Letter (which routed 26 tracks to melody):

- **Gate 1 — minimum length** (`MIN_NOTES = 25`): drops fragment tracks (a 5-note stab, a 19-note cue).
- **Gate 2 — skyline**: reduces any track to its top voice (highest note at each instant). A no-op on already-monophonic lines; on a divisi/harmonized track it extracts the melodic top and discards the harmony underneath. This is what guarantees the corpus is genuinely monophonic — it caught the trumpets, horns, and harmonized clarinets that the router waved through as "melody" but were secretly two-voice.
- **Gate 3 — de-duplication**: drops near-copies (an octave-doubled line, a section brass unison). This is the subtle one: it compares tracks by a **transposition-invariant signature** (the sequence of intervals — the *shape*, not the absolute pitches, since a piccolo doubling a flute an octave up is the *same melody*), and the comparison is **alignment-tolerant** (it slides one sequence against the other to catch doublings offset by a pickup). Threshold `0.80`.

**Gate ordering matters:** skyline runs *before* the length gate, so length judges the *final* monophonic line — this caught idx 24, which passed length raw (37 notes) but skyline gutted it to 13.

**A real discovery.** We'd assumed since Week 0 that A Letter's piccolo and flutes doubled each other. The signature comparison **falsified it** — their interval sequences were completely different (18% similar, not ~100%). Your writing uses *independent contrapuntal lines*, not octave doublings. This mattered: had we built de-dupe on the *assumption* of doublings, we'd have deleted genuine distinct melodies. De-dupe by *measurement*, not by hunch. (We kept the gate anyway — the cost of a missed doubling is asymmetric: it would contaminate train/val, corrupting the honesty of evaluation. A rarely-firing safety net is worth it.)

## 5. From one piece to the whole catalog

**Batch harvest** ran the pipeline across the catalog with the error-isolation a batch needs: `try/except` per piece (one bad file can't kill the run), *flag-don't-write* for pieces yielding no melody, and an end-of-run report. A "list before you write" discovery step caught **51 files where 49 were expected** — two working files (`avanti_melody`, `flutesolo1`) had strayed into the `midi/` folder. An `EXCLUDE` set handles those plus the deliberately-dropped `ex_machina`.

Result: **462 clean melody files from 46 pieces.** Four pieces flagged; we resolved all four — 2 tuba concertos got overrides (1 file each), 2 piano solos (`album`, `prelude_in_c#_minor`) were scoped *out* of v1 (piano-solo melody extraction — separating melody from accompaniment in a two-hand texture — is a genuinely harder problem than skyline can handle; documented as future work).

## 6. Chunking, augmentation, and the honest split

**Chunking** (`chunk.py`): slice each melody into overlapping 15-second windows (7.5s step = 50% overlap, so any phrase is captured whole in some window), drop sparse windows (`MIN_CHUNK_NOTES = 8`), and — crucially — **shift each chunk's timestamps to start at 0**, so a chunk pulled from 600s into a melody isn't 600 seconds of silence followed by a few notes. Result: **462 melodies → 5,515 chunks.** This hit the plan's §5 promise: "turn 100 pieces into several thousand training examples."

**Augmentation** (`augment.py`): transpose every chunk into all 12 keys, using *centered* shifts (−6 to +5) rather than 0-to-+11 — same key coverage, but half the register displacement, keeping notes musically plausible. A bounds guard (pitch 21–108) rejects any transposition that would push a note out of range (it rejects the *whole* shift rather than clamping individual notes, which would distort the melody's shape). Result: **5,515 chunks → 66,122 files** (58 transpositions rejected by the bounds guard — 0.09%, confirming almost all melodies sit in comfortable mid-register). Conceptual keystone: transposition doesn't create *new* melodies — it's the *same* shape in 12 keys, which teaches the model **key-invariance** (it's forced to treat the interval shape as the signal and absolute pitch as noise, because only the shape is constant across the 12 copies).

**Split** (`split.py`): the plan's most important stage for research integrity. We hold out **~15% of whole pieces** (not chunks) for validation, so no piece's material straddles the train/val divide — and *all* transpositions of *all* chunks of a piece move together. Selection is **random with a fixed seed (42)** for reproducibility, which Week 3's System-A-vs-B comparison depends on (a shifting validation set would make comparisons meaningless). The 7 held-out pieces: `a_letter, chorale_and_procession, concerto, el_mar, elegy, forest_overture, spanish_sunrise`.

Result: **42,967 train + 23,155 val files.** Note val is **35% of files** despite being **15% of pieces** — because several *file-heavy* pieces (A Letter has 22 melody lines) landed in the holdout. The honest framing for the paper: *"held out 7 of 46 pieces."* Your true statistical power is **7 pieces**, not 23,155 files — the files are highly correlated chunks-and-transpositions of those 7 sources.

## 7. Recurring engineering lessons

Several lessons showed up more than once — worth internalizing:

- **Single source of truth for names.** A hardcoded `'aletter'` in an early test collided with the batch's computed `'a_letter'`, silently doubling A Letter's files (write-to-new-name doesn't overwrite, it accumulates). A value that's *also* computed somewhere should never be typed by hand.
- **Tests must not write into pipeline folders.** A chunk-writer test dropped `test_chunk*.mid` into `chunks/`; the augmenter then transposed them into 36 stray files that rode all the way to the split. Tests belong in throwaway locations, never in a directory a later stage reads as input.
- **List before you write.** The discovery-and-count step caught the stray files, the naming collision, *and* the 51-vs-49 mismatch — every time, before any damage.
- **Refactor preserves behavior.** When we renamed the numbered lesson-scripts (`12_route_families.py` → `routing.py`, etc.) into a clean importable pipeline, the test was that A Letter still produced *exactly 22 files*. Same output before and after = reorganized, not broken.
- **Imports look only in X.** `from routing import ROLE_AMBIGUOUS` failed because that name lived in `router.py`, not `routing.py` — Python doesn't search all files for a name, only the one you asked.

---

## Where things stand

**Week 1 is complete — and the meatiest part (the full data pipeline) is done.** From 46 dense scores: 462 melody lines → 5,515 chunks → 66,122 transposed examples → an honest 42,967/23,155 by-piece train/val split. Every stage was built from a probe up, understood, and validated on the two hardest test cases (dense winds, full orchestra).

**Pipeline modules:** `routing.py` (family→role table, overrides), `router.py` (gm_family + route_track), `harvest.py` (three gates + writer), `batch_harvest.py`, `chunk.py`, `batch_chunk.py`, `augment.py`, `split.py`.

**Provisional values flagged for revisiting:** `DUPE_THRESHOLD = 0.80` (validated only on the "independent" end — never yet tested against a *confirmed* doubling), `MIN_CHUNK_NOTES = 8` (has a fast-vs-slow bias — could drop legitimately slow, sustained phrases), and the 35%-of-files validation size (a different seed could land closer to 15% of files if Week 2 feels data-starved).

## The bridge to Week 2

One item from the plan's Week 1 list remains, and it's the natural first step of fine-tuning: **`MIDI → events` tokenization.** The 66,122 files are `.mid`, not yet in the AMT's event-token language. The data is *shaped* right (short, monophonic, augmented, split) — it just needs translating into tokens the model can train on. That's where Week 2 opens.

## Next session (Week 2 — fine-tune the baseline, System A)

Tokenize the corpus with `midi_to_events`, set up a Hugging Face `transformers` training loop (recall the Week 0 note: the `anticipation` package does prep + inference only, *not* training), fine-tune `music-small-800k` on the training set while watching train-vs-val loss for overfitting, then generate continuations from held-out seeds and run the objective metrics + the transposition-aware novelty check. Milestone: *System A produces stylistically plausible, novel continuations, and you have baseline metric numbers.*
