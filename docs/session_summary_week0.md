# Session Summary — Week 0 (Scope + Setup)

*A narrative of how we got from "I'm in Week 0, what's first?" to a fully working environment and a locked v1 scope. Focuses on the reasoning behind each major decision, not the step-by-step debugging.*

---

## 1. Exporting the catalog from Sibelius

We started where the plan said to: getting the music out of Sibelius before touching any code. The guiding principle that recurred all session was **consistency across files prevents silent data corruption** — the dangerous kind of bug that never throws an error, it just quietly makes your data wrong.

Each export setting got justified on that basis:
- **Two formats** — `.mid` to train on (the model's native, time-ordered event format), `.mxl` as a richer archive that preserves notation detail MIDI discards. Train on MIDI not because MusicXML is "too detailed," but because MIDI is the format the tokenizer eats and is already shaped as a sequence.
- **Unfold repeats ON** — so the MIDI reflects the music as it sounds in time, and so repeat structures don't get represented inconsistently across pieces.
- **Type 1 / PPQ 960 / General MIDI / one file per piece / all instruments kept** — each chosen to preserve structure and keep every file uniform.

Pieces under ~20 seconds were set aside (not deleted) — too short to be real training phrases, but useful later as seed prompts.

## 2. Inventorying the data — and a plot twist

The first inventory (`music21`) gave counts, keys, and tempos, but reported almost every instrument as "Unknown." That gap mattered, because the instrument breakdown was exactly what we needed to decide v1's scope. So we ran a **second inventory with `pretty_midi`**, which reads General MIDI program numbers directly.

That second pass changed the picture:
- **49 pieces** — the bottom of the planned range, which promoted the "small data" mitigations from optional to load-bearing.
- The catalog is **large wind/orchestral ensemble music** (mostly 24–41 parts), but the high part counts turned out to be **section divisi** (e.g. ~3 clarinet parts), not 30 unrelated instruments. A consistent wind/brass core appears in ~80% of pieces — a real stylistic fingerprint.

## 3. The v1 scope decision

The data created a tension: the **hardest** version of the task (full multi-track, which requires learning how parts fit together vertically) combined with the **least** data (49 pieces). Modeling all parts jointly was off the table.

The clean escape — "just train on a solo instrument" — wasn't directly available either, since only ~5 pieces are truly solo. The resolution: **extract one consistent melodic voice (the top woodwind line — Flute/Piccolo) from every piece.** That turns 49 orchestral scores into 49 examples of the same kind of voice, collapsing the hard joint-modeling problem into the manageable "melody continuation" task AMT already does well.

A useful lens emerged for handling outliers: for a **melody** model, the inclusion test is "does it have a usable melodic line?" — not "is it a good piece?" On that basis we dropped **Ex Machina** (electronic, no flute) and kept **Jazz** and **Ohio River Cruise** pending what the extractor actually pulls out.

## 4. From "v1" to "stage one of something bigger"

A key idea surfaced mid-session: rather than v1 being a stripped-down simplification, treat it as **the first stage of a conditioning cascade** — melody first, then harmony conditioned on the melody, then bass, then percussion.

The crucial distinction we nailed down: this only works as **sequential conditioning** (each part written in response to the parts already there), *not* as independently-trained models stacked together (which would produce locally-plausible parts that don't fit vertically). Harmony isn't independent of melody — it's a relationship.

We then found this isn't speculative: the AMT base model **natively supports melody-conditioned generation** (`controls=` + `ops.combine`). The architecture we reasoned toward is a built-in feature of the chosen model. v1 stays scoped to melody (respecting the 4-week timeline and the plan's scope-creep risk), with the cascade logged as the project's north star and future-work section.

## 5. A sharpened evaluation requirement

A challenge about transposition ("if two pieces have identical intervals in different keys, aren't they the same song?") led to a real refinement of the plan. The answer is yes — and that has two consequences that must be designed together:
- **Transposition augmentation works** precisely because it teaches the model *key-invariance* (vary the absolute notes, hold the structure constant → the model learns the structure, not the literal pitches). It does *not* manufacture new songs.
- Therefore the **novelty/anti-plagiarism metric must be transposition-aware** — comparing intervals/contour/rhythm, not absolute pitches — or it would wrongly pass a transposed copy as "novel."

## 6. Standing up the environment

The GPU side went smoothly: PyTorch with CUDA installed, and the "hello world" confirmed the RTX 4090 is visible to PyTorch (`cuda.is_available() == True`). A useful concept locked in here: the driver's CUDA version is a *ceiling*; PyTorch bundles its own CUDA runtime under it, so no separate system-wide CUDA toolkit is needed.

The `anticipation` package was the hard part. It isn't on PyPI, so it installs from its GitHub repo (we read the README as the source of truth rather than guessing commands). The README also clarified two things that fed back into the plan: the package does **data prep + inference only, not training**, and it ships the exact melody-extraction and melody-conditioning tools v1 needs.

The real obstacle was a dependency chain: `anticipation` pins an old `transformers`, which pins an old `tokenizers` that has no prebuilt wheel for Python 3.12 — forcing a from-source compile that ultimately failed because the old code is rejected by a modern Rust compiler. Rather than fight that, we **pivoted to a Python 3.11 environment (`venv311`)**, where that old `tokenizers` installs as a ready-made wheel — no compilation. System Python 3.12 was left untouched (the "interpreters coexist; venv layers on top" principle in action).

The day ended on the genuine milestone — a **functional import test** of the whole AMT stack returning `all imports OK` — not merely a "pip succeeded" message.

---

## Where things stand

**Week 0 is complete.** Data exported, inventoried, and scoped (melody model, flute line, Ex Machina dropped); research question and metrics frozen (with the transposition-aware novelty refinement); environment fully working with the AMT stack importing on Python 3.11.

**Active environment:** `source ~/music-rag/venv311/bin/activate` (the old 3.12 `venv` is now dead weight).

## Next session (Week 1, the fun part)

Load `stanford-crfm/music-small-800k`, feed it the example MIDI, generate a continuation, and **listen** — the "understand what good and bad sound like before training" milestone. From there: build the data pipeline (extract the flute melody line, chunk, transpose-augment, train/val split by whole piece).
