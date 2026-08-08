# Project Plan — Personal-Style Symbolic Music Generation with Retrieval Augmentation

*A 4-week summer research project. Living document — update the "Decisions log" and "Parking lot" as you go.*

*Last updated: end of Week 1 (data pipeline complete).*

---

## 1. One-line summary

Fine-tune a pretrained MIDI model on my own catalog so it continues/infills musical phrases **in my style**, then test whether **retrieving motifs from my own corpus** (RAG) improves the results compared to the fine-tuned model alone.

## 2. Research question

> Does retrieval augmentation — conditioning generation on similar phrases drawn from the composer's own corpus — improve the *stylistic fidelity* and *musical quality* of phrase continuations, compared to a fine-tuned model with no retrieval?

This "with vs. without RAG" contrast is the experimental spine of the paper. Everything else exists to make that comparison clean.

## 3. Locked-in decisions (the spec)

| Dimension | Decision |
|---|---|
| Primary task | Phrase → MIDI continuation / infilling |
| Conditioning | Seed MIDI phrase (no text captions needed) |
| Training strategy | Transfer learning: fine-tune a pretrained model (not from scratch) |
| Novel contribution | Retrieval from the composer's own corpus, compared with/without |
| Data | **46 harvestable pieces** exported from Sibelius Ultimate (of 49 exported; see Decisions log). Consistent wind/orchestral ensemble. |
| **v1 target** | **Melody model**: continue/infill a single extracted melodic voice. Stage one of a planned conditioning cascade (see §14). |
| **v1 melody routing** | Pooled by family: any melody-register voice (flutes, oboes, clarinets, saxes, trumpets, horns, mid-register trombones/bassoons, violins) is a candidate. Timbre is discarded — the model learns melodic *shape*, not instrument. |
| Base model (default) | `stanford-crfm/music-small-800k` (Anticipatory Music Transformer, 128M) |
| Compute | RTX 4090 (24 GB), 64 GB RAM, Windows + WSL2 |
| Python | **3.11** (venv `venv311`). Forced by `anticipation`'s pinned `tokenizers==0.13.3`. |
| Deliverable | Written research report / paper |
| Evaluator | The composer (blinded) + 1–2 additional musicians |

## 4. The two systems you'll compare

**System A — Baseline (no retrieval)**
`seed phrase → fine-tuned AMT → continuation`

**System B — RAG variant**
`seed phrase → retrieve k similar phrases from my corpus → condition the fine-tuned AMT on them → continuation`

Both share the *same* fine-tuned generator. The only difference is the retrieval layer. That isolates the variable you're studying.

## 5. The central constraint: small data

The single biggest design driver. Mitigations, in order of importance — **all now realized in the Week 1 pipeline:**

1. **Transfer learning.** Fine-tune a model that already knows general music. *(Week 2.)*
2. **Data augmentation.** ✅ Transpose every chunk into all 12 keys (centered ±6 semitones). Realized: **5,515 chunks → 66,122 examples**, a ~12× multiplier.
3. **Chunking.** ✅ Overlapping 15s windows (50% overlap). Realized: **462 melodies → 5,515 chunks.**
4. **Parameter-efficient fine-tuning (LoRA) / layer freezing.** *(Week 2 — try full fine-tune first, switch to LoRA if memorization appears.)*
5. **Strict validation split + early stopping.** ✅ By-**piece** hold-out (~15% of pieces), fixed seed. *(Early stopping in Week 2.)*

## 6. Tooling & environment

- **OS layer:** WSL2 (Ubuntu) on Windows.
- **Core:** Python 3.11 (`venv311`), PyTorch (CUDA cu124), Hugging Face `transformers`.
- **Symbolic music:** the `anticipation` package (AMT tokenization + inference), `pretty_midi` (parsing, note-level manipulation — the pipeline's workhorse), `music21` (Week 0 inventory), `pdfplumber` (score-roster extraction — see §7 note).
- **Experiment hygiene:** simple print/CSV logging; `git` from day one.

> ⚠️ **Tokenizer pitfall:** the tokenizer is tied to the base model. Fine-tuning AMT → use the `anticipation` encoding (`midi_to_events`), **not** MidiTok. Mixing tokenizers silently breaks everything.

> ⚠️ **Training-tooling gap (Week 0):** `anticipation` does **prep + tokenization + inference only — no training code.** Since the model loads as a standard `AutoModelForCausalLM`, fine-tune (Week 2) with HF `transformers`' own training machinery.

> 📌 **AMT natively supports melody-conditioned generation** (`generate(..., controls=melody)` + `ops.combine`) — the exact mechanism the future cascade needs.

> 🧱 **Pipeline module structure (Week 1).** Reusable code lives in named modules (not numbered lesson scripts): `routing.py` (family→role table, register threshold, piece overrides), `router.py` (`gm_family`, `route_track`), `harvest.py` (three gates + writer), `batch_harvest.py`, `chunk.py`, `batch_chunk.py`, `augment.py`, `split.py`.

## 7. Model selection shortlist

**Path A (recommended): fine-tune the Anticipatory Music Transformer.**
- `stanford-crfm/music-small-800k` — 128M params, the safe default for a 4090.
- `stanford-crfm/music-medium-800k` — larger; try if small works and time allows.
- *Why:* purpose-built for continuation AND infilling, open weights, ready-made MIDI↔token helpers.

**Backup candidates (Week 1 only if AMT disappoints):** `SkyTNT/midi-model`, `Moonbeam`.

**Path B (learning exercise):** small GPT from scratch (MidiTok/REMI). Toy version only.

## 8. Retrieval design (System B) — *Week 3*

The retrieval database = your own corpus of melody chunks. **The augmented-chunk filenames carry full provenance** (`piece_idxNN_melody_chunkNN_t±NN.mid`), which is what RAG needs: when a phrase is retrieved you can trace it back to its source piece — essential for the novelty/anti-plagiarism analysis.

- **v1 (start here):** simple similarity — feature vector (pitch-class histogram, melodic interval n-grams, rhythm profile) → nearest neighbors. *(Note: the transposition-invariant interval signature already built in `harvest.py`'s de-dupe is a natural starting point.)*
- **v2 (upgrade if time):** semantic retrieval with **CLaMP**.
- **Conditioning:** prepend retrieved phrase tokens as control/anticipation events. Keep the seed identical across A and B.

## 9. Evaluation plan

**Objective metrics**
- Style-distribution similarity: pitch-class, interval, rhythm-duration histograms vs. the real corpus.
- **Fréchet Music Distance (FMD)** against your corpus.
- Key/tempo consistency between seed and continuation.
- **Novelty / anti-plagiarism check** *(critical with small data + retrieval):* n-gram overlap between output and training set. **⚠️ Must be TRANSPOSITION-AWARE** — compare on intervals/contour/rhythm, *not* absolute pitches (a transposed copy is still a copy). *(This is the same principle the pipeline's de-dupe and augmentation already rest on — the transposition-invariant signature is reusable here.)*

**Subjective evaluation**
- **Blind A/B listening test.** Randomize so you don't know RAG vs. baseline when rating.
- Likert (1–7): overall musical quality, and "sounds like my style."
- 1–2 other musicians to reduce single-evaluator bias.

> **Note on validation size:** the held-out set is **7 pieces**, not "23,155 files." The files are highly correlated chunks/transpositions of those 7 sources — so the *piece* count is the true statistical power. Report as "held out 7 of 46 pieces." This small piece count is *why* the plan pairs objective metrics with the listening test.

## 10. Week-by-week plan

### Week 0 — scope + setup ✅ COMPLETE
Data exported (49 pieces), inventoried, scoped to a melody model; environment working; research question + metrics frozen (with the transposition-aware novelty refinement).

### Week 1 — ML foundations + first inference + **the data pipeline** ✅ COMPLETE
- [x] Learn transformers hands-on (Karpathy) — *background track, ongoing.*
- [x] Run AMT inference out of the box: unconditional (chaotic) and seeded continuation (coherent). Ear calibrated to the baseline.
- [x] Build the data pipeline: **melody routing → harvest → chunk → augment → split.**
- **Done when:** you can generate a continuation from the pretrained model ✅, and your data is extracted + augmented + split ✅.
- **Realized corpus:** 46 pieces → **462 melody lines → 5,515 chunks → 66,122 augmented examples → 42,967 train / 23,155 val (by piece).**
- **Remaining bridge to Week 2:** `MIDI → events` tokenization of the corpus (the model can't train on raw `.mid`).

### Week 2 — fine-tune the baseline (System A)
- [ ] **Tokenize** the train/val corpus with `midi_to_events` (the Week 1 bridge).
- [ ] Fine-tune `music-small-800k` on the corpus via HF `transformers`. Watch train vs. val loss for overfitting.
- [ ] Generate continuations from held-out seeds; run the objective metrics + novelty check.
- [ ] Iterate: learning rate, augmentation, LoRA vs. full fine-tune.
- **Done when:** System A produces stylistically plausible, *novel* continuations and you have baseline metric numbers.

### Week 3 — build retrieval (System B)
- [ ] Build the retrieval index over your corpus (v1 feature-based; reuse the interval signature).
- [ ] Wire retrieval → conditioning into the generation pipeline.
- [ ] Generate **matched** outputs: same seeds through A and B.
- **Done when:** you have paired A/B outputs for a fixed seed set.

### Week 4 — evaluate + write
- [ ] Blind listening test (you + others); objective metrics for both systems.
- [ ] Analyze: does RAG help? On which dimensions? Failure modes?
- [ ] Write the paper (§12). Include negative/mixed results honestly.
- **Done when:** the paper draft is complete with results, figures, discussion.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Overfitting / memorizing your corpus | Augmentation (done), LoRA, val-set early stopping, novelty metric |
| Environment setup eats days | Done in Week 0 |
| Evaluator bias | Blind A/B, extra musicians |
| Scope creep | Piano-solo melody extraction & roster-based fine ID kept in the parking lot |
| RAG just retrieves near-copies | Transposition-aware novelty metric; tune retrieval k and conditioning strength |
| **Silent data loss in the pipeline** *(new)* | "List before you write" + flag-don't-write batch discipline caught strings, stray files, naming collisions. Any fully-excluded piece is investigated (cause 1 harmless vs. cause 2 router gap). |
| **Small validation piece count (7)** *(new)* | Pair objective metrics with the listening test; don't rely on val loss alone. |

## 12. Paper outline

1. Introduction & motivation (personal-style generation, the RAG question)
2. Related work (AMT/infilling, symbolic music gen, RAG for music)
3. Method (data pipeline, fine-tuning, retrieval design, A vs. B setup)
4. Experiments (metrics, listening test protocol)
5. Results (objective + subjective, novelty analysis)
6. Discussion, limitations, future work
7. Conclusion

## 13. Learning resources

1. Andrej Karpathy — "Let's build GPT from scratch."
2. MidiTok docs — tokenization concepts.
3. `anticipation` package README + Colab — the base model.
4. Papers to read & cite: Anticipatory Music Transformer (Thickstun et al., 2023); Retrieval Augmented Generation of Symbolic Music with LLMs (Jonason et al., 2023); VMB (2024); MidiCaps (Melechovsky et al., 2024) & text2midi (Bhandari et al., 2025); Fréchet Music Distance (Retkowski et al., 2024).

## 14. Parking lot (stretch goals — only after the core works)

- **The conditioning cascade (north star).** v1 = the **melody** stage. Future stages add harmony, then bass, then percussion, each **conditioned on everything generated before it** (sequential conditioning — Version B). AMT's `controls=`/`ops.combine` supports this natively.
  - *Pipeline groundwork already laid:* `routing.py` already tags every track `melody`/`bass`/`harmony`/`percussion`. Later stages just change which role the harvest consumes. **The shelved score-roster extractor** (fine instrument IDs from PDFs) is infrastructure for these stages — v1 doesn't need it, but it recovers the alto-vs-bass-clarinet distinctions the harmony/bass stages may want.
- **Piano-solo melody extraction.** *(New — surfaced Week 1.)* `album` and `prelude_in_c#_minor` are solo-piano pieces excluded from v1: skyline can't separate melody from accompaniment in a two-hand chordal texture. A real technique (melodic-line extraction from polyphony) would recover them.
- **Density-aware chunk floor.** *(New.)* `MIN_CHUNK_NOTES=8` is a flat note-count floor; it has a fast-vs-slow bias and could drop legitimately slow, sustained phrases. A tempo/density-scaled floor would be fairer.
- **Stratified train/val split.** *(New.)* Current split is random-seeded; a stratified split (by ensemble type / size) would guarantee a representative validation set on the small 46-piece corpus.
- Text → MIDI in your style (requires captioning your catalog).
- Multi-instrument / full-arrangement infilling.
- CLaMP-based semantic retrieval (v2).
- Interactive tool / demo.

## 15. Decisions log

- *(Week 0)* Primary task = phrase→MIDI infilling/continuation; base model = AMT small; fine-tune (not from scratch); RAG over personal corpus.

**Week 0 session (data + setup):** export settings frozen (MIDI Type 1, PPQ 960, unfold repeats, GM sounds, one file per piece); pieces <~20s set aside as seed prompts; 49 pieces; inventory switched `music21`→`pretty_midi` for GM programs; **v1 = melody model** (extract one consistent melodic voice from every piece); v1 is stage one of a **conditioning cascade** (sequential conditioning); dropped **Ex Machina** (electronic, no melody); novelty metric must be **transposition-aware**; training-tooling gap noted; **environment = Python 3.11 in `venv311`**.

**Week 1 session (first inference):**
- Ran AMT inference. Unconditional = chaotic (no anchor + repetition trap); seeded continuation = coherent. Confirmed the "prompt collapses the model into a coherent space" principle and calibrated the ear to the untrained baseline.

**Week 1 session (routing — the big architectural decision):**
- **Built, then shelved, a score-roster extractor.** Parses each PDF's first system with `pdfplumber` (x-position margin filter → group fragments by `top` → parse against instrument vocabulary) to recover true instrument names the GM programs collapse. Worked and cross-validated against MIDI track order — but **shelved for v1** in favor of Option B. Kept as cascade infrastructure.
- **Pivoted to Option B: order-independent per-track routing** (`gm_family(program)` → `FAMILY_ROLE` → register refinement). Reason: the roster-walk aligner desynced permanently on an out-of-order interleaved track (muted trumpet inside the trombones); a forward-only pointer can't recover. Stateless per-track routing is structurally immune. For a **family-pooled melody model**, we only need family→role, not exact per-track identity.
- **Register threshold** (`BASS_REGISTER_MAX=55`) sorts `ROLE_AMBIGUOUS` families (clarinet, sax, trombone, bassoon, strings) — family default is melody; median pitch ≤55 flips to bass. Validated on A Letter's 6 clarinets (clean A#5→B1 staircase; bass/contrabass correctly flipped).
- **Bassoon & trombone routed to melody** (1st chairs carry tenor lines; register prunes the bass passages) — chose "route generously, let register prune" over per-chair disambiguation.
- **Strings added** (programs 40–45, 48, 49 → `strings`, role-ambiguous; 46 harp → `harmony`). Caught because A Letter (wind ensemble) had no violins, so orchestral violin melodies would have been *silently dropped*. Validated on Paradise Lost Mov 1.
- **Piece-level overrides** (`PIECE_OVERRIDES`): `concerto_for_tuba2`, `concerto_no_1` force tuba→melody (they're tuba concertos — the soloist is a tuba). Chosen over a global rule, which would corrupt the 44 pieces where tuba is correctly bass.

**Week 1 session (harvest gates):**
- Three gates: **length** (`MIN_NOTES=25`), **skyline** (universal monophonic top-voice reducer), **de-dupe** (transposition-invariant interval signature, alignment-tolerant, `DUPE_THRESHOLD=0.80`). Gate order: **skyline before length** (length judges the final reduced line).
- **Discovery:** A Letter's flutes/piccolo are *independent contrapuntal lines*, not doublings (falsified a Week 0 assumption via signature comparison — 18% similar, not ~100%). De-dupe by measurement, not by hunch. Kept the gate as an asymmetric-cost safety net.
- `DUPE_THRESHOLD=0.80` is **provisional** — validated only on the "independent" end (nothing scored above ~28%); never yet tested against a confirmed doubling.

**Week 1 session (batch + corpus):**
- Batch discipline: `try/except` per piece, flag-don't-write on empty output, end-of-run report, "list before you write" discovery step.
- `EXCLUDE` set: `ex_machina` (dropped), `avanti_melody` + `flutesolo1` (stray working files caught by the count check).
- **462 melody files from 46 pieces.** Piano solos `album` + `prelude_in_c#_minor` scoped **out** of v1 (skyline can't extract melody from chordal piano) → parking lot.

**Week 1 session (chunk / augment / split):**
- **Chunking:** 15s windows, 7.5s step (50% overlap), `MIN_CHUNK_NOTES=8`, timestamps shifted to 0 per chunk. → 5,515 chunks. `MIN_CHUNK_NOTES` is **provisional** (fast-vs-slow bias).
- **Augmentation:** all 12 keys via centered shifts (−6..+5), bounds guard (pitch 21–108, rejects whole out-of-range shifts). → 66,122 files (58 bounds rejections, 0.09%). Teaches key-invariance (shape = signal, absolute pitch = noise).
- **Split:** by **whole piece** (~15%), random with **fixed seed 42** for reproducibility (Week 3 comparisons require an identical val set). 7 val pieces (`a_letter, chorale_and_procession, concerto, el_mar, elegy, forest_overture, spanish_sunrise`) / 39 train. → 42,967 train / 23,155 val files. Val is 35% of *files* but 15% of *pieces* (file-heavy pieces in the holdout).

**Week 1 recurring engineering lessons:** single source of truth for names (a hardcoded `'aletter'` collided with computed `'a_letter'`, doubling files); tests must not write into pipeline input folders (stray `test_chunk` files rode into augmentation); "list before you write" catches count mismatches early; refactors must preserve behavior (renamed modules verified by identical 22-file output); `from X import Y` searches only `X`.

- *(add as you go…)*
