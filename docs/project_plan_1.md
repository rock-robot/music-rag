# Project Plan — Personal-Style Symbolic Music Generation with Retrieval Augmentation

*A 4-week summer research project. Living document — update the "Decisions log" and "Parking lot" as you go.*

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
| Data | **49 pieces** exported from Sibelius Ultimate (bottom of the original ~50–200 estimate → small-data mitigations in §5 are now load-bearing, not optional). Consistent wind/orchestral ensemble. |
| **v1 target** | **Melody model**: continue/infill a single extracted melodic voice (top woodwind line — Flute/Piccolo). This is **stage one of a planned conditioning cascade** (see Decisions log & §14). |
| Base model (default) | `stanford-crfm/music-small-800k` (Anticipatory Music Transformer, 128M) |
| Compute | RTX 4090 (24 GB), 64 GB RAM, Windows + WSL2 |
| Python | **3.11** (in venv `venv311`). Forced by `anticipation`'s pinned `tokenizers==0.13.3`, which won't build on 3.12 — see Decisions log. |
| Deliverable | Written research report / paper |
| Evaluator | The composer (blinded) + 1–2 additional musicians |

## 4. The two systems you'll compare

**System A — Baseline (no retrieval)**
`seed phrase → fine-tuned AMT → continuation`

**System B — RAG variant**
`seed phrase → retrieve k similar phrases from my corpus → condition the fine-tuned AMT on them → continuation`

Both share the *same* fine-tuned generator. The only difference is the retrieval layer. That isolates the variable you're studying.

## 5. The central constraint: small data (~50–200 pieces)

This is the single biggest design driver. A personal catalog is tiny by ML standards (the base model was trained on ~170k files). You will **not** train from scratch. Mitigations, in order of importance:

1. **Transfer learning.** Start from a model that already knows general music; only nudge it toward your style. This is exactly how the field handles small datasets.
2. **Data augmentation.** Transpose every piece into multiple keys (±6 semitones, or all 12) — this alone multiplies your data ~12×. Add mild tempo scaling and velocity jitter. Augment at the MIDI-file level so it's tokenizer-agnostic.
3. **Chunking.** Split each piece into many overlapping phrase-length segments. 100 pieces can become several thousand training examples.
4. **Parameter-efficient fine-tuning (LoRA) or layer freezing.** Updating a small number of parameters resists overfitting on small data. Try full fine-tuning first; switch to LoRA if you see memorization.
5. **Strict validation split + early stopping.** Hold out ~15% of pieces (whole pieces, not chunks) to detect overfitting honestly.

## 6. Tooling & environment

- **OS layer:** WSL2 (Ubuntu) on your Windows machine. PyTorch+CUDA works on native Windows too, but most of this ecosystem is smoother on Linux.
- **Core:** Python 3.11, PyTorch (CUDA build), Hugging Face `transformers`.
- **Symbolic music:** the `anticipation` package (for AMT — handles tokenization + infilling structure), plus `pretty_midi` and `music21` for parsing/feature extraction. `MidiTok` if you build your own model (see §7, Path B).
- **Experiment hygiene:** Weights & Biases or a simple CSV log; `git` from day one.

> ⚠️ **Tokenizer pitfall:** the tokenizer is tied to the base model. If you fine-tune AMT, use the `anticipation` package's encoding (`midi_to_events`), **not** MidiTok. Mixing tokenizers silently breaks everything. Only use MidiTok/REMI if you build your own model in Path B.

> ⚠️ **Training-tooling gap (discovered Week 0):** the `anticipation` package provides **data prep + tokenization + sampling/inference only — it does NOT contain training code.** Its README says to bring your own training codebase (the authors used Levanter). Since the model loads as a standard `AutoModelForCausalLM`, the plan is to fine-tune (Week 2) using Hugging Face `transformers`' own training machinery rather than Levanter. Flag this when planning Week 2.

> 📌 **AMT natively supports melody-conditioned generation** (`generate(..., controls=melody)` + `ops.combine`). This is exactly the sequential-conditioning mechanism the future cascade needs — see §14 / Decisions log.

## 7. Model selection shortlist

**Path A (recommended): fine-tune the Anticipatory Music Transformer.**
- `stanford-crfm/music-small-800k` — 128M params, the safe default for a 4090.
- `stanford-crfm/music-medium-800k` — larger; try if small works and you have time.
- *Why:* purpose-built for continuation AND infilling, open weights, ready-made MIDI↔token helpers, trained on Lakh MIDI. Lowest-friction path to a working baseline.

**Backup candidates (evaluate in Week 1 only if AMT disappoints):**
- `SkyTNT/midi-model` — open MIDI event transformer for generation.
- `Moonbeam` — 2025 MIDI foundation model with conditional generation + infilling.

**Path B (learning exercise, not the production model): build a small GPT from scratch** with MidiTok/REMI tokens on a generic corpus, then fine-tune on your data. More educational, more risk. Use the *toy* version of this in Week 1 to understand transformers; don't bet the project on it.

## 8. Retrieval design (System B)

The retrieval database = your own corpus of phrases.

- **v1 (start here):** simple similarity — represent each phrase by a feature vector (pitch-class histogram, melodic interval n-grams, rhythm profile) and retrieve nearest neighbors to the seed.
- **v2 (upgrade if time):** semantic retrieval with **CLaMP** (contrastive language–music embeddings) for more musically-aware matching.
- **Conditioning mechanism:** prepend retrieved phrase tokens to the model's context as additional control/anticipation events. Keep the seed identical across A and B so only retrieval differs.

## 9. Evaluation plan (design this in Week 0 — it defines "success")

**Objective metrics**
- Style-distribution similarity: compare pitch-class, interval, and rhythm-duration histograms of generated output vs. your real corpus.
- **Fréchet Music Distance (FMD)** against your corpus as the "real" distribution.
- Key/tempo consistency between seed and continuation.
- **Novelty / anti-plagiarism check** *(critical with small data + retrieval):* measure n-gram overlap between generated output and the training set. With a tiny corpus and a retrieval system, the model may just regurgitate your existing phrases — you must show the output is *new*, not copied. This is both a metric and a research-integrity point reviewers will look for. **⚠️ This metric must be TRANSPOSITION-AWARE:** compare on intervals/contour/rhythm, *not* absolute pitches. A transposed copy is still a copy (Mary-in-C and Mary-in-F are the same song), so a pitch-literal metric would wrongly rubber-stamp transposed regurgitation as "novel." This requirement is the flip side of why transposition augmentation works (it teaches key-invariance) — the two must be designed together.

**Subjective evaluation**
- **Blind A/B listening test.** Randomize so you don't know which output is RAG vs. baseline when rating.
- Likert scales (1–7) for: overall musical quality, and "sounds like my style."
- Recruit 1–2 other musicians to reduce single-evaluator bias.

## 10. Week-by-week plan

### Week 0 — the next 2 days (scope + setup) ✅ COMPLETE
- [x] Export catalog from Sibelius: **MIDI (.mid)** for training + **MusicXML (.mxl)** as a richer archive. Unfold repeats. One file per piece. *(Export settings used: Type 1, PPQ 960, unfold repeats on, General MIDI sounds, all instruments kept, one file per piece. Pieces under ~20s excluded from training and set aside as future seed prompts.)*
- [x] Inventory the data: count pieces, instruments, key/tempo spread. Decide single-instrument subset vs. full multi-track for v1. *(49 pieces. `music21` returned mostly "Unknown" instruments, so a second `pretty_midi` inventory was run — it reads GM program numbers directly. Result: a consistent wind/orchestral ensemble, mostly 24–41 parts/piece, where high part counts are section divisi rather than many distinct instruments. v1 decision below.)*
- [x] Set up WSL2 + Python + PyTorch (CUDA) + the `anticipation` package; confirm the GPU is visible to PyTorch. *(GPU "hello world" passed: `torch.cuda.is_available() == True`, RTX 4090 detected, PyTorch cu124. AMT stack imports cleanly — see env note below. **Active environment is now `venv311` (Python 3.11), NOT the original `venv` (3.12)** — see Decisions log for why.)*
- [x] Write the research question and evaluation metrics down (this doc's §2 and §9) — freeze them. *(Frozen. One refinement: the novelty/anti-plagiarism metric must be transposition-aware — see Decisions log.)*
- **Done when:** data is exported and counted, environment runs a GPU "hello world," and metrics are decided. ✅

> **Activate the project environment each session with:** `source ~/music-rag/venv311/bin/activate` (the old 3.12 `venv` is dead weight and can be deleted once you're confident). In VS Code, re-select the interpreter to the `venv311` one.

### Week 1 — ML foundations + first inference
- [ ] Learn transformers hands-on (Karpathy "Let's build GPT"); build the toy GPT (Path B taste).
- [ ] Run AMT *inference* out of the box: feed it a seed, get a continuation, listen. Understand what "good" and "bad" look like before training.
- [ ] Build the data pipeline: MIDI → events, chunking, transposition augmentation, train/val split.
- **Done when:** you can generate a continuation from a pretrained model and your data is tokenized + augmented + split.

### Week 2 — fine-tune the baseline (System A)
- [ ] Fine-tune `music-small-800k` on your corpus. Watch train vs. val loss for overfitting.
- [ ] Generate continuations from held-out seeds; run the objective metrics + novelty check.
- [ ] Iterate: adjust learning rate, augmentation, LoRA vs. full fine-tune.
- **Done when:** System A produces stylistically plausible, *novel* continuations and you have baseline metric numbers.

### Week 3 — build retrieval (System B)
- [ ] Build the retrieval index over your corpus (v1 feature-based).
- [ ] Wire retrieval → conditioning into the generation pipeline.
- [ ] Generate **matched** outputs: same seeds through A and B.
- **Done when:** you have paired A/B outputs for a fixed seed set, ready for evaluation.

### Week 4 — evaluate + write
- [ ] Run the blind listening test (you + others); collect objective metrics for both systems.
- [ ] Analyze: does RAG help? On which dimensions? Any failure modes?
- [ ] Write the paper (§12 outline). Include negative/mixed results honestly — they're still a contribution.
- **Done when:** the paper draft is complete with results, figures, and discussion.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Overfitting / memorizing your corpus | Augmentation, LoRA, val-set early stopping, novelty metric |
| Environment setup eats days | Do it in Week 0; WSL2; ask for help early |
| Evaluator bias (you judge your own style) | Blind A/B, extra musicians |
| Scope creep (text→MIDI, multi-instrument) | Keep them in the parking lot until core works |
| RAG just retrieves near-copies | Measure novelty; tune retrieval k and conditioning strength |

## 12. Paper outline

1. Introduction & motivation (personal-style generation, the RAG question)
2. Related work (AMT/infilling, symbolic music gen, RAG for music)
3. Method (data, fine-tuning, retrieval design, the A vs. B setup)
4. Experiments (metrics, listening test protocol)
5. Results (objective + subjective, novelty analysis)
6. Discussion, limitations, future work
7. Conclusion

## 13. Learning resources (in order)

1. Andrej Karpathy — "Let's build GPT from scratch" (transformers, hands-on).
2. MidiTok docs — symbolic music tokenization concepts.
3. `anticipation` package README + Colab — your actual base model.
4. Papers to read & cite:
   - Anticipatory Music Transformer (Thickstun et al., 2023) — your base model.
   - Retrieval Augmented Generation of Symbolic Music with LLMs (Jonason et al., 2023) — closest RAG precedent.
   - VMB: Multimodal Music Generation with Retrieval Augmentation (2024) — music-to-music conditioning.
   - MidiCaps (Melechovsky et al., 2024) & text2midi (Bhandari et al., 2025) — context for the field / your stretch goal.
   - Fréchet Music Distance (Retkowski et al., 2024) — evaluation metric.

## 14. Parking lot (stretch goals — only after the core works)
- **The conditioning cascade (north star).** v1 (this project) = the **melody** stage. Future stages add harmony, then bass, then percussion, each **conditioned on everything generated before it** (Version B — sequential conditioning, NOT independently-trained part models, which wouldn't fit together vertically). AMT's `controls=`/`ops.combine` already supports this natively. Frame v1 in the paper as stage one of this planned system — it strengthens the motivation and gives a clean future-work section.
- Text → MIDI in your style (requires captioning your catalog).
- Multi-instrument / full-arrangement infilling.
- CLaMP-based semantic retrieval (v2).
- Interactive tool / demo.

## 15. Decisions log
- *(Week 0)* Primary task = phrase→MIDI infilling/continuation; base model = AMT small; fine-tune (not from scratch); RAG over personal corpus.

**Week 0 session (data + setup):**
- **Export settings frozen:** MIDI Type 1 (keeps per-part separation, vs. Type 0 which flattens to one track), PPQ 960 (fine, tuplet-friendly timing grid), unfold repeats ON (avoids silent inconsistency across pieces + lost material), General MIDI sounds (standard, tool-readable instrument labels), all instruments kept, one file per piece. Both `.mid` (train on) and `.mxl` (richer archive) exported. Rationale theme throughout: **consistency across files prevents silent data corruption.**
- **Pieces <~20s excluded** from the training set, set aside as potential future seed prompts. (Don't over-prune — dataset is already small.)
- **Data reality:** 49 pieces (bottom of the original estimate). Mostly large wind/orchestral ensembles (24–41 parts), but parts are largely **section divisi**, not 30 distinct instruments. Core wind/brass instruments (Clarinet, Tuba, Flute, Oboe, Horn, Trumpet, Trombone, Bassoon) appear in ~80% of pieces → a consistent stylistic fingerprint.
- **Inventory tooling:** switched from `music21` to `pretty_midi` for the instrument breakdown — `music21` returned mostly "Unknown"; `pretty_midi` reads GM program numbers directly.
- **v1 = MELODY MODEL, not full multi-track.** Rationale: multi-track forces the model to learn vertical part-relationships (the hard, data-hungry skill) — exactly the wrong thing to attempt with 49 pieces. A near-universal solo voice doesn't exist (only ~5 truly solo pieces), so instead **extract one consistent melodic voice (top woodwind — Flute/Piccolo) from every piece.** Trade-off accepted: v1 captures melodic/harmonic style but not orchestrational style (named as a limitation).
- **v1 is stage one of a planned CONDITIONING CASCADE** (melody → harmony → bass → percussion), built as **sequential conditioning** (each stage conditioned on prior stages — "Version B"), NOT independently-trained models ("Version A," which would produce parts that don't fit together). AMT supports this natively via `controls=`/`ops.combine`. Logged as north star in §14.
- **Outlier handling for the melody corpus:** inclusion criterion is "**has a usable melodic line**," not "is a good piece overall." Dropped **Ex Machina** (electronic; no flute line at all). Kept **Jazz** and **Ohio River Cruise** for now — but Jazz has no flute (its melody, if used, would come from trumpet), and the splitter output will confirm whether each actually yields a clean melodic line.
- **Novelty metric must be TRANSPOSITION-AWARE** (see §9). Derived from the principle that two transpositions of a phrase are the same song: a pitch-literal novelty check would wrongly pass transposed copies. Transposition augmentation (teaches key-invariance) and the novelty metric (must ignore key) are two sides of the same coin and must be designed together.
- **Training-tooling gap noted** (see §6): `anticipation` does prep + inference only, not training. Plan to fine-tune via HF `transformers` since the model loads as `AutoModelForCausalLM`.
- **Environment = Python 3.11 in venv `venv311`.** Started on 3.12; `anticipation` pins `transformers==4.29.2` → `tokenizers==0.13.3`, which has no prebuilt wheel for 3.12 and **fails to compile from source on a modern Rust toolchain** (old code rejected by current compiler). Rather than downgrade Rust or patch source, pivoted to a fresh 3.11 venv where `tokenizers==0.13.3` installs as a prebuilt wheel — no compilation. System 3.12 left untouched (interpreters coexist; venv layers package isolation on top). Verified with a functional import test (`all imports OK`), not just pip's success message.
- *(add as you go…)*
