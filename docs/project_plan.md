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
| Data | ~50–200 personal pieces, exported from Sibelius Ultimate |
| Base model (default) | `stanford-crfm/music-small-800k` (Anticipatory Music Transformer, 128M) |
| Compute | RTX 4090 (24 GB), 64 GB RAM, Windows + WSL2 |
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
- **Novelty / anti-plagiarism check** *(critical with small data + retrieval):* measure n-gram overlap between generated output and the training set. With a tiny corpus and a retrieval system, the model may just regurgitate your existing phrases — you must show the output is *new*, not copied. This is both a metric and a research-integrity point reviewers will look for.

**Subjective evaluation**
- **Blind A/B listening test.** Randomize so you don't know which output is RAG vs. baseline when rating.
- Likert scales (1–7) for: overall musical quality, and "sounds like my style."
- Recruit 1–2 other musicians to reduce single-evaluator bias.

## 10. Week-by-week plan

### Week 0 — the next 2 days (scope + setup)
- [ ] Export catalog from Sibelius: **MIDI (.mid)** for training + **MusicXML (.mxl)** as a richer archive. Unfold repeats. One file per piece.
- [ ] Inventory the data: count pieces, instruments, key/tempo spread (use `music21`). Decide single-instrument subset vs. full multi-track for v1.
- [ ] Set up WSL2 + Python + PyTorch (CUDA) + the `anticipation` package; confirm the GPU is visible to PyTorch.
- [ ] Write the research question and evaluation metrics down (this doc's §2 and §9) — freeze them.
- **Done when:** data is exported and counted, environment runs a GPU "hello world," and metrics are decided.

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
- Text → MIDI in your style (requires captioning your catalog).
- Multi-instrument / full-arrangement infilling.
- CLaMP-based semantic retrieval (v2).
- Interactive tool / demo.

## 15. Decisions log
- *(Week 0)* Primary task = phrase→MIDI infilling/continuation; base model = AMT small; fine-tune (not from scratch); RAG over personal corpus.
- *(add as you go…)*
