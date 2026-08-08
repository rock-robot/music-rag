# Project Plan — Personal-Style Symbolic Music Generation with Retrieval Augmentation

*A 4-week summer research project. Living document — update the "Decisions log" and "Parking lot" as you go.*

*Last updated: end of Week 2 (System A fine-tuned; LoRA chosen; generation recipe set; first objective metric built).*

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
| Data | **46 harvestable pieces** → **462 melody lines → 5,515 chunks → 66,122 augmented → 42,967 train / 23,155 val (by piece) → 3,084 / 1,641 packed 1024-token blocks.** |
| **v1 target** | **Melody model**: continue/infill a single extracted melodic voice. Stage one of a planned conditioning cascade (see §14). |
| **v1 melody routing** | Pooled by family: any melody-register voice is a candidate. Timbre discarded — the model learns melodic *shape*. |
| Base model | `stanford-crfm/music-small-800k` (Anticipatory Music Transformer, 128M params, GPT-2 architecture) |
| **System A (chosen)** | **LoRA fine-tune** of the base model (`r=8, alpha=16, target=["c_attn"]`, ~295K trainable params). Beat full fine-tuning head-to-head (val ~0.227 vs 0.243). See Decisions log. |
| **Generation recipe** | LoRA adapter, `anticipation.sample.generate`, **`top_p=0.90`**, ~10s clips (≤~11s reliable horizon), collapse-filter any batch. |
| Compute | RTX 4090 (24 GB), 64 GB RAM, Windows + WSL2 |
| Python | **3.11** (venv `venv311`). `transformers==4.29.2` (pinned by `anticipation`), `peft==0.4.0` (pinned to protect that chain). |
| Deliverable | Written research report / paper |
| Evaluator | The composer (blinded) + 1–2 additional musicians |

## 4. The two systems you'll compare

**System A — Baseline (no retrieval)** — *built (Week 2).*
`seed phrase → fine-tuned (LoRA) AMT → continuation`

**System B — RAG variant** — *Week 3.*
`seed phrase → retrieve k similar phrases from my corpus → condition the fine-tuned AMT on them → continuation`

Both share the *same* fine-tuned generator. The only difference is the retrieval layer. That isolates the variable. (The conditioning mechanism for System B is likely AMT's `controls=` argument, spotted in the verified `generate` signature — see §8.)

## 5. The central constraint: small data

The single biggest design driver. Mitigations, in order of importance — **all now realized:**

1. **Transfer learning.** ✅ Fine-tuned `music-small-800k` (baseline val 1.64 → 0.24).
2. **Data augmentation.** ✅ All 12 keys → 66,122 examples (~12×). *Note: transposition leaves intervals unchanged but changes pitch-classes — matters for which metrics use augmented vs. raw data (§9).*
3. **Chunking.** ✅ 462 melodies → 5,515 chunks.
4. **Parameter-efficient fine-tuning (LoRA).** ✅ **This turned out to be the decisive mitigation.** Full fine-tuning overfit after one epoch; LoRA (0.23% of params trainable) postponed the overfitting turn ~10 epochs and reached a lower val floor. LoRA is System A.
5. **Strict validation split + early stopping.** ✅ By-piece hold-out (7 pieces, seed 42); save-on-best + patience=3. *(Future: add `min_delta` so plateau noise doesn't reset patience — see parking lot.)*

## 6. Tooling & environment

- **OS layer:** WSL2 (Ubuntu) on Windows.
- **Core:** Python 3.11 (`venv311`), PyTorch (CUDA cu124), Hugging Face `transformers==4.29.2`, `peft==0.4.0`.
- **Symbolic music:** `anticipation` (AMT tokenization + inference), `pretty_midi` (parsing / note-level work), `music21` (Week 0 inventory), `pdfplumber` (shelved roster extractor).
- **Experiment hygiene:** print/CSV logging; `git` from day one.

> ⚠️ **Tokenizer pitfall:** the tokenizer is married to the base model. Use `anticipation`'s `midi_to_events`, not MidiTok. The integer ranges (time≈0, duration≈10000, note≥11000) are baked into the model's weights; a different tokenizer → the model reads noise → loss never drops.

> ⚠️ **Training-tooling gap:** `anticipation` does prep + tokenization + inference only — **no training code.** Since the model loads as `AutoModelForCausalLM`, we fine-tune with a hand-written HF `transformers` loop (built Week 2).

> ⚠️ **`token_out_embeddings` warning is benign.** The checkpoint's untied output head is discarded by HF's tied-by-default `GPT2LMHeadModel`. Investigated twice; proven harmless by a working fine-tune. The authors load it this way too.

> ⚠️ **Dependency chain is fragile.** `peft` must be pinned (`0.4.0`) so it doesn't drag `transformers` forward and break the `transformers→tokenizers→anticipation` chain. Always verify with a functional import test, not pip's success message.

> 📌 **AMT vocab (from `anticipation.vocab`):** `SEPARATOR=55025`, `AUTOREGRESS=55026`, `ANTICIPATE=55027`, `CONTEXT_SIZE=1024`, `VOCAB_SIZE=55028`, `NOTE_OFFSET=11000`, `DUR_OFFSET=10000`, `REST=27512`. No PAD token (→ drop-the-tail when packing).

> 📌 **`generate` signature (verified):** `generate(model, start_time, end_time, inputs=None, controls=None, top_p=1.0, debug=False, delta=500)`. Only `top_p` for sampling (no `repetition_penalty`). `controls=` is the melody-conditioning hook → the likely System B / cascade mechanism.

> 🧱 **Pipeline / training modules.** Week 1: `routing.py`, `router.py`, `harvest.py`, `batch_harvest.py`, `chunk.py`, `batch_chunk.py`, `augment.py`, `split.py`. Week 2: tokenize+pack script (→ `packed/train.txt`, `packed/val.txt`), `train.py` (full FT), `train_lora.py` (LoRA), `compare_generate.py` / three-way listen, `style_metrics.py` (interval histogram + collapse filter).

## 7. Model selection

**Chosen: fine-tune the Anticipatory Music Transformer with LoRA.**
- `stanford-crfm/music-small-800k` — 128M params, safe default for a 4090. (Confirmed: 128,103,936 params, embedding (55028, 768), context 1024. "800k" = training steps, not params.)
- **Full fine-tune vs. LoRA decided in favor of LoRA** (Week 2). Full FT overfits after one epoch; LoRA generalizes better on this tiny corpus.
- Backup candidates (`SkyTNT/midi-model`, `Moonbeam`) not needed — AMT works.

## 8. Retrieval design (System B) — *Week 3*

The retrieval database = your own corpus of melody chunks. **Augmented-chunk filenames carry full provenance** (`piece_idxNN_melody_chunkNN_t±NN.mid`) — trace any retrieved phrase back to its source piece (essential for the novelty analysis).

- **v1 (start here):** simple similarity — feature vector (pitch-class histogram, melodic interval n-grams, rhythm profile) → nearest neighbors. *The transposition-invariant interval signature from `harvest.py`'s de-dupe is a natural starting point, and the `interval_counts`/`histogram` tooling from Week 2 §9 is reusable here.*
- **v2 (upgrade if time):** semantic retrieval with **CLaMP**.
- **Conditioning:** prepend retrieved phrase tokens as control/anticipation events — likely via the `controls=` argument. Keep the seed identical across A and B.

## 9. Evaluation plan

**Objective metrics** — *tooling foundation built Week 2; interval histogram done.*
- **Style-distribution similarity:** pitch-class, interval, rhythm-duration histograms vs. the real corpus. Pattern: reduce MIDI → feature histogram → normalize → compare. *The interval histogram is built.* **Use the un-augmented `corpus/` (462 original melodies) as the real reference** — fewest artifacts, and transposition-invariant metrics aside, augmented files distort pitch-class stats.
  - **Interval-histogram result (Week 2):** absolute intervals; corpus vs. batched (20-seed) generations, collapse-filtered. Filtered pretrained and LoRA distributions are **nearly identical** — both over-concentrate on whole-steps (~0.69 vs. corpus 0.369) and under-represent the corpus's variety — *despite* a clear subjective preference for LoRA. **Lesson: one histogram is too coarse to capture the audible difference; this validates the multi-metric design.**
- **Fréchet Music Distance (FMD)** against your corpus. *(Not yet built. A single distance number per histogram is the intermediate step / seed of FMD.)*
- Key/tempo consistency between seed and continuation.
- **Novelty / anti-plagiarism check** *(critical; not yet built):* n-gram overlap between output and the training set. **⚠️ Must be TRANSPOSITION-AWARE** — compare intervals/contour/rhythm, not absolute pitches. Reuses the transposition-invariant signature the pipeline already rests on. **Use the *train* set specifically** as the reference here (unlike style similarity, which uses the whole corpus) — the question is whether the model regurgitated what it *trained* on.

**Subjective evaluation**
- **Blind A/B listening test.** Randomize so you don't know RAG vs. baseline when rating. *(Week 2's ear-validation was non-blind and single/few-sample — fine for development, but the paper needs the blind, multi-sample protocol.)*
- Likert (1–7): overall musical quality, and "sounds like my style."
- 1–2 other musicians to reduce single-evaluator bias.

> **Note on validation size:** the held-out set is **7 pieces**, not "1,641 blocks." Report as "held out 7 of 46 pieces." This small piece count is *why* the plan pairs objective metrics with the listening test.

> **Note on generation for metrics:** always **batch-generate and collapse-filter**. `top_p=0.90` collapses into a repetition trap ~20% of the time (intermittent — single samples hide it). Collapsed runs are trivially detectable (absurd note counts / high interval-0 fraction). Aggregate metrics over the *clean* runs and report the failure rate.

## 10. Week-by-week plan

### Week 0 — scope + setup ✅ COMPLETE
Data exported (49 pieces), inventoried, scoped to a melody model; environment working; research question + metrics frozen (with the transposition-aware novelty refinement).

### Week 1 — ML foundations + first inference + the data pipeline ✅ COMPLETE
AMT inference calibrated; pipeline built (routing → harvest → chunk → augment → split). Realized: 46 pieces → 462 melodies → 5,515 chunks → 66,122 augmented → 42,967 train / 23,155 val.

### Week 2 — fine-tune the baseline (System A) ✅ COMPLETE
- [x] **Tokenize + pack** the corpus (`midi_to_events` → provenance `.txt` per chunk → packed 1024-blocks with triple-SEP walls + AR codes, tail dropped, separate train/val ribbons). → 3,084 / 1,641 blocks, 0 tokenization failures.
- [x] **Fine-tune `music-small-800k`** via a hand-written HF `transformers` loop. Fixed measurement artifacts (dropout/running-avg) and a learning-rate divergence (→ LR `5e-6` + grad clipping). Full-FT result: overfits after one epoch, best val **0.2434**.
- [x] **LoRA experiment** (`r=8`, LR `2e-4`). Best val **~0.227**, gap held ~0.03–0.05 throughout — beats full FT and doesn't memorize. **LoRA is System A.**
- [x] **Generate from a held-out seed; ear-validate.** LoRA clearly most "like me." Found + fixed the repetition-collapse failure (→ `top_p=0.90`, ~11s horizon).
- [x] **First objective metric** (interval histogram) built, debugged (collapse filter), and interpreted.
- **Done when:** System A produces stylistically plausible, novel continuations and you have baseline metric numbers. ✅ *(Novelty numbers pending — see Week 4.)*

### Week 3 — build retrieval (System B)
- [ ] Build the retrieval index over your corpus (v1 feature-based; reuse the interval signature + Week 2 histogram tooling).
- [ ] Wire retrieval → conditioning into generation (likely `controls=`).
- [ ] Generate **matched** outputs: same seeds through A and B (batch-generate + collapse-filter).
- **Done when:** you have paired A/B outputs for a fixed seed set.

### Week 4 — evaluate + write
- [ ] Finish the objective metrics: pitch-class + rhythm histograms, a distance number / FMD, and the **transposition-aware novelty check**.
- [ ] Blind, multi-sample listening test (you + others); objective metrics for both systems.
- [ ] Analyze: does RAG help? On which dimensions? Failure modes?
- [ ] Write the paper (§12). Include negative/mixed results honestly.
- **Done when:** the paper draft is complete with results, figures, discussion.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Overfitting / memorizing your corpus | **LoRA (decisive)** + augmentation + val early-stopping + novelty metric. Full FT overfits in 1 epoch; LoRA doesn't. |
| Environment / dependency breakage | Pin versions (`transformers==4.29.2`, `peft==0.4.0`); verify with functional import tests. |
| Evaluator bias | Blind A/B, extra musicians. |
| Scope creep | Piano-solo melody extraction & roster-based fine ID kept in the parking lot. |
| RAG just retrieves near-copies | Transposition-aware novelty metric; tune retrieval k and conditioning strength. |
| Silent data loss in the pipeline | "List before you write" + flag-don't-write + reconciliation checks. |
| Small validation piece count (7) | Pair objective metrics with the listening test; don't rely on val loss alone. |
| **Intermittent generation collapse** *(new)* | `top_p=0.90` still collapses ~20% of the time (repetition trap). Batch-generate + auto-filter collapsed runs (note-count / interval-0 threshold); report the rate. Future: tune `top_p` / shorten horizon (parking lot). |
| **Single-sample validation hides intermittent failures** *(new)* | Validate anything about *generation* on a batch, never one clip (this bit us on both `top_p` choice and the interval histogram). |
| **A single metric misreads style** *(new)* | Multi-metric design (interval + pitch-class + rhythm + FMD + listening) — one interval histogram missed the audible LoRA-vs-pretrained difference. |

## 12. Paper outline

1. Introduction & motivation (personal-style generation, the RAG question)
2. Related work (AMT/infilling, symbolic music gen, RAG for music)
3. Method (data pipeline, tokenization/packing, LoRA fine-tuning, retrieval design, A vs. B setup)
4. Experiments (metrics, listening test protocol)
5. Results (objective + subjective, novelty analysis; incl. full-FT-vs-LoRA and generation failure-mode findings)
6. Discussion, limitations, future work (the conditioning cascade; generation robustness)
7. Conclusion

## 13. Learning resources

1. Andrej Karpathy — "Let's build GPT from scratch."
2. MidiTok docs — tokenization concepts.
3. `anticipation` package README + Colab — the base model.
4. HuggingFace `peft` / LoRA docs.
5. Papers: Anticipatory Music Transformer (Thickstun et al., 2023); RAG of Symbolic Music with LLMs (Jonason et al., 2023); VMB (2024); MidiCaps (Melechovsky et al., 2024) & text2midi (Bhandari et al., 2025); Fréchet Music Distance (Retkowski et al., 2024); LoRA (Hu et al., 2021).

## 14. Parking lot (stretch goals — only after the core works)

- **The conditioning cascade (north star).** v1 = the **melody** stage. Future stages add harmony → bass → percussion, each **conditioned on everything before it** (sequential conditioning). AMT's `controls=`/`ops.combine` supports this natively (the `controls=` arg is confirmed in the `generate` signature). `routing.py` already tags every track's role; the shelved score-roster extractor is infrastructure for these stages.
- **Generation robustness** *(new, Week 2).* `top_p=0.90` collapses ~20% of the time (repetition trap; correlates with long runs → drift). Current mitigation: filter. Future: lower `top_p`, shorter horizon, or a repetition guard.
- **Early-stopping `min_delta`** *(new).* Plateau noise (~0.001 "improvements") kept resetting patience on the LoRA run. Require a meaningful improvement threshold.
- **LoRA rank sweep** *(new).* `r=8` beat full FT. Try `r=4` (tighter) and `r=16` (more capacity) — capacity vs. memorization tradeoff. Trainable count scales linearly with `r`.
- **Distance number / FMD** *(new).* Collapse each style histogram into a single "distance from corpus" for a citable A-vs-B number.
- **Piano-solo melody extraction.** `album`, `prelude_in_c#_minor` excluded (skyline can't split melody from chordal piano).
- **Density-aware chunk floor;** **stratified train/val split** (from Week 1).
- Text → MIDI; multi-instrument infilling; CLaMP semantic retrieval; interactive demo.

## 15. Decisions log

- *(Week 0)* Primary task = phrase→MIDI infilling/continuation; base model = AMT small; fine-tune (not from scratch); RAG over personal corpus.

**Week 0 session (data + setup):** export settings frozen (MIDI Type 1, PPQ 960, unfold repeats, GM sounds, one file per piece); pieces <~20s set aside as seed prompts; 49 pieces; inventory switched `music21`→`pretty_midi`; **v1 = melody model**; v1 = stage one of a **conditioning cascade** (sequential); dropped **Ex Machina**; novelty metric must be **transposition-aware**; training-tooling gap noted; **environment = Python 3.11 in `venv311`**.

**Week 1 session (routing / harvest / chunk / augment / split):** score-roster extractor built then shelved; pivoted to order-independent per-track routing; register threshold sorts ambiguous families; strings added; piece-level overrides for tuba concertos; three harvest gates (length, skyline, transposition-invariant de-dupe); **462 melodies from 46 pieces**; chunk (15s/50% overlap) → 5,515; augment (12 keys) → 66,122; split by whole piece (seed 42) → 42,967 train / 23,155 val (7 val pieces).

**Week 2 session (tokenization bridge):**
- Tokenized all 66,122 chunks with `midi_to_events` → one provenance-preserving `.txt` per chunk (`tokenized/train`, `tokenized/val`). **0 failures** (clean Week 1 input). Filenames preserved verbatim for Week 3 retrieval.
- **Packed** into fixed 1024-token blocks: stream chunks into a **triple-SEP-walled ribbon** (SEP=55025 ×3 between chunks), slice every **1023**, prepend **AR=55026** to each block, **drop the tail** (~0.02%; no PAD token exists in the vocab). **Separate ribbons per split** to prevent a train/val seam leak. → **3,084 train / 1,641 val blocks.**
- All constants read from the *installed* `anticipation.vocab`, not memory/paper.

**Week 2 session (fine-tuning):**
- Model confirmed: 128M params, vocab 55028, context 1024, embedding (55028, 768). "800k" = steps.
- **`token_out_embeddings` warning investigated twice, proven benign** (authors load it identically; head is re-trained during FT anyway).
- Baselines (measured): val **1.6387**, train **1.4808** (start close; ~0.16 offset from 7-piece val small-sample luck).
- **Two measurement bugs fixed:** train loss must be measured like val — through `evaluate()` (eval-mode, single-shot) — else dropout (train-mode) + running-average distortion make the gap read negative. Clean diagnostic flipped gap −0.12 → +0.058; also confirmed **no leak**.
- **Learning-rate divergence:** `5e-5` blew up mid-epoch-1. Cause = LR too high for a tiny redundant corpus (fast early drop was the warning). Fix: **LR `5e-6` + gradient clipping (`max_norm=1.0`)**.
- **Full fine-tune result: overfits after one epoch.** Best val **0.2434** (epoch 0); train falls / val rises thereafter; early-stop at epoch 3. Save-on-best + patience=3 worked.

**Week 2 session (LoRA — System A chosen):**
- Pinned **`peft==0.4.0`** to protect `transformers==4.29.2`; verified by functional import test.
- LoRA = freeze base, train low-rank adapters (B·A) in attention. `r=8, alpha=16, dropout=0.05, target=["c_attn"]` → **294,912 trainable params (0.23%)** (traceable: 12 layers × (768×8 + 8×2304)).
- Clarified: LoRA constrains *dimensionality*, not *step size* → LR *raised* to **`2e-4`**.
- **Result: best val ~0.227 (epoch ~10), beating full FT's 0.243.** Gap held ~0.03–0.05 through the whole descent (no memorization); overfitting turn postponed ~10 epochs. **LoRA is System A.**
- Logged: add `min_delta` to early stopping (plateau noise reset patience).

**Week 2 session (generation):**
- Three-way listen (pretrained / full-FT / LoRA), same seed (`fluteSolo1`) + same random seed. **LoRA clearly most "like me"** (winding/conjunct vs. base model's disjunct).
- **Repetition-collapse finding:** clean melody ~11–13s, then degenerates into stacked repeated notes. Localized to **sampling, not training** (same weights generate clean melody earlier).
- `generate` exposes only `top_p` (verified signature; no `repetition_penalty`). **Sweep → `top_p=0.90` generates clean** (0 overlaps/chords); 0.95 and 0.98 fall off a non-monotonic "collapse cliff." **Recipe: LoRA, `top_p=0.90`, ~10s clips.**

**Week 2 session (first objective metric):**
- **Interval histogram** built (`interval_counts` → `histogram` → aggregate; **absolute** intervals = step-vs-leap magnitude, direction discarded). Reference = **462 original melodies** (73,383 intervals).
- **Scaled up** to 20 generations/model (varied random seed, matched across models) for trustworthy distributions.
- **Phantom-bug caught:** raw LoRA histogram showed 58% repeated-notes — traced to **4/20 collapsed generations** (~340 notes each) swamping the aggregate. → **`top_p=0.90` collapse is intermittent (~20%).** Added a **collapse filter** (note-count > 200 or interval-0 frac > 0.4); kept 15/20 pretrained, 16/20 LoRA.
- **Clean result:** LoRA interval-0 0.582 → 0.002 (ear vindicated). But filtered pretrained ≈ LoRA on interval distribution — both over-concentrate on whole-steps (~0.69 vs corpus 0.369), under-represent variety — *despite* the subjective LoRA preference. **A single interval histogram is too coarse to capture audible style; validates the multi-metric plan.**

- *(add as you go…)*
