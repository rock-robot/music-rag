# Project Plan — Personal-Style Symbolic Music Generation with Retrieval Augmentation

*A 4-week summer research project. Living document — update the "Decisions log" and "Parking lot" as you go.*

*Last updated: end of Week 3 (System B retrieval + conditioning built; three-system A/A′/B experiment run; collapse characterized as condition-independent).*

---

## 1. One-line summary

Fine-tune a pretrained MIDI model on my own catalog so it continues/infills musical phrases **in my style**, then test whether **retrieving motifs from my own corpus** (RAG) improves the results compared to the fine-tuned model alone.

## 2. Research question

> Does retrieval augmentation — conditioning generation on similar phrases drawn from the composer's own corpus — improve the *stylistic fidelity* and *musical quality* of phrase continuations, compared to a fine-tuned model with no retrieval?

This "with vs. without RAG" contrast is the experimental spine of the paper. Everything else exists to make that comparison clean.

**Refinement (Week 3):** the clean comparison is now three-way — **A vs. A′ vs. B** — because prepending retrieved material also lengthens the prompt, a confound that must be controlled. See §4.

## 3. Locked-in decisions (the spec)

| Dimension | Decision |
|---|---|
| Primary task | Phrase → MIDI continuation / infilling |
| Conditioning | Seed MIDI phrase (no text captions needed) |
| Training strategy | Transfer learning: fine-tune a pretrained model (not from scratch) |
| Novel contribution | Retrieval from the composer's own corpus, compared with/without |
| Data | **46 harvestable pieces** → **462 melody lines → 5,515 chunks → 66,122 augmented → 42,967 train / 23,155 val (by piece) → 3,084 / 1,641 packed 1024-token blocks.** |
| **Retrieval index** | **3,583 un-augmented (`_t+00`) train-split chunks** — 39 of 46 pieces. Excludes val (leak) and transpositions (collapse to identical vectors). See §8. |
| v1 target | **Melody model**: continue/infill a single extracted melodic voice. Stage one of a planned conditioning cascade (see §14). |
| v1 melody routing | Pooled by family: any melody-register voice is a candidate. Timbre discarded — the model learns melodic *shape*. |
| Base model | `stanford-crfm/music-small-800k` (Anticipatory Music Transformer, 128M params, GPT-2 architecture) |
| **System A (chosen)** | **LoRA fine-tune** of the base model (`r=8, alpha=16, target=["c_attn"]`, ~295K trainable params). Beat full fine-tuning head-to-head (val ~0.227 vs 0.243). |
| **Generation recipe** | LoRA adapter, `anticipation.sample.generate`, **`top_p=0.90`**, ~10s clips (≤~11s reliable horizon), collapse-filter any batch. |
| **Conditioning mechanism (Week 3)** | **Prepend** retrieved phrases into `inputs=` before the seed; `start_time` = end of seed. **NOT `controls=`** (that is the anticipatory/cascade mechanism — see §8). |
| **Retrieval hyperparameters (Week 3)** | `k=2`, `RETRIEVED_SECONDS=2.5`, `gap=0.25s`, `max_per_piece=1`, **octave-only alignment**. Bounded by the prompt budget (see §8). |
| Compute | RTX 4090 (24 GB), 64 GB RAM, Windows + WSL2 |
| Python | **3.11** (venv `venv311`). `transformers==4.29.2`, `peft==0.4.0` (both pinned). |
| Deliverable | Written research report / paper |
| Evaluator | The composer (blinded) + 1–2 additional musicians |

## 4. The systems you'll compare

**System A — Baseline (no retrieval)** — *built (Week 2).*
`seed phrase → fine-tuned (LoRA) AMT → continuation`

**System A′ — Length-matched control (Week 3, NEW)**
`[random train chunks] + seed → fine-tuned AMT → continuation`
Prepends **randomly chosen** train chunks (not retrieved), matched in count and length to System B. Isolates the effect of *prompt length / any prepended corpus material* from the effect of *relevant* retrieval.

**System B — RAG variant** — *built (Week 3).*
`seed phrase → retrieve k similar phrases → prepend → fine-tuned AMT → continuation`

**Reading the comparison:** A→A′ measures the *length / any-material* effect; **A′→B measures retrieval's genuine contribution**, with length held constant. The headline result is **A′ vs. B**, not A vs. B — comparing A vs. B alone would attribute length effects to retrieval. All three share the *same* LoRA generator (one model object, enforced in code).

## 5. The central constraint: small data

The single biggest design driver. Mitigations, all realized:

1. **Transfer learning.** ✅ Fine-tuned `music-small-800k` (baseline val 1.64 → 0.24).
2. **Data augmentation.** ✅ All 12 keys → 66,122 examples (~12×). *Transposition leaves intervals unchanged but changes pitch-classes — matters for which metrics use augmented vs. raw data (§9), and is why the retrieval index uses **un-augmented** chunks (§8).*
3. **Chunking.** ✅ 462 melodies → 5,515 chunks.
4. **Parameter-efficient fine-tuning (LoRA).** ✅ The decisive mitigation. Full FT overfit after one epoch; LoRA postponed the turn ~10 epochs and reached a lower val floor.
5. **Strict validation split + early stopping.** ✅ By-piece hold-out (7 pieces, seed 42); save-on-best + patience=3.

**New Week 3 constraint — the retrieval prompt budget.** Because every chunk was re-zeroed to a 0–15s timeline before tokenizing, the LoRA adapter only ever saw time-tokens in ~0–15s. A System B prompt (retrieved phrases + seed) must therefore fit **inside ~15 seconds** or generation happens *out of distribution* for the adapter and falls back toward generic base-model behavior. This hard-caps `k × retrieved-length`. See §8.

## 6. Tooling & environment

- **OS layer:** WSL2 (Ubuntu) on Windows.
- **Core:** Python 3.11 (`venv311`), PyTorch (CUDA cu124), `transformers==4.29.2`, `peft==0.4.0`.
- **Symbolic music:** `anticipation` (AMT tokenization + inference), `pretty_midi` (parsing / note-level work), `music21` (Week 0 inventory), `mido` (AMT's internal MIDI writer — note `events_to_midi` returns a `mido.MidiFile`, saved with `.save()`, **not** `pretty_midi`'s `.write()`), `numpy` (feature vectors / cosine).
- **Experiment hygiene:** print/CSV/JSON logging; `git` from day one.

> ⚠️ **Tokenizer pitfall:** the tokenizer is married to the base model. Use `anticipation`'s `midi_to_events`. Integer ranges (time≈0, duration≈10000, note≥11000) are baked into the weights.

> ⚠️ **Training-tooling gap:** `anticipation` does prep + tokenization + inference only — no training code. Fine-tune with a hand-written HF `transformers` loop (Week 2).

> ⚠️ **`token_out_embeddings` warning is benign.** Investigated twice (Week 2), proven harmless. It reappears on every model load — know its shape so a *new* warning stands out.

> ⚠️ **Dependency chain is fragile.** Pin `peft==0.4.0` so it can't drag `transformers` forward. Verify with a functional import test.

> 📌 **AMT vocab:** `SEPARATOR=55025`, `AUTOREGRESS=55026`, `ANTICIPATE=55027`, `CONTEXT_SIZE=1024`, `VOCAB_SIZE=55028`, `NOTE_OFFSET=11000`, `DUR_OFFSET=10000`, `REST=27512`, `CONTROL_OFFSET` (offsets control events into a separate vocab slice).

> 📌 **`generate` signature (verified) + body semantics (read Week 3):** `generate(model, start_time, end_time, inputs=None, controls=None, top_p=1.0, ...)`.
> - Events in `inputs` **before `start_time`** are the prompt; events **after** are silently treated as controls. So "prepend the retrieved phrase" = build one stream `[retrieved][seed]` and set `start_time` to the seed's end.
> - `controls=` puts the model in **`ANTICIPATE` mode** (not `AUTOREGRESS`), offsets events by `CONTROL_OFFSET`, and **clips anything before `DELTA`** — i.e. it is a *future*, time-aligned mechanism, not a "here's earlier music" slot.
> - **`generate` returns the whole stream** (prompt + seed + continuation). Every generation MUST be clipped to `t > start_time` before any metric or listening.

> 🧱 **Week 3 modules.** `features.py` (feature vectors, `piece_name`, `load_notes`, `load_generated_notes`, `clip_and_rezero`, `window`, `cosine`); `build_index.py` (index matrix + provenance JSON); `retrieve.py` (`load_index`, `retrieve` with diversity cap); `condition.py` (`align_to_seed`, `truncate`, `splice`, `build_conditioned_prompt`, `notes_to_midi`); `seeds.py` (frozen stratified seed set); `generate_ab.py` (three-system runner); `check_budget.py` (throwaway prompt-length diagnostic).

## 7. Model selection

**Chosen: fine-tune AMT with LoRA.** Confirmed 128,103,936 params, embedding (55028, 768), context 1024. Full FT overfits after one epoch; LoRA generalizes better on this tiny corpus. Backups (`SkyTNT/midi-model`, `Moonbeam`) not needed.

## 8. Retrieval design (System B) — *built Week 3*

**Index = 3,583 un-augmented (`_t+00`) train-split melody chunks (39 pieces).** Two independent reasons for these filters:
- **Un-augmented:** the similarity metric is (largely) transposition-invariant, so the 12 transpositions of a chunk produce *identical* vectors. Indexing them would collapse the effective top-k (three keys of one phrase instead of three phrases).
- **Train-only:** a val seed must never retrieve its own continuation. Val chunks overlap (50%) their neighbours, so an in-index val chunk would hand the model the answer.
- *Recovered via the `_t+00` trick:* `data/train/*_t+00.mid` is exactly "un-augmented chunks, train pieces only" — both properties from one glob, read off disk rather than recomputed.

**Feature vector (35-dim, hand-built — the "embedding"):**
- Interval histogram (~25 bins, signed, clamped ±12) — melodic shape.
- Duration histogram (~7 log-spaced bins) — rhythmic character.
- Contour (3 bins, down/same/up) — re-emphasizes direction (folded away by Week 2's `abs()`).
- **Pitch-class dropped** for v1: the retrieved phrase is transposed to the seed's register anyway, so matching on key is wasted signal.
- Each block normalized to sum 1, then concatenated (so each dimension votes equally regardless of note count). Compared by **cosine similarity** (scale-invariant → a short seed and a long chunk compare fairly).
- Validation: corpus self-similarity mean **0.52** (well-spread, discriminative — not everything-looks-alike); top-k retrievals sound like relatives *by ear* and **beat the volume prior** (a small piece won retrievals over `paradise_lost`, which is 33% of the index).

**Retrieval (`retrieve`):** cosine → `argsort` → diversity walk with `max_per_piece=1` (one chunk per source piece, else one over-represented piece monopolizes the top-k). Provenance survives in filenames for the Week 4 novelty check.

**Conditioning (`condition.py`):**
- **Align:** transpose the retrieved phrase to the seed's register by **octave-only** median-pitch match (Week 3 decision — full-semitone matching produced ±25–33 shifts that jammed phrases against the pitch ceiling and correlated with collapse; octave-only preserves the phrase's internal key while fixing register).
- **Truncate:** `window()` clips each retrieved phrase to `RETRIEVED_SECONDS=2.5` — clipping **both onsets and sustains** (a sustained note slipping past the onset cut was the "fast-vs-slow bias," appearing a 4th time).
- **Splice:** lay phrases on a running clock (each chunk is re-zeroed, so naive concat would stack them), `gap=0.25s` between, seed last, `start_time` = seed end + 0.05 (quantization pad).
- **Budget:** `build_conditioned_prompt` returns an `ok` flag; prompts over `MAX_PROMPT_SECONDS=14` are **excluded and reported**, not silently generated. 4 of 20 seeds excluded → **16 seeds run**.

**v2 (parking lot):** interval **n-grams** (order-sensitive — histograms are bags: `{+2,+2,−5}` = `{−5,+2,+2}`), CLaMP semantic retrieval.

## 9. Evaluation plan

**Objective metrics** — *interval histogram built Week 2; the rest are Week 4.*
- Style-distribution similarity: pitch-class, interval, rhythm-duration histograms vs. the **un-augmented `corpus/`** (462 original melodies).
  - *Week 2 finding:* filtered pretrained ≈ LoRA on the interval histogram *despite* a clear subjective LoRA preference — one histogram is too coarse; validates the multi-metric design.
- **Fréchet Music Distance (FMD)** / a single distance-from-corpus number. *(Not yet built.)*
- **Novelty / anti-plagiarism check** *(critical; not yet built):* **TRANSPOSITION-AWARE** n-gram overlap. **Two-sided for System B** (Week 3): compare each generation against (a) **the retrieved phrases in its own prompt** (retrieval regurgitation) *and* (b) **the train set** (memorization). System B is *expected* to score higher on train-overlap because it was shown corpus material — report both so the two effects are separable. Measure on the **clipped** generation only.

**Subjective evaluation**
- Blind, multi-sample A/B(/A′) listening test. Likert (1–7): overall quality, "sounds like my style." 1–2 extra musicians.

> **Analysis discipline (Week 3, learned twice):** compute metrics **per-seed, then aggregate — never pool raw generations.** Pooled rates are dominated by their worst member (one seed drove 10 of 14 collapses; the Week 2 histogram was swamped by 4 runaway files).

> **Generation for metrics:** always **batch-generate and collapse-filter**. `top_p=0.90` collapses intermittently. Collapse is trivially detectable (note-count > 200 or interval-0 fraction > 0.4). Aggregate over `ok` runs; report the rate.

## 10. Week-by-week plan

### Week 0 — scope + setup ✅ COMPLETE
Data exported (49→46 pieces), inventoried, scoped to a melody model; environment working; research question + metrics frozen.

### Week 1 — ML foundations + first inference + data pipeline ✅ COMPLETE
AMT inference calibrated; pipeline built (routing → harvest → chunk → augment → split). 46 pieces → 462 melodies → 5,515 chunks → 66,122 augmented → 42,967 train / 23,155 val.

### Week 2 — fine-tune the baseline (System A) ✅ COMPLETE
Tokenize + pack → 3,084 / 1,641 blocks; full FT (best val 0.243) vs. **LoRA (best val 0.227, chosen)**; generation ear-validated, `top_p=0.90` recipe, collapse failure characterized; first objective metric (interval histogram) built.

### Week 3 — build retrieval (System B) ✅ COMPLETE
- [x] **Feature-based retrieval index** over the corpus: 35-dim hand-built vector (interval + duration + contour), cosine over 3,583 un-augmented train chunks, diversity cap. Validated by ear and against the volume prior.
- [x] **Conditioning pipeline:** prepend (not `controls=`, confirmed from source), octave-only register alignment, onset+sustain truncation, time-splice, prompt-budget enforcement (`≤14s`).
- [x] **Frozen stratified seed set** (`seeds.py`): 20 candidates round-robin across the 7 val pieces (viability-filtered to ≥6 notes/6s), note counts recorded for density stratification. 4 budget-excluded → 16 run.
- [x] **Three-system experiment** (A / A′ / B), matched seeds and random seeds, clip-and-rezero, split collapse classifier, provenance logged. ~240 clean generations (5 samples × 16 seeds × 3 systems).
- **Done when:** paired A/A′/B outputs for a fixed seed set. ✅ (`results_A.json`, `results_Aprime.json`, `results_B.json`.)

**Week 3 collapse result (per-seed):** collapse is **condition-independent**. Pooled rates (A 6.2% / A′ 2.5% / B 8.8%) were a mirage — **one seed (`el_mar_idx04`) produced 10 of 14 total collapses**, and it collapses even in plain System A (seed only), so the cause is seed-intrinsic, not retrieval. Excluding it: **A 1.3% / A′ 1.3% / B 2.7%** — no meaningful difference between conditions at n=75. Conditioning (random or retrieved) does not materially change stability. *Report the outlier as a failure case; don't let it distort the comparison.*

### Week 4 — evaluate + write
- [ ] Finish objective metrics: pitch-class + rhythm histograms, distance/FMD, **two-sided transposition-aware novelty**.
- [ ] Run all metrics **per-seed** on the `ok` generations of A / A′ / B; the headline is **A′ vs. B**.
- [ ] Blind, multi-sample listening test.
- [ ] Analyze: does retrieval help, on which dimensions, with what failure modes? Stratify by seed density.
- [ ] Write the paper (§12). Include the null collapse result and negative/mixed findings honestly.
- **Done when:** the paper draft is complete with results, figures, discussion.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Overfitting / memorizing your corpus | **LoRA (decisive)** + augmentation + val early-stopping + novelty metric. |
| Environment / dependency breakage | Pin versions; verify with functional import tests. |
| Evaluator bias | Blind A/B(/A′), extra musicians. |
| Scope creep | Piano-solo extraction, roster-based fine ID, n-gram retrieval kept in the parking lot. |
| RAG just retrieves near-copies | Transposition-aware **two-sided** novelty metric; tune `k` and truncation. |
| Silent data loss in the pipeline | "List before you write" + flag-don't-write + reconciliation asserts. |
| Small validation piece count (7) | Pair objective metrics with the listening test. |
| Intermittent generation collapse | Batch-generate + auto-filter; report the rate. |
| **Length confound in A-vs-B** *(Week 3)* | **System A′** (random-corpus, length-matched control). Headline is A′-vs-B. |
| **Prompt budget vs. adapter range** *(Week 3)* | Retrieval material must fit within ~15s (the adapter's training-time range) or generation goes out-of-distribution. Caps `k × length`; over-budget seeds excluded and reported. |
| **Retrieval-corpus imbalance** *(Week 3)* | `paradise_lost` (4 movements) is ~33% of the index; argmax retrieval exposes it where averaging (FT) absorbed it. `max_per_piece=1` cap; movements share motifs, so watch it in the novelty check. |
| **Extreme transposition on alignment** *(Week 3)* | Full-semitone matching produced ±25–33 shifts (register-jamming, collapse-correlated). Switched to **octave-only** alignment. |
| **Multi-instrument generations** *(Week 3)* | `generate` can emit >1 instrument track; pooling must skyline (highest note per onset) so simultaneous notes don't create phantom unison intervals in the metrics. |
| **Pooled small-set rates mislead** *(Week 3)* | Analyze per-seed, then aggregate. One bad seed swings a pooled rate by points. |
| **Trill vs. collapse ambiguity** *(Week 3)* | The count-based collapse detector can mistake a long trill for a collapse (or miss it). Spot-check collapse verdicts by ear; note as a known false-positive risk. |

## 12. Paper outline

1. Introduction & motivation (personal-style generation, the RAG question)
2. Related work (AMT/infilling, symbolic music gen, RAG for music)
3. Method (data pipeline, tokenization/packing, LoRA fine-tuning, feature-based retrieval, the prepend conditioning mechanism, the A/A′/B design)
4. Experiments (metrics, listening protocol, prompt-budget constraint)
5. Results (objective + subjective, novelty analysis; full-FT-vs-LoRA, generation failure modes, condition-independent collapse)
6. Discussion, limitations, future work (the conditioning cascade; generation robustness; retrieval-corpus imbalance)
7. Conclusion

## 13. Learning resources

1. Karpathy — "Let's build GPT from scratch."
2. MidiTok docs — tokenization concepts.
3. `anticipation` README + Colab — the base model.
4. HuggingFace `peft` / LoRA docs.
5. Papers: Anticipatory Music Transformer (Thickstun et al., 2023); RAG of Symbolic Music with LLMs (Jonason et al., 2023); VMB (2024); MidiCaps (Melechovsky et al., 2024) & text2midi (Bhandari et al., 2025); Fréchet Music Distance (Retkowski et al., 2024); LoRA (Hu et al., 2021).

## 14. Parking lot (stretch goals — only after the core works)

- **The conditioning cascade (north star).** v1 = the **melody** stage. Future stages add harmony → bass → percussion, each conditioned on everything before it (sequential conditioning). **AMT's `controls=` is the mechanism** — confirmed Week 3 to be the anticipatory, time-aligned slot (`ANTICIPATE` mode, `CONTROL_OFFSET`), exactly what "write a part around the parts already there" needs. `routing.py` already tags every track's role; the shelved score-roster extractor is infrastructure for these stages.
- **Interval n-gram retrieval (v2).** Histograms are order-blind; n-grams capture melodic *order*. The first lever if retrieval quality is the bottleneck.
- **Generation robustness.** `top_p=0.90` collapses intermittently; correlates with long runs / drift. Future: lower `top_p`, shorter horizon, repetition guard.
- **A distinct-pitch collapse check.** `len(set(pitches)) < 3` catches degenerate loops the interval-0 check misses, without flagging varied trills.
- **Pitch-class in the feature vector (low-weighted).** Dropped for v1; arguable that key-relationship is part of style.
- **Early-stopping `min_delta`; LoRA rank sweep (r=4, r=16); distance-number / FMD.**
- **Piano-solo melody extraction** (`album`, `prelude_in_c#_minor`); **density-aware chunk floor; stratified train/val split.**
- Text → MIDI; multi-instrument infilling; CLaMP; interactive demo.

## 15. Decisions log

- *(Week 0)* Primary task = phrase→MIDI infilling/continuation; base model = AMT small; fine-tune (not from scratch); RAG over personal corpus.

**Week 0 (data + setup):** export settings frozen; pieces <~20s set aside as seed prompts; 49 pieces; inventory `music21`→`pretty_midi`; **v1 = melody model**; v1 = stage one of a conditioning cascade; dropped **Ex Machina**; novelty metric must be transposition-aware; environment = Python 3.11 `venv311`.

**Week 1 (pipeline):** order-independent per-track routing; register threshold; strings added; tuba-concerto overrides; three harvest gates; **462 melodies from 46 pieces**; chunk → 5,515; augment → 66,122; split by piece (seed 42) → 7 val pieces.

**Week 2 (fine-tune):** tokenize + pack → 3,084 / 1,641 blocks; measurement bugs fixed; LR `5e-6` + grad clipping; full FT overfits in 1 epoch (val 0.243); **LoRA chosen (val 0.227)**; `top_p=0.90` recipe; interval histogram + collapse filter.

**Week 3 (retrieval + conditioning):**
- **Conditioning mechanism = prepend via `inputs=`, NOT `controls=`.** Confirmed by reading the `generate` source: `controls=` triggers `ANTICIPATE` mode (untrained for our adapter), offsets by `CONTROL_OFFSET`, and clips past events — it is the *future-aligned cascade* mechanism, not RAG. Prepending = build `[retrieved][seed]` in `inputs`, set `start_time` to the seed's end. **`generate` returns the whole stream; always clip to `t > start_time` before metrics.**
- **Retrieval index = 3,583 un-augmented (`_t+00`) train chunks.** Un-augmented (transpositions → identical vectors → collapsed top-k) and train-only (val seed must not retrieve its own continuation). Recovered via the `_t+00` glob (both properties, read off disk). `piece_name()` = single source of truth for piece identity (parse the `_idx` landmark from the *filename*, not the path).
- **Feature vector = 35-dim** (interval + duration + contour), per-block normalized, cosine. Pitch-class dropped (phrase is transposed to seed's key anyway). Corpus self-similarity mean 0.52 (discriminative); retrieval beats the volume prior; top-k sound like relatives by ear.
- **`retrieve` diversity cap `max_per_piece=1`** — uncapped, one piece monopolized the top-5 (near-copies); cap costs ~0.03 similarity for genuine diversity and restores a real ranking gradient.
- **Alignment = octave-only** median-pitch match. Full-semitone matching gave ±25–33 shifts (register-jamming against the 21–108 clamp; correlated with collapse). Octave-only preserves the phrase's internal key.
- **`window()` clips onsets AND sustains** — single source of truth for time-windowing (`load_seed`, `truncate` both use it). Onset-only clipping let sustained notes drag phrases past their nominal length (the "fast-vs-slow bias," 4th appearance).
- **Prompt budget ≤14s** (adapter trained only on 0–15s timelines). Over-budget prompts excluded and reported, not silently generated out-of-distribution. Retrieval hyperparameters (`k=2`, `2.5s`, `gap 0.25`) chosen to fit the budget. 4 of 20 seeds excluded → 16 run.
- **System A′ added** — random train chunks, length-matched to B, deterministic per seed. Isolates length from retrieval. Headline comparison = A′ vs. B. B's budget exclusions applied to all three systems so the seed set is identical.
- **Frozen stratified seed set** — round-robin across the 7 val pieces (not plain random, which the file-heavy val imbalance would skew), viability-filtered (≥6 notes/6s), note counts stored for density stratification.
- **`load_generated_notes` (flatten + skyline)** — generations can emit >1 instrument; pooling must skyline so simultaneous notes don't create phantom unison intervals. `load_notes`'s single-instrument assert correctly fired on this and localized it.
- **Collapse is condition-independent.** Per-seed analysis: one seed (`el_mar_idx04`) drove 10 of 14 collapses and collapses in plain System A → seed-intrinsic. Ex-outlier: A 1.3% / A′ 1.3% / B 2.7%, no meaningful difference. **Analyze per-seed, then aggregate — never pool raw** (pooled rates are dominated by the worst member).
- **`events_to_midi` returns a `mido.MidiFile`** → save with `.save()`, not `pretty_midi`'s `.write()`. Three libraries now touch MIDI; the file is the interchange format, so every crossing quantizes — verify round-trips.
- *(add as you go…)*
