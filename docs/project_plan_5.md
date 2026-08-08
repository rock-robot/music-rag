# Project Plan — Personal-Style Symbolic Music Generation with Retrieval Augmentation

*A 4-week summer research project. Living document — update the "Decisions log" and "Parking lot" as you go.*

*Last updated: end of Week 4 (all objective metrics built; listening study run; the A/A′/B result is in).*

---

## 1. One-line summary

Fine-tune a pretrained MIDI model on my own catalog so it continues/infills musical phrases **in my style**, then test whether **retrieving motifs from my own corpus** (RAG) improves the results compared to the fine-tuned model alone.

## 2. Research question

> Does retrieval augmentation — conditioning generation on similar phrases drawn from the composer's own corpus — improve the *stylistic fidelity* and *musical quality* of phrase continuations, compared to a fine-tuned model with no retrieval?

**The answer (Week 4): no.** Across three objective metrics (each against five corpus-reference constructions) and a blinded two-axis listening test, no method found retrieval to improve on a length-matched control. Every point estimate favoured the control; none of the listening-test differences reached significance. The clean comparison is three-way — **A vs. A′ vs. B** — because prepending retrieved material also lengthens the prompt, a confound the A′ control isolates (see §4).

## 3. Locked-in decisions (the spec)

| Dimension | Decision |
|---|---|
| Primary task | Phrase → MIDI continuation / infilling |
| Conditioning | Seed MIDI phrase (no text captions needed) |
| Training strategy | Transfer learning: fine-tune a pretrained model (not from scratch) |
| Novel contribution | Retrieval from the composer's own corpus, compared with/without, **with a length-matched control** |
| Data | **46 harvestable pieces** → **462 melody lines → 5,515 chunks → 66,122 augmented → 42,967 train / 23,155 val (by piece) → 3,084 / 1,641 packed 1024-token blocks.** |
| Retrieval index | **3,583 un-augmented (`_t+00`) train-split chunks** (39 of 46 pieces). Excludes val (leak) and transpositions (identical vectors). |
| v1 target | **Melody model**: continue/infill a single extracted melodic voice. Stage one of a planned conditioning cascade (§14). |
| Base model | `stanford-crfm/music-small-800k` (Anticipatory Music Transformer, 128M params, GPT-2 arch.) |
| System A | **LoRA fine-tune** (`r=8, alpha=16, target=["c_attn"]`, ~295K trainable). Beat full FT (val ~0.227 vs 0.243). |
| Generation recipe | LoRA adapter, `anticipation.sample.generate`, **`top_p=0.90`**, ~10s clips, collapse-filter every batch. |
| Conditioning mechanism | **Prepend** retrieved phrases into `inputs=` before the seed; `start_time` = seed end. **NOT `controls=`** (that is the anticipatory/cascade mechanism). |
| Retrieval hyperparameters | `k=2`, `RETRIEVED_SECONDS=2.5`, `gap=0.25s`, `max_per_piece=1`, **octave-only alignment**. Bounded by the ~15s prompt budget. |
| Compute | RTX 4090 (24 GB), 64 GB RAM, Windows + WSL2 |
| Python | **3.11** (`venv311`). `transformers==4.29.2`, `peft==0.4.0` (both pinned), `scipy` for the stats. |
| Deliverable | Written research report / paper |
| Evaluator | The composer (blinded) + 14 additional listeners via a web-based listening study |

## 4. The systems compared

**System A — Baseline (no retrieval).** `seed → fine-tuned (LoRA) AMT → continuation`

**System A′ — Length-matched control.** `[random train chunks] + seed → fine-tuned AMT → continuation`. Prepends **randomly chosen** train chunks, matched in count and length to B. Isolates prompt length / any-prepended-material from *relevant* retrieval.

**System B — RAG variant.** `seed → retrieve k similar phrases → prepend → fine-tuned AMT → continuation`

**Reading the comparison:** A→A′ measures the length/any-material effect; **A′→B measures retrieval's genuine contribution**, length held constant. The headline is **A′ vs. B**. All three share one LoRA generator (enforced in code).

## 5. The central constraint: small data

All five mitigations realized; **LoRA was the decisive one** (full FT overfit after one epoch, LoRA postponed the turn ~10 epochs and reached a lower val floor). Transfer learning, all-12-key augmentation (~12×), overlapping chunking, and by-piece early stopping round out the set.

**Prompt-budget constraint (Week 3):** chunks were re-zeroed to a 0–15s timeline before tokenizing, so the adapter only ever saw time-tokens in ~0–15s. A System B prompt must fit inside ~15s or generation goes out-of-distribution. Caps `k × retrieved-length`; over-budget seeds are excluded and reported.

## 6. Tooling & environment

- **OS layer:** WSL2 (Ubuntu) on Windows.
- **Core:** Python 3.11 (`venv311`), PyTorch (CUDA cu124), `transformers==4.29.2`, `peft==0.4.0`, `scipy`.
- **Symbolic music:** `anticipation` (tokenization + inference), `pretty_midi`, `music21` (Wk0 inventory), `mido` (AMT's internal MIDI writer — `events_to_midi` returns a `mido.MidiFile`, `.save()` not `.write()`), `numpy`.
- **Audio render (Week 4):** `fluidsynth` + FluidR3_GM soundfont, `ffmpeg`/`ffprobe`. Reverb and chorus **off** (`-R 0 -C 0`) — dry rendering keeps the handoff crisp and removes a variable.
- **Listening study (Week 4):** rebuilt in Lovable, backed by Supabase (per-answer rows, anonymous inserts under an RLS insert policy). Audio + manifest served from GitHub raw URLs.
- **Experiment hygiene:** print/CSV/JSON logging; `git`.

> ⚠️ **Tokenizer married to base model.** Use `anticipation`'s `midi_to_events`.
> ⚠️ **`anticipation` has no training code** — hand-written HF `transformers` loop (Wk2).
> ⚠️ **`token_out_embeddings` warning is benign** (investigated twice, Wk2).
> ⚠️ **Pin `peft==0.4.0`** so it can't drag `transformers` forward. Verify with a functional import test.
> 📌 **`generate` returns the whole stream** (prompt + seed + continuation). Always clip to `t > start_time` before any metric or playback.

> 🧱 **Week 4 modules.** `metrics_core.py` (features, `to_hist`, `tv_distance`, `aggregate_pooled`/`aggregate_per_seed`), `metrics_rhythm.py` (refined duration bins + coarse projection), `filters.py` (`stats`, two-mode `is_collapsed`, `tv_nonzero`), `results.py` (load + path-rebuild), `run_metrics.py` (five-reference sensitivity table), `novelty.py` + `novelty_stage2.py` (two-sided transposition-aware overlap), `corpus_reference.py` (five reference constructions), `select_clips.py` (matched-index clip picks), `presentation.py` (seed+pause+continuation assembly, range-aware instrument routing), `render_audio.py` (level-matched MP3s), `manifest_paired.py` (filename-paired, blinded), `wilcoxon_ab.py`, `aaprime_evidence.py`.

## 7. Model selection

**Chosen: fine-tune AMT with LoRA.** Full FT overfits after one epoch; LoRA generalizes better on this tiny corpus. Backups (`SkyTNT/midi-model`, `Moonbeam`) not needed.

## 8. Retrieval design (System B)

Index = 3,583 un-augmented (`_t+00`) train-split chunks. 35-dim hand-built feature vector (interval + duration + contour), pitch-class dropped, cosine similarity, `max_per_piece=1` diversity cap. Conditioning = octave-only alignment, onset+sustain truncation (`window()`), time-splice on a running clock, prompt budget ≤14s.

**Week 4 mechanism finding — the retrieved phrase is usually in a foreign key.** Octave-only alignment (chosen Wk3 to avoid ±25–33-semitone register jams) preserves the phrase's internal key. Measured against each B generation: **mean pitch-class offset 3.0 semitones, only 15.8% of retrieved phrases in the seed's key.** So System B's prompts are *bitonal by construction* — the leading explanation for why B's pitch-class fidelity got **worse**, not better. The v2 fix is pitch-class alignment (match key) followed by octave-only register correction.

## 9. Evaluation plan — all built

**Objective metrics (per-seed, then aggregate — never pool raw).**
- **Interval histogram** (Wk2): filtered pretrained ≈ LoRA — one histogram too coarse to catch the audible difference; motivated the multi-metric design.
- **Pitch-class histogram** (Wk4): the one *non*-transposition-invariant feature, so the reference must be **un-augmented** (augmentation smears pitch-classes toward uniform, erasing the signal).
- **Rhythm/duration histogram** (Wk4): durations are continuous, so bin edges are a design choice. Edges frozen against **corpus percentiles only, before any A′/B comparison**, and defined as a strict **refinement** of the retrieval bins (recoverable by summing) so the two schemes can't contradict.
- **Reference sensitivity (Wk4).** "Distance from corpus" depends heavily on how the corpus reference is built — up to **0.28 TV** between five constructions (whole/excerpt × pooled/averaged, plus a training-distribution `chunk_avg`), which is ~10× the effect size. **Every distance is reported under all five references.** Finding: reference choice materially moves *rhythm* metrics only; pitch-class and interval are near-invariant (density correlates with note length, not tonal content).
- **Collapse filter (Wk4).** Two degeneracy modes: static repetition (`zero_frac > 0.4`) and two-pitch oscillation (`distinct < 3`) — caught by a **union**; neither alone suffices. Because the filter uses `zero_frac`, the interval-0 bin is conditioned, so the interval distance drops bin 0 and renormalizes (`tv_nonzero`). Robustness checked across three filter criteria.
- **Two-sided novelty (Wk4).** Transposition-aware interval n-gram overlap, calibrated against a **ceiling** (val pieces vs. train — genuine novelty) and a **floor** (each system's own intervals shuffled — chance). Frozen at **n=5** (max separation) and **n=8** (7.5× chance = near-damning). Result: all systems overlap the train set at ~the rate of the composer's own held-out pieces — **no memorization**; B's own-prompt overlap ≈ A′'s — **no retrieval regurgitation**. Ordered n-grams match ~25× more than shuffled at n=8 — the model generates real melodic structure, not vocabulary soup.

**Subjective evaluation — the listening study (Wk4).**
- **Blinded forced-choice A′-vs-B**, two axes ("sounds like the same composer" / "the better piece of music"). Chosen over Likert for sensitivity to a ~0.03 effect: each listener is their own control, mirroring the per-seed pairing.
- **15 participants, 108 judgments, all 15 seeds covered (4–10 each).** Tiered lengths (Quick 6 / Standard 12 / Full 15) to widen recruiting; rotating start offset keeps coverage even.
- **Result:** style mean B-preference **0.396** (Wilcoxon p = 0.11), quality **0.353** (p = 0.064). Both lean toward **A′**; neither significant. Underpowered at 15 seeds to certify an effect this small — reported as a **trend toward A′, converging with the objective direction**, not a proven harm.

## 10. Week-by-week plan

### Weeks 0–3 ✅ COMPLETE
Data pipeline (46 pieces → 462 melodies → 66,122 augmented); LoRA fine-tune chosen (val 0.227); `top_p=0.90` generation recipe; feature-based retrieval + prepend conditioning; three-system A/A′/B experiment (~240 clean generations); collapse characterized as condition-independent.

### Week 4 — evaluate + write ✅ COMPLETE
- [x] Pitch-class + rhythm/duration histograms (un-augmented reference; frozen, refined bins).
- [x] Five-reference sensitivity table; distance = TV (FMD deferred).
- [x] Two-sided transposition-aware novelty with calibrated ceiling/floor.
- [x] Two-mode collapse filter + three-criterion robustness check.
- [x] Key-drift mechanism diagnostic (3.0 semitones, 15.8% in-key).
- [x] Matched-index clip selection (`el_mar_idx04` excluded — all 5 B samples collapse).
- [x] Level-matched dry audio render; filename-paired blinded manifest.
- [x] Web listening study (Lovable + Supabase), 15 participants, 108 judgments.
- [x] Per-seed analysis + Wilcoxon on both axes.
- **Done when:** the result is in and defensible. ✅ *(Paper draft is the remaining artifact.)*

**The result, stated once:** *No method — three objective metrics across five references, plus a blinded two-axis listening test — found retrieval to improve on the length-matched control. Every point estimate favoured A′; no listening difference was significant. The A→A′ leg (prepending corpus material improves fidelity) is robust and objective-only. The mechanism for retrieval's non-benefit is measured: octave-only alignment leaves B's prompts in a foreign key 84% of the time.*

## 11. Risks & mitigations

| Risk | Mitigation / outcome |
|---|---|
| Overfitting | LoRA (decisive) + augmentation + early stopping + novelty check (no memorization found). |
| Length confound in A-vs-B | **System A′** (random-corpus, length-matched). Headline is A′-vs-B. |
| Prompt budget vs. adapter range | Retrieval ≤~15s or OOD; over-budget seeds excluded and reported. |
| Reference choice dominates distance | Report all five; finding: only rhythm metrics are sensitive. |
| Filter/metric circularity | `tv_nonzero` drops the conditioned bin; three-criterion robustness. |
| Pooled small-set rates mislead | Per-seed then aggregate — everywhere. |
| Intermittent generation collapse | Batch-generate + two-mode union filter; report the rate. |
| Underpowered listening test | 15 seeds, small effect → reported as a trend, not certified. Honest null-to-negative. |
| **Blinding via opaque filenames** *(Wk4)* | Slot (1/2) in filenames, system only in the private key. Salt on the render hash was left at default — filenames aren't cryptographically opaque, but decoding needs the exact seed names (not in the public repo). Noted as a limitation. |
| **Supabase RLS** *(Wk4)* | Anonymous study → insert/update policies for `anon`; no select policy (public can't read the results table). |
| **Stale manifest durations** *(Wk4)* | `sessions`/manifest embed durations at build time; re-render → rebuild manifest → re-deploy, always in that order. |

## 12. Paper outline

1. Introduction & motivation (personal-style generation, the RAG question).
2. Related work (AMT/infilling, symbolic music gen, RAG for music).
3. Method (pipeline, tokenization/packing, LoRA, feature retrieval, prepend conditioning, the **A/A′/B** design).
4. Experiments (objective metrics + five-reference reporting, prompt-budget constraint, the blinded listening protocol).
5. Results (full-FT-vs-LoRA; the A→A′ context effect; the A′→B non-effect across metrics and listeners; two-sided novelty; condition-independent collapse; the key-drift mechanism).
6. Discussion, limitations, future work (the cascade; key-aware alignment as the retrieval v2; underpowered subjective test; reference-construction sensitivity).
7. Conclusion.

## 13. Learning resources

Karpathy "Let's build GPT"; MidiTok docs; `anticipation` README; HF `peft`/LoRA docs. Papers: Anticipatory Music Transformer (Thickstun 2023); RAG of Symbolic Music with LLMs (Jonason 2023); VMB (2024); MidiCaps (Melechovsky 2024) & text2midi (Bhandari 2025); Fréchet Music Distance (Retkowski 2024); LoRA (Hu 2021).

## 14. Parking lot (stretch goals)

- **The conditioning cascade (north star).** v1 = melody stage. Future: harmony → bass → percussion, sequential conditioning via AMT's `controls=` (confirmed the anticipatory, `CONTROL_OFFSET` slot). `routing.py` already tags every track's role.
- **Key-aware retrieval alignment (v2 — now motivated by data).** Pitch-class-align retrieved phrases to the seed's key, then correct register by octave. Directly addresses the 3.0-semitone / 15.8%-in-key finding — the most concrete future-work item.
- **A″ control** (shuffle A′'s prepended pitches, hold rhythm/time) to separate the context effect into *positional* vs. *melodic-content* — a reviewer will ask.
- **FMD proper** (Retkowski) to replace the placeholder TV distance.
- **Interval n-gram retrieval; CLaMP semantic retrieval; distinct-pitch collapse guard; density-aware chunk floor; stratified split; piano-solo melody extraction; LoRA rank sweep; early-stop `min_delta`.**
- **A-vs-A′ subjective test.** Rendered but never paired into the Lovable app (`COMPARISON=("Aprime","B")`). Would need a re-render + more Full-tier runs; the objective A→A′ result already stands, so low priority.

## 15. Decisions log

*(Weeks 0–3 entries unchanged — see project_plan_4.md.)*

**Week 4 (evaluation + listening study + write-up):**
- **Pitch-class reference must be un-augmented.** It is the one non-transposition-invariant style feature; augmenting the reference smears pitch-classes toward uniform and erases the signal being measured (the mirror image of why augmentation teaches key-invariance in training).
- **Rhythm bins frozen from corpus percentiles before any comparison**, and defined as a strict refinement of the retrieval bins (coarse recoverable by summing) so two binning schemes can't tell contradictory stories. Chose **duration** over **IOI**: extracted orchestral parts have tacet gaps (IOI 99th pct 23s, max 320s) that conflate rhythm with rest structure.
- **Trill/figuration check falsified a hunch.** Short notes (<0.1s = 25% of corpus) are 80% fast *non-ornamental* figuration, only 11% trills — reducing them would delete a core stylistic feature. Kept as a data finding; did not reduce.
- **Collapse filter is two-mode (union).** `zero_frac>0.4` (static) OR `distinct<3` (two-pitch loop). The `distinct<3` guard alone caught nothing (missed 96%-repeat runs among 4–6 pitches); the old `zero_frac` alone missed pure two-pitch oscillations. Because the filter touches interval-0, the interval distance drops bin 0 and renormalizes. Robustness verified across three criteria.
- **Five-reference sensitivity is a reported result, not a hidden choice.** Corpus constructions differ by up to 0.28 TV (~10× the effect). Adopted `chunk_avg` (the training distribution) as primary; report all five. Reference choice moves only rhythm metrics — pitch-class/interval are density-invariant.
- **Aggregation policy is a claim:** `aggregate_pooled` (note-weighted) for the corpus reference; `aggregate_per_seed` (seed-weighted) for the generations. Pooling generations would let a prolific/pathological seed dominate — the recurring "analyze per-seed, never pool raw" lesson, now enforced in two function signatures.
- **Two-sided novelty calibrated against a within-composer ceiling (val vs train) and a chance floor (self-shuffled).** Frozen at n=5 (max separation) and n=8 (7.5× chance). No memorization; no retrieval regurgitation; ordered n-grams match ~25× shuffled at n=8 (real structure, not vocabulary soup). Rare-gram provenance (`len(pieces)==1`) to see which pieces B draws from.
- **Key-drift mechanism measured.** Octave-only alignment → retrieved phrase mean 3.0 semitones off the generation's key, 15.8% in-key. B's prompts are bitonal by construction → explains worse pitch-class fidelity. Motivates key-aware alignment as retrieval v2.
- **`el_mar_idx04` excluded from BOTH analyses.** All 5 of its System B generations collapse under the union filter; it also drove 10/14 collapses in Wk3 and collapses in plain System A → seed-intrinsic. One consistent, documented exclusion across objective and subjective.
- **Matched-index clip selection.** One sample per seed per system, the lowest index surviving in *all* systems — preserves the matched `torch.manual_seed(i)` pairing from Wk3. All 15 usable seeds resolved to sample 0.
- **Audio render: dry, level-matched, range-aware.** FluidSynth reverb/chorus off (crisp handoff); `loudnorm`/`dynaudnorm` to one target (loudness can't cue the choice); instrument assigned **per seed** (not per system — a timbre split inside a pair would be judged instead of the melody), scored by range rather than strict containment, **horn dropped** (its GM patch screamed on high melodies). Calibration excerpts windowed from the **first note**, not t=0 (extracted parts rest at the start), with a duration assert — three identical file sizes had silently shipped empty clips.
- **Listening study: forced-choice over Likert**, two axes, per-seed then Wilcoxon. Blinding = opaque `sNN_slot` filenames + private `PAIRING_KEY.json`; slot randomized per seed so slot order can't reveal system. Rebuilt in Lovable + Supabase; anonymous inserts need an RLS insert policy; no select policy so the public can't read results.
- **Result:** A′→B non-effect confirmed on both objective (all metrics/references favour A′) and subjective (style p=0.11, quality p=0.064, both toward A′) axes. Reported as convergence on a null-to-negative effect, not proven harm. A→A′ remains robust and objective-only.
