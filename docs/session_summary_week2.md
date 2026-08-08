# Session Summary — Week 2 (Tokenization Bridge → Fine-Tuning → Generation → First Metric)

*A narrative of how we got from "a prepared but untokenized corpus" to two fine-tuned models, an ear-validated (and debugged) generation process, and the first working objective metric. Focuses on the reasoning behind each major decision — and, just as importantly, on the anomalies we chased down rather than waved through.*

---

## 1. The bridge: turning prepared music into model input

Week 1 left the corpus *ready* but not *readable* by the model: 66,122 augmented MIDI chunks, still `.mid` files. The model eats one thing only — a flat sequence of integers — so the whole of Week 2 opened with **tokenization**, the "bridge" the plan named.

Two functions from the `anticipation` package do it: `midi_to_events` (MIDI → integer events) and `events_to_midi` (back again). The core concept that anchored everything: **the tokenizer is married to the base model.** Those exact integer ranges — time near 0, duration near 10000, note starting at 11000 — are baked into the pretrained model's 128M weights. Use a different tokenizer (MidiTok, our own) and every integer would mean something else; the model would read noise and loss would never drop. This is why "mixing tokenizers silently breaks everything," and why we only ever used `midi_to_events`.

We built it in three increments, inspecting the artifact at each stage:

1. **Tokenize one chunk, round-trip it.** Confirmed the encoding (each note = 3 integers: time, duration, note) and that we could read its output.
2. **Batch over the corpus.** Looped `midi_to_events` over all 66,122 files with fail-soft discipline (`try/except` per file, flag-don't-write on empty, end-of-run reconciliation). Result: **0 failures** — clean input upstream (Week 1's gates) meant clean output downstream. Wrote one `.txt` per chunk, filename preserved verbatim, so **provenance survives tokenization** (essential for Week 3 retrieval and the novelty check).
3. **Pack into fixed 1024-token blocks.** The model trains on uniform blocks, but chunks are wildly varying lengths (24 to 200+ tokens). So we streamed all chunks end-to-end into one ribbon, dropped a **triple-SEP wall** (three `55025`s, confirmed from `anticipation.vocab`) between chunks so the model can see phrase boundaries, sliced every **1023** tokens, and prepended the **`AR` control code (`55026`)** to each block (making it a valid, independently-shufflable training example). Dropped the leftover tail (~0.02%). Packed **train and val into separate ribbons** so no single block could straddle the train/val boundary and leak.

Realized: **3,084 train blocks / 1,641 val blocks.** Every constant (SEP, AR, context length 1024, vocab 55028) was read from the installed package, not trusted from memory or the paper.

## 2. Loading the model — and a warning we investigated twice

`AutoModelForCausalLM.from_pretrained("stanford-crfm/music-small-800k")` loaded a **128,103,936-parameter** GPT-2, vocab **55028**, context **1024**, embedding table **(55028, 768)** — every number matching what we'd packed against. A naming note logged: "800k" is training *steps*, not parameters; "small" is the size class.

The load threw a warning: `token_out_embeddings` weights "not used." The checkpoint ships a *separate, untied* output head; HuggingFace's tied-by-default `GPT2LMHeadModel` discarded it and tied the output to the input embeddings instead. We **raised the alarm, investigated, and stood down**: the authors' own README loads the model this identical way, Week 1 inference had run through it fine, and for *fine-tuning* the output head gets re-optimized regardless. Verdict: expected and benign — later *proven* benign by a working fine-tune (val 1.64 → 0.24). The meta-lesson we kept returning to all week: **investigate warnings, then decide — neither ignore nor panic.**

Baselines (honest, measured over the sets): **val 1.6387, train 1.4808.** They start close (the base model is equally ignorant of both), with a small ~0.16 offset from small-sample luck (val is only 7 pieces). Measuring both baselines mattered — it set the starting line so any train/val gap that opened later would be attributable to fine-tuning, not to the sets differing.

## 3. The training loop — and the bugs it took to make it honest

We built the loop as four moves (forward → backward → optimizer step → zero-grad), proved the machinery with a **smoke test** (deliberately overfitting 8 blocks until loss crashed toward zero — the pass condition is that the model *can* memorize, proving every wire is connected), then ran it for real. Two rounds of debugging made the numbers trustworthy:

**Measurement artifacts.** The first real run showed a *negative* train–val gap (val below train), which is backwards. We over-alarmed, then isolated it: two artifacts, both making *measured* train loss look worse than val. (1) Train loss was a running average across an improving epoch; val was measured once at the end. (2) Train was measured in `.train()` mode (dropout on); val in `.eval()` mode (dropout off). The fix was one clean change — measure **both** through `evaluate()` (eval-mode, single-shot). A clean diagnostic confirmed it: the gap flipped from −0.12 to **+0.058** (train 0.167, val 0.225), which *also* proved there's no train/val leak (val sits cleanly above train when measured identically).

**Learning-rate divergence.** At `5e-5`, the model cratered in epoch 0 then *blew up* mid-epoch-1 (val exploded back to ~1.71). We initially chased exotic theories (eval mode, tied heads); a 10-second test ruled them out, and Occam won: **the learning rate was too high.** The valley geometry: big steps make fast early progress far from the minimum, then overshoot once close — so the suspiciously fast epoch-0 drop was itself the warning sign. Fix: **LR → `5e-6`** plus **gradient clipping** (`max_norm=1.0`, a seatbelt against any single monster step). Stable thereafter.

**The result of the full fine-tune.** With honest measurement and a stable LR, the curve was textbook:

```
epoch 0   train 0.151   val 0.243   ← best, saved
epoch 1   train 0.111   val 0.275
epoch 2   train 0.092   val 0.305
epoch 3   train 0.069   val 0.342   → early stop (patience 3)
```

**Finding: the model overfits after a single epoch.** Val bottomed at epoch 0 and rose every epoch after while train kept falling — the scissors of memorization. Save-on-best captured epoch 0 and never overwrote it; patience halted the run. System A (full fine-tune) exists, best val **0.2434**, baseline 1.64 → 0.24.

## 4. LoRA — a controlled experiment, and a better model

The "overfits after one epoch" finding was the perfect motivation for LoRA (parameter-efficient fine-tuning): if full freedom (128M weights) let the model memorize 46 pieces, *constrain* how much it can change. The hypothesis was explicit: **LoRA should postpone the overfitting turn and possibly reach a lower val floor.**

Tooling landmine handled first: `peft`'s current versions require a brand-new `transformers`, which would shatter our pinned `transformers==4.29.2` chain. We pinned **`peft==0.4.0`** and verified (functional import test, not pip's word) that `transformers` stayed 4.29.2 and `anticipation` still imported. Same Week-0 discipline.

LoRA mechanism: freeze the base weights, inject skinny low-rank adapter matrices (B·A) into the attention projections, train only those. With `r=8, alpha=16, target=["c_attn"]` → **294,912 trainable params (0.23%)** — traceable exactly from rank × architecture (12 layers × (768×8 + 8×2304)). A clarified misconception: LoRA constrains *dimensionality* (how many knobs), not *step size* (learning rate) — which is why we could *raise* the LR to `2e-4` (fresh adapters tolerate bigger steps).

The result vindicated the hypothesis:

```
epoch 0   val 0.320    (gap ~0.03)
epoch 3   val 0.242    (gap ~0.03)  ← already matching full-FT's best
epoch 8   val 0.228    (plateau)
epoch 10  val 0.2266   ← best
```

**LoRA won:** best val **~0.227** vs. full-FT's **0.243**, and — more importantly — the train–val gap held ~0.03–0.05 *throughout the descent* (vs. full-FT's blowout to 0.27). The constraint acted as the right regularizer for 46 pieces; the overfitting turn was postponed ~10 epochs. A clean, controlled result: same data, same seed, same loop, one variable changed. **LoRA is System A.** (A note logged for future runs: early stopping should use a `min_delta` — the plateau's tiny "new bests" kept resetting patience on noise.)

## 5. Generation — validated by ear, then debugged

Numbers said LoRA generalizes better; the week's ethic said *confirm by ear*. Same seed (`fluteSolo1`, a Week-0 held-out clip), same random seed, three-way listen — pretrained vs. full-FT vs. LoRA. **LoRA sounded clearly the most "like me"** — winding, conjunct motion vs. the base model's disjunct character.

But a chord appeared near the end of the LoRA clip, and full-FT had note-to-note "bleeding." Rather than discount it, we **looked at the raw note timings** — and found the real story: the model generates *pristine* monophonic melody for ~11–13 seconds, then **degenerates into a repetition collapse** (the same note emitted repeatedly, stacking with long overlapping durations into a "chord"). Crucially, this localized the problem to **sampling, not training**: the same weights that collapsed at 14s wrote clean melody at 4s, so the fix belongs in *how we decode*, not in the model.

The `anticipation` `generate` is a custom function; its signature (verified) exposes exactly one sampling knob: **`top_p`** (no `repetition_penalty` — that belongs to HuggingFace's different `generate`). A `top_p` sweep, inspected with the note-timing diagnostic:

- **`top_p=0.90`** → spotless (0 overlaps, 0 chords), ear-confirmed best.
- `top_p=0.95` → catastrophic (195 notes stacked at one instant).
- `top_p=0.98` → the original collapse.

The collapse is a **non-monotonic cliff**, not a smooth dial — which is exactly why we swept and inspected rather than reasoned to a value. **Generation recipe fixed: LoRA adapter, `top_p=0.90`, ~11s reliable horizon** (generate 10s to stay clear of drift).

A concept locked in here: too-*low* `top_p` causes a *different* repetition (a timid model always picking its top token on a held note), so the craft is the middle band — confident enough not to drift, adventurous enough to still sound like you. `top_p=0.90` threaded it.

## 6. The first objective metric — and the trap it took to trust it

We turned "sounds winding/like me" into numbers: the **melodic-interval histogram**. The reusable pattern under all of §9's metrics: reduce MIDI to a feature histogram, normalize to a distribution, compare generated vs. real. We chose **absolute** intervals (fold +2 and −2 together) — the honest justification being that it measures *step-vs-leap magnitude* and discards *direction* (not, as first reasoned, "starting note," which intervals already ignore for free). Reference corpus: the **462 original melodies** (not chunked, not transposed — fewest artifacts between the data and your real writing), 73,383 intervals.

The one-shot table (one generation per model) looked encouraging but was an anecdote. We **scaled up** — 20 generations per model, varying the *random seed* (`manual_seed(i)`, matched across the two models for fairness) from the single `fluteSolo1` prompt — to get trustworthy distributions.

Then the trap: the aggregated LoRA histogram claimed **58% repeated-notes (interval-0)** — contradicting both the ear and the individual clean files. We refused to trust the surprising number and hunted it down. It was *not* a metric bug in the usual sense: inspecting all 20 files revealed a **split population** — 16 healthy files (~40–110 notes, ~0% repeats) and **4 collapsed files** (gen_12,13,18,19: ~340 notes each, 74–87% repeats). The four runaways, being huge, contributed ~58% of all intervals and swamped the histogram.

The real finding underneath: **`top_p=0.90` doesn't eliminate the collapse — it makes it intermittent (~20% of generations).** Single-sample validation of `top_p` had the *exact* blind spot as single-sample listening: we'd drawn a good roll. We added a **collapse filter** (note-count > 200 or interval-0 fraction > 0.4 — collapsed runs are trivially detectable by their absurd note density), dropped the degenerate runs (kept 15/20 pretrained, 16/20 LoRA), and re-aggregated.

The **clean, trustworthy result** was more nuanced than "LoRA wins," and better science for it:

- LoRA's interval-0 dropped from 0.582 → **0.002** — the repeated-note mass was *entirely* the collapse artifact. Your ear was right.
- But filtered pretrained and LoRA interval distributions are **nearly identical**: both over-concentrate on whole-steps (~0.69 vs. the corpus's 0.369) and under-represent the corpus's variety (notably perfect-fourths, corpus 0.100 vs. models ~0.02–0.03) — *despite* the clear subjective preference for LoRA.

The lesson (which the plan's multi-metric design already anticipated): **a single interval histogram is too coarse to capture what the ear hears.** The audible "winding vs. disjunct" difference must live in dimensions this metric doesn't see — rhythm, contour (which `abs()` folded away), phrase structure. This is *why* §9 pairs multiple histograms + FMD + a listening test. We felt the reason empirically.

---

## Where things stand

**Week 2 is complete.** Corpus tokenized and packed (3,084/1,641 blocks); full fine-tune and LoRA both trained, with LoRA winning a clean head-to-head (val 0.227 vs 0.243) and chosen as System A; generation validated by ear, its `top_p=0.90` recipe set, and its intermittent-collapse failure mode characterized and filterable; the first objective metric (interval histogram) built, debugged, and yielding a real (nuanced) result plus a reusable tooling foundation for the remaining metrics.

**Recurring engineering lessons (Week 2):** each `python file.py` is a fresh universe (loaded models don't carry between runs); self-contained scripts and single-source-of-truth functions prevent the drift bugs that bit us twice (the `evaluate()` signature mismatch, the missing `is_collapsed`); measure surprising numbers before trusting conclusions built on them (the negative gap, the LR divergence, the phantom 0.582 all dissolved under inspection); anything about *generation* must be validated on a *batch*, because this model's failure mode is intermittent and single samples hide it.

## Next session (Week 4 evaluation core — with Week 3 retrieval alongside)

Build the remaining objective metrics on the tooling laid this week: **pitch-class** and **rhythm/duration** histograms (same shape, different feature — but use the *un-augmented* corpus, since transposition changes pitch-classes), a single **distance number** collapsing each histogram into "how far from the corpus" (seed of FMD), and the **transposition-aware novelty / anti-plagiarism check** — the research-integrity centerpiece, which reuses the corpus-aggregation machinery and the transposition-invariant interval signature. In parallel, Week 3's **retrieval (System B)** conditions generation on retrieved motifs via the `controls=` argument we spotted in the `generate` signature.

**Active generation recipe:** LoRA adapter (`checkpoints/best_lora`) on the base model, `top_p=0.90`, ~10s clips, collapse-filter any batch before use.
