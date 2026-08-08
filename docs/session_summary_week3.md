# Session Summary — Week 3 (Retrieval + Conditioning → the Three-System Experiment)

*A narrative of how we got from "System A exists" to a complete, validated System B and a clean three-way A/A′/B experiment. Focuses on the reasoning behind each decision — and, as in Week 2, on the anomalies we chased down rather than waved through.*

---

## 1. Splitting the problem in two

The week opened by refusing to treat "RAG" as one thing. It's two genuinely different problems, and collapsing them is how beginners get lost:

- **Retrieval** — given a seed, which corpus phrases are "similar"? A *search* problem: feature vectors and cosine similarity, no neural net involved, testable on its own.
- **Conditioning** — how do we make the model actually *use* the retrieved phrases? A *sequence-construction* problem, and the one with a trap in it.

We built retrieval first, precisely because it's inspectable before a model is ever involved — you can look at what it retrieves and ask "is that a relative?" long before generation.

## 2. The conditioning trap: `controls=` vs. prepending

The plan had said System B would "likely" use AMT's `controls=` argument. We slowed that down and **read the `generate` source** rather than trusting the signature — the same discipline as reading constants from `anticipation.vocab`. The source settled it:

- `controls=` flips the model into **`ANTICIPATE` mode** (control code 55027), which our LoRA adapter was *never fine-tuned for* (it only ever saw `AUTOREGRESS`).
- Control events are offset by `CONTROL_OFFSET` into a separate vocab slice and interpreted as *future*, time-aligned material to write *around* — the accompaniment-to-a-melody mechanism.
- Anything before `DELTA` is **clipped**, so `controls=` literally cannot mean "here's earlier music."

Conclusion: `controls=` is the **cascade** mechanism (melody → harmony), not RAG. RAG is **prepending** — put the retrieved phrase in the *past*, before the seed, and let the model continue. And the source revealed how simple that is: events in `inputs` before `start_time` are the prompt; `start_time` is the switch. Build `[retrieved][seed]`, set `start_time` to the seed's end, done.

The most valuable line in the source was the return statement: **`generate` returns the whole stream** — prompt, seed, and continuation together. For System B that means a verbatim chunk of training corpus sits bolted to the front of every output. **Every generation must be clipped to `t > start_time` before any metric or listener touches it** — a paper-destroying bug pre-empted by reading twelve lines of someone else's code.

## 3. Building retrieval — the hand-built "embedding"

The core concept: to rank ~4,000 chunks against a seed, reduce each to a **fixed-length feature vector** and compare vectors. In text RAG a neural net makes that vector; in v1 *we* make it from music theory — interpretable, training-free, every axis nameable.

The vector became 35 dimensions: interval histogram (shape), duration histogram (rhythm), contour (direction). We **dropped pitch-class** — since we transpose the retrieved phrase to the seed's key anyway, matching on key is wasted signal. We **added contour** deliberately, because Week 2's `abs()` had folded direction away and made that metric too coarse; retrieval shouldn't repeat the mistake.

Two design principles fell out:
- **Normalize each block to sum 1 before concatenating**, so each musical dimension votes equally regardless of note count.
- **Cosine similarity** because it's scale-invariant — a 7-interval seed and a 60-interval chunk compare on *direction*, not magnitude.

A recurring realization: a short seed's vector is a **noisy, low-resolution sketch** (7 intervals → each bin swings 14 points). Not fixable by arithmetic — a reason to keep bins coarse and always eyeball the top-k.

## 4. The index, and a landmark-parsing lesson

The index had to be **un-augmented** (transpositions collapse to identical vectors under a transposition-invariant metric — the effective top-k would shrink to one) and **train-only** (a val seed must never retrieve its own 50%-overlapping continuation). Those two properties lived in different folders — until we noticed the `_t+00` identity-transposition files in `data/train/` are *exactly* "un-augmented train chunks." One glob, both properties, read off disk rather than recomputed.

Parsing piece names from filenames taught the landmark principle: **anchor on the reliable landmark, not the ends.** Piece names have unpredictable underscores (left fails) and varying suffixes (right fails), but every file has exactly one `_idx`. `str.partition("_idx")` (which reports whether it found anything, unlike `split`) became the single-source-of-truth `piece_name()`. A path-vs-filename contamination bug slipped through two passing assertions — a reminder that **a `len() == 39` test can never catch a defect that transforms every element identically.**

Validation was two-sided: the corpus self-similarity mean came out **0.52** (well-spread and discriminative — not everything-looks-alike), and the top-k **beat the volume prior** (a small piece won retrievals over `paradise_lost`, which is a third of the index). Retrieval measuring *shape*, not volume, confirmed by ear.

## 5. Diversity, and the argmax insight

The first retrieval returned **five chunks from one piece**. Not de-duplication failure — five different lines that piece simply dominated that region of feature space. The fix was a **`max_per_piece=1`** cap, which cost only ~0.03 similarity and *restored a real ranking gradient* (the uncapped top-5 spanned 0.004 — pure noise).

The deeper insight, which recurred all week: **retrieval selects (argmax); fine-tuning averages.** A weak or unrepresentative piece is 1/462nd of a gradient — washed out. But it can be *the entire* retrieved prompt for a specific seed. Averages are robust to outliers; argmaxes are maximally exposed. This is why `paradise_lost`'s 33% share and the `ohio_river_cruise` skeleton-parts piece matter far more now than they did during training.

## 6. Conditioning: three coupled problems

Splicing a retrieved phrase and a seed into one stream surfaced three interlocking problems:

1. **Key clash.** Transposition-invariant retrieval can return a phrase in a distant key. Solution: retrieve on shape, then **align on key** — the same coin as why augmentation works.
2. **Time collision.** Every chunk was re-zeroed to t=0, so naive concatenation stacks them into a chord. Solution: a **running clock** in `splice`.
3. **Context budget.** The prompt eats the 1024-token / ~15s window.

Problem 3 became the week's biggest discovery (§8). Problems 1 and 2 each had a subtle bug: alignment by full-semitone median-match produced **±25–33-semitone shifts** that jammed phrases against the pitch ceiling and correlated with collapse — fixed by switching to **octave-only** alignment (preserve the phrase's internal key, only move whole octaves). And truncation-by-onset let **sustained notes drag phrases past their nominal length** — the "fast-vs-slow bias" appearing a *fourth* time, finally fixed by a shared `window()` that clips onsets *and* sustains.

## 7. The prompt-budget discovery

The smoke test generated at **t=41–51 seconds**. The chunks had all been re-zeroed to a 0–15s timeline before tokenizing, so the **LoRA adapter never saw a time-token past ~15s.** Generating at t=41s pushes the model out of the adapter's distribution — it falls back toward the generic base model, which drifts and collapses. This wasn't a bug so much as a *constraint we hadn't seen*: **the retrieval budget is bounded by the adapter's training time-range.** RAG can only prepend as much material as fits in ~15s.

The fix: truncate retrieved phrases (`k=2`, `2.5s` each, `gap=0.25`) so `start_time` lands near ~12s, and **exclude-and-report** any prompt over 14s rather than generating it out-of-distribution (4 of 20 seeds excluded). Exclusions are data, not failure.

## 8. The length confound → System A′

Fixing the budget left System B's prompt at ~12s vs. System A's ~6s — so B differed from A in **two** ways: retrieval *and* length. The plan's promise ("the only difference is the retrieval layer") was broken.

The fix was a **third system, A′**: prepend **random** train chunks, matched in count and length to B, but with no *relevant* retrieval. Now the chain is clean — **A→A′ is the length effect; A′→B is retrieval's genuine contribution.** The headline comparison became A′-vs-B. Random-chunk (not neutral-filler) A′ was the stronger choice: it isolates *similarity* as the active ingredient, directly answering "does RAG just retrieve near-copies, or does the similarity do real work?"

This upgraded the experiment from "we tried RAG" to "we isolated retrieval's contribution with a matched-length control" — a better design than the plan originally specified.

## 9. The seed set — freezing the inputs

`seeds.py` froze 20 seeds, **round-robin across the 7 val pieces** rather than plain-random (which the known file-heavy val imbalance would have skewed toward one piece). A **viability filter** (≥6 notes in the 6s window) fired on the first run — one seed had 3 notes — and rather than lower the threshold (which would feed a garbage vector into retrieval), we regenerated with the filter *before* anything was computed. Note counts were stored per seed, because a real finding was already visible: seed density spans 8× (4 to 33 notes), and sparse seeds retrieve on noisy vectors — a **density stratifier for Week 4**, not just a caveat.

## 10. Wiring, and the bugs the batch surfaced

Assembling `generate_ab.py` produced the usual crop, each a familiar lesson in new clothes:
- `classify` returned a *string* but the runner summed it as a *bool* — tallied into a `Counter` for three-way `ok/collapsed/empty` reporting instead.
- Budget exclusions had to be **computed once on B and applied to all three systems**, or A ran 20 seeds and B ran 17 — comparing different populations.
- `events_to_midi` returns a **`mido.MidiFile`** (`.save()`), not a `pretty_midi` object (`.write()`). Three libraries now touch MIDI; the file is the interchange format, and every crossing quantizes — hence a round-trip check and a `+0.05s` pad on `start_time` so a tick of rounding can't flip a seed note into a control.
- `load_notes`'s single-instrument assertion **fired on a generation** — because `generate` can scatter notes across >1 instrument. The assert did its job (localized a real assumption break instead of silently dropping a track), and we added `load_generated_notes` that flattens and **skylines**, so simultaneous pooled notes can't create phantom unison intervals in the Week 4 metrics.

## 11. The collapse result — a mirage dissolved by per-seed analysis

The pooled collapse rates looked like a finding: A 6.2%, A′ 2.5%, B 8.8% — "retrieval collapses more." Then we computed **per-seed**, and the story inverted: **one seed (`el_mar_idx04`) produced 10 of 14 total collapses**, and it collapses even in *plain System A* (seed only). The cause is **seed-intrinsic**, not retrieval. Excluding it: **A 1.3% / A′ 1.3% / B 2.7%** — no meaningful difference at n=75.

The lesson — the *third* time this exact shape has bitten (Week 2's four-runaway-files-swamp-the-histogram; the phantom 0.582): **aggregate rates over a small, heterogeneous set are dominated by their worst member. Analyze per-seed, then aggregate — never pool raw.** The honest write-up reports the outlier as a failure case *and* the clean ex-outlier result: conditioning (random or retrieved) does not materially change generation stability. Which is fine — collapse was never the research question. Style is, and that's Week 4.

---

## Where things stand

**Week 3 is complete.** Retrieval built and validated (0.52 self-similarity, beats the volume prior, sounds right by ear); conditioning built with the mechanism *confirmed from source* (prepend, not `controls=`); octave-only alignment, onset+sustain truncation, and a prompt budget bounded by the adapter's training range; a three-system A/A′/B experiment with matched seeds, provenance logging, and a split collapse classifier; ~240 clean paired generations saved. Collapse characterized as condition-independent.

**New findings logged:** the `controls=`/prepend distinction; the prompt-budget constraint; the length confound → A′; the retrieval-corpus imbalance (argmax exposes what averaging absorbed); seed-density-dependent retrieval quality; multi-instrument generation → skyline pooling; and per-seed-not-pooled analysis discipline.

## Next session (Week 4 — evaluation + write)

Build the remaining objective metrics on Week 2's tooling: **pitch-class** and **rhythm/duration** histograms (un-augmented corpus reference), a **distance number / FMD**, and the **two-sided transposition-aware novelty check** (vs. the retrieved prompt *and* vs. the train set — System B is *expected* to score higher on train-overlap, so separate the two). Run everything **per-seed** on the `ok` generations of A / A′ / B, with **A′-vs-B as the headline**. Then the blind, multi-sample listening test, the density-stratified analysis, and the paper draft — negative and null results (including the condition-independent collapse) reported honestly.

**Active generation recipe:** LoRA adapter, `top_p=0.90`, ~10s clips, `k=2` retrieval, octave-only alignment, prompt budget ≤14s, collapse-filter and per-seed-aggregate every batch.
