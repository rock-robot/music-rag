# Session Summary — Week 4 (Objective Metrics → Listening Study → the Result)

*A narrative of how we got from "System B exists" to a complete objective battery, a deployed blinded listening study, and a defensible answer to the research question. As in earlier weeks, the focus is on the reasoning behind each decision — and on the anomalies we chased down rather than waved through, because Week 4 had several that would each have produced a wrong or fragile result if trusted.*

---

## 1. The metrics are one idea wearing four hats

Week 4 opened by refusing to treat the four objective metrics as four separate things. They are the same move — the one already made in Week 2 with the interval histogram: **reduce each melody to a feature → turn it into a distribution → compare against a reference.** Only two knobs change between metrics: *which feature*, and *which reference*. A third, quieter axis decides everything downstream: **is the feature transposition-invariant?** That axis is what made pitch-class the tricky one.

The organizing table we kept returning to: interval and rhythm are transposition-invariant (a rising whole-step is a rising whole-step in every key; transposition doesn't touch time), so their reference can be anything. **Pitch-class is the lone exception** — transposition *rotates* it. Aggregate the augmented corpus for pitch-class and each melody's tonal content smears across all 12 rotations, flattening the histogram toward uniform and erasing the very signal you're measuring. So the pitch-class reference *must* be the un-augmented 462 originals. That is the exact mirror of why augmentation helps training: it destroys pitch-class information, which is wonderful for teaching key-invariance and fatal for measuring key.

## 2. Reading the real code changed the plan twice

Two corrections came straight from reading files rather than trusting memory. First, Week 2's `interval_counts` read `instruments[0]` directly — fine then, but Week 3 had discovered `generate` scatters notes across multiple instruments. Running it unchanged on the A/A′/B generations would have *silently dropped* every note on instrument 1+, with no assert to catch it. The fix was structural: **metrics take notes, not paths.** Loading (and skylining) is `features.py`'s job; measuring is the metric's.

Second, `generate_ab.py` had already `clip_and_rezero`'d every generation before saving — so the files on disk were continuation-only, re-zeroed to t=0. That killed a whole class of prompt-stripping hazards I'd worried about: the artifact on disk was already correct, so there was nothing to correct for. It also turned `load_notes`'s single-instrument assert into a free integrity check that the clip step had run.

A third, uglier truth surfaced here: the runner's multi-instrument diagnostic could **never have fired** — it read the file *after* overwriting it with the single-instrument clipped version, and `multi_instr` was never even initialized. The Week 3 run completing was proof the branch never ran, which told us nothing about the data. Same shape as every in-place-overwrite bug: *a diagnostic must read its input before anything downstream mutates it.* The multi-instrument rate is simply unmeasured, and (since the skyline neutralizes the risk it monitored) that's a logged footnote, not a threat.

## 3. Rhythm — where binning stops being given

Pitch-class hands you 12 bins by nature. Duration hands you nothing: it's continuous, so the edges are a design decision, and the wrong ones can manufacture or hide a difference. Inspecting the corpus first (the honest move — choose bins from the corpus, which isn't a party to the A′-vs-B comparison, then freeze them) showed a *smear*, not clean quantized peaks — tempo variation across 46 pieces spreading each written value across many seconds. So log-spaced bins, and the tempo conflation logged as a stated limitation.

The data also overturned a default I'd offered: IOI is *usually* the truer rhythmic feature, but these are extracted orchestral parts. IOI's 99th percentile was 23 seconds and its max 320 — a flute sitting tacet for five minutes is not a rhythmic relationship. Duration it was, with the edges frozen as a strict **refinement** of the retrieval bins (each coarse edge preserved, so the 7-bin retrieval histogram is exactly recoverable by summing) — the only way two binning schemes can coexist without ever telling contradictory stories.

A tangent worth its time: the hypothesis that trills explained the 25%-of-corpus short-note spike was **falsified by measurement**. Trills are 3% of notes and 11% of short notes; 80% of short notes are fast *non-ornamental* figuration — real melodic motion, the flute flourishes that make the style. Reducing them would have deleted the most characteristic thing in the data. We measured before cutting, and kept everything.

## 4. The two-mode collapse filter, and the circularity it exposed

The generation collapse we'd characterized in Week 2 has *two* degeneracy modes, and no single test catches both. Static repetition (a note hammered over and over) shows as high interval-0 fraction. A two-pitch oscillation (a-b-a-b for ten seconds) has *zero* repeats and sails through interval-0, but is just as degenerate — caught only by distinct-pitch count. The parking-lot idea (`distinct < 3` as a *replacement*) turned out badly under-inclusive: it caught **nothing** among 96%-repeat runs spread over 4–6 pitches. The answer was a **union** — either mode fires — verified by running the interval metric under three different filter criteria and showing the headline held.

That surfaced a real circularity: the filter conditions on interval-0, and the interval histogram *reports* interval-0. Conditioning on the statistic you report truncates it — and unequally, since B lost more generations than A′. The fix was not a cleverer filter but a cleaner metric: **compute the interval distance over the non-zero bins, renormalized** (`tv_nonzero`), and report interval-0 separately with its caveat. Filter freely; measure something the filter can't touch.

## 5. The reference is not one thing — and that nearly swamped everything

The single most important methodological fact of the week: **"distance from corpus" depends enormously on how you construct "the corpus."** We built the reference five ways (whole vs. 10s-excerpt × pooled vs. per-unit-averaged, plus a training-distribution `chunk_avg`) and the constructions disagreed by up to **0.28 TV** — roughly *ten times* the A′-vs-B effect we were trying to measure. A distance built on an unexamined reference would have been meaningless, and FMD stacked on top would have inherited all of it.

Two things saved it. An **identity check** proved the machinery: exhaustive 10s tiling of a melody can't change a *pooled* histogram (pooling is just addition; every note still counts once), and the assert held — so length-under-pooling was a non-issue, killing my first "clip-length artifact" story. And running the A′-vs-B delta under *all five* references showed the **sign held in every one** (and in every metric): a headline robust to a choice that moves absolute numbers by 0.28 is a far stronger claim than a single number. The bonus finding: reference choice moves only *rhythm* metrics — pitch-class and interval are near-invariant, because the constructions differ along a density axis that correlates with note length but not tonal content.

The `>2.50s` duration bin resolved cleanly here too: three independent references converge on ~0.063 for the corpus while all systems sit at ~0.006 — **the model genuinely under-produces long notes 8×**, a real stylistic limitation, not an artifact. Paired with a measured 3× density gap (generations ~30 notes/10s vs. corpus ~7), the two are one fact: shorter notes, more of them.

## 6. Novelty — the research-integrity centerpiece, calibrated

A raw overlap number is uninterpretable — tonal melody shares short patterns by chance. So we built it in two stages, calibration first. The **ceiling** is genuine novelty: the composer's own held-out val pieces vs. the train set (18.8% of 8-grams already recur across the composer's pieces — your vocabulary). The **floor** is chance: each melody's own intervals shuffled, order destroyed. The gap between them is where order carries information, and we froze **n=5** (max separation) and **n=8** (a match there is 7.5× chance — near-damning).

The result was clean on both sides. Generations overlap the train set at ~the val ceiling → **no memorization**. B's overlap with its *own retrieved prompt* ≈ A′'s with its random prompt → **no retrieval regurgitation** — the "RAG just copies" fear, answered with a control rather than an assurance. And ordered intervals match ~25× more than shuffled at n=8, proving the model writes real melodic *structure*, not vocabulary soup. (One design fix: the provenance table had to count *rare* n-grams — unique to one piece — because common patterns diluted every piece to a flat ~5%.)

## 7. The mechanism: why retrieval didn't help

The most useful diagnostic of the week explained the objective result before the listening test confirmed it. Week 3's octave-only alignment (chosen to avoid ±25–33-semitone register jams) preserves a phrase's internal key. Measured across every B generation: the retrieved phrase sits a **mean 3.0 semitones** off the generation's key, and only **15.8%** are actually in-key. So System B's prompts are **bitonal by construction** — and B's pitch-class distance from the corpus got *worse*, exactly as a foreign-key prompt would predict, and with the *lowest variance* of the three systems (consistently smeared rather than sometimes-right). That gives the paper a result, a control that isolates it, and a mechanism — a complete story. The v2 fix writes itself: align on pitch-class first, correct register by octave.

## 8. From generations to a listening study

Turning ~240 clean generations into something people could rate needed pieces that never existed. **Clip selection** picks one sample per seed per system — the lowest index surviving the union filter in *all three*, so the matched `torch.manual_seed(i)` pairing from Week 3 is preserved (picking different indices per system would have thrown away the pairing and put sampling noise back into the headline). All 15 usable seeds resolved to sample 0; `el_mar_idx04` dropped out entirely (all 5 B samples collapse), giving one consistent exclusion across both analyses.

**Rendering** taught its own lessons through failures. The seed and continuation were reassembled with a 2s pause (the prompt itself never played — System A has none, so playing it would reveal the condition). Instrument was assigned **per seed, not per system** — a timbre split inside a pair would be judged instead of the melody. Reverb and chorus off, for a crisp handoff and one fewer variable. Levels normalized to one target, so loudness couldn't cue the choice. And a cascade of small bugs, each a familiar shape: **horn** screamed on high melodies (its GM patch has no samples up there) and was dropped; **calibration clips came out as three identical 33 KB files** — the window took the first 10s of the *timeline*, and extracted parts rest at the start, so all three were empty, silence that `loudnorm` then amplified into a glitch. Three identical file sizes were the only signal; a duration assert now turns that into a crash.

## 9. Blinding, moved into the filenames

The Lovable rebuild changed the blinding model. With pairing encoded in filenames, the names could carry *pair* and *slot* (1/2) but never *system* — and since a fixed "slot 1 = A′" would itself be the leak, slot assignment is coin-flipped per seed, with the slot→system mapping kept only in a private `PAIRING_KEY.json`. The database (Supabase) never knew which slot was which condition; only the join against the private key, run afterward in `music-rag`, turns a "2" into "B." A Supabase RLS wall (401 on insert) was the expected anonymous-study snag — insert/update policies for `anon`, and deliberately *no* select policy, so participants can write but the public can't read the results table.

## 10. The result, and reading it honestly

Fifteen participants, 108 judgments, all 15 seeds covered (4–10 each) — the rotating offset spread coverage evenly across a mix of completers and short Quick runs. The join and per-seed rates gave: **style B-preference 0.396, quality 0.353** — both *below* 0.5, meaning listeners leaned toward **A′**, the length-matched control. Wilcoxon: **style p = 0.11, quality p = 0.064.** Neither significant; at 15 seeds the study is underpowered to certify an effect this small.

The correct reading is not "retrieval hurts" but **"no method found retrieval to help, and everything leaned the same way."** Every independent instrument — three objective histograms across five references, plus two subjective axes — pointed at A′ over B. Convergence across independent methods on a null-to-slightly-negative effect is a *stronger* result than any single test, and the key-drift measurement explains *why*. The one leg with no subjective test — A→A′, the finding that prepending corpus material *improves* fidelity — stays robust and objective-only, because the A-vs-A′ pairs were never built into the Lovable app (`COMPARISON=("Aprime","B")`); the objective result already stands, so it wasn't worth a re-render.

The methodological contribution to protect in the write-up: **A′ is why there is a result at all.** A-vs-B alone would have read as a flat null. Splitting it revealed two opposite-signed effects — context helps, relevance slightly hurts — that nearly cancel. That is the paper.

---

## Where things stand

**Week 4 is complete.** Four objective metrics built, debugged, and made robust (per-seed aggregation, five-reference reporting, two-mode filtering, transposition-aware novelty with calibrated ceiling and floor). A deployed, blinded, level-matched listening study with 15 participants. A clear, defensible, honestly-null-to-negative answer to the research question, with a measured mechanism. The remaining artifact is the paper draft.

**Recurring engineering lessons (Week 4):** measure before you cut (trills, the 0.28 reference spread, the empty calibration clips all dissolved or resolved under inspection); analyze per-seed, then aggregate — never pool raw (the fourth appearance, now enforced in function signatures); a diagnostic must read its input before anything mutates it (the dead multi-instrument probe); freeze design parameters against data that isn't part of the comparison (bin edges from the corpus, before A′/B); and the crash that surfaces is worth more than the silent success that doesn't (the semicolon CSV, the RLS 401, the schema KeyError each pointed straight at their cause).

## Next

Fold everything into the paper (§12 outline): the A→A′ context effect, the A′→B non-effect across metrics and listeners, the two-sided novelty, the condition-independent collapse, and the key-drift mechanism — negative and null results reported honestly, because they are the finding. Then the code goes to GitHub.
