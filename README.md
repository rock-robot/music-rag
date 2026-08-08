# Personal-Style Symbolic Music Generation with Retrieval Augmentation

The purpose of this project was to fine-tune an Anticipatory Music Transformer on a personal \~46-piece catalogue to continue melodic phrases in my compositional style, then test whether retrieval augmentation (RAG) from my own corpus improves on a length-matched control. Overall it was a 4-week research project.

## Result

**Retrieval did not improve on the control.** Across three objective metrics (each against five corpus-reference constructions) and a blinded two-axis listening study (15 participants, 108 judgments), every estimate favoured the length-matched control A′ over the retrieval system B; no listening-test difference was significant (style *p*\=0.11, quality *p*\=0.064). Prepending corpus material *did* improve stylistic fidelity (A→A′) — context quantity mattered more than relevance. The measured mechanism: octave-only alignment leaves retrieved phrases in a foreign key 84% of the time.

See `docs/project_plan_5.md` for the full writeup and `docs/session_summary_week*.md` for the week-by-week narrative.

## The three systems

- **A** — LoRA-fine-tuned AMT, seed only.  
- **A′** — random train chunks prepended (length-matched control).  
- **B** — retrieved similar chunks prepended (the RAG system).

Headline is **A′ vs B** (A→A′ isolates the length confound).

## Setup

Specific version of python is necessary to run:

- Python **3.11** in a venv (`transformers==4.29.2` pins `tokenizers==0.13.3`, which has no 3.12 wheel).  
- `pip install -r requirements.txt`  \>\> generate this: `pip freeze > requirements.txt`  
- The `anticipation` package is installed from its GitHub repo, **not** PyPI.  
- GPU: \>\> RTX 4090 / CUDA cu124 / WSL2.

## Data

I've chosen not to include the raw catalogue as they are my personal compositions and the sheer size of the data is unreasonable to dump. However, you can use these tools to create your own stylized continuations given a large .MID library yourself.

## Pipeline (run in order)

**Week 1 — data pipeline** `routing.py` → `harvest.py`/`batch_harvest.py` → `chunk.py`/`batch_chunk.py` → `augment.py` → `split.py`  \>\> melody extraction → chunk → augment → split

**Week 2 — fine-tune** `tokenize_corpus.py` → `packer.py` → `train.py` / `train_lora.py`  \>\> tokenize → pack into 2048 token strings → train models

**Week 3 — retrieval** `features.py` → `build_index.py` → `retrieve.py` → `condition.py` → `seeds.py` → `generate_ab.py`  \>\> built 35d similarity vector → build index of clips for RAG to choose from → cosine similarity to find closest clips to seed clips (max one per piece) → align and augment data for proper RAG setup  → choose a set of 20 random seeds from val → generate clips from pretrained and RAG

**Week 4 — evaluation** `run_metrics.py` (histograms \+ 5-reference sensitivity), `novelty.py`/`novelty_stage2.py`, `check_key_drift.py`, `select_clips.py` → `render_audio.py` → `manifest_paired.py`, then `wilcoxon_ab.py`  \>\> histogram, pitch class, and novelty checks → select and render clips for trials → create manifest for external app → wilcoxon checks on research data

## Gotchas (read before re-running)

- **Tokenizer is married to the base model** — use `anticipation`'s `midi_to_events`, never MidiTok. A different tokenizer → the model reads noise.  
- **`generate` returns the whole stream** — always clip to `t > start_time`.  
- **Analyze per-seed, never pool raw** — pooled rates are dominated by their worst member (bit us four times).  
- **Re-render → rebuild manifest → redeploy, in that order** — durations are embedded at build time; a stale manifest makes playback bars race.  
- **Keys are private** — `PAIRING_KEY.json` maps listening-study slots to systems and is gitignored. The public data reveals nothing without it.

## Repo layout

Currently all scripts are flat in root (organizing into src/ is a planned follow-up) `listening-study/` is a separate repo.

## Not included

Model weights (`checkpoints/`), the audio corpus, `venv311/`, and the private keys are gitignored. Running `train_lora.py` after following the week one pipeline with your own data lets you generate your own model weights.