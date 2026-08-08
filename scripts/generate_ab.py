"""generate_ab.py — run the frozen seed set through Systems A, A', and B.

  A       seed only (the fine-tuned model alone)
  Aprime  random train chunks + seed (length-matched control)
  B       retrieved (similar) train chunks + seed (the RAG system)

All three run on the SAME seeds: B's budget exclusions are computed first and
applied to every system, so the comparison holds seed set constant.
"""

import json
import random
import pretty_midi
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

from anticipation.convert import midi_to_events, events_to_midi
from anticipation.sample import generate

from features import load_notes, clip_and_rezero, piece_name, load_generated_notes
from retrieve import load_index, retrieve
from seeds import load_seed_set, load_seed
from condition import (align_to_seed, truncate, notes_to_midi,
                       build_conditioned_prompt)

BASE      = "stanford-crfm/music-small-800k"
ADAPTER   = "checkpoints/best_lora"
TOP_P     = 0.90
GEN_SECS  = 10.0
K         = 2                 # locked in Week 3: k=2 keeps prompts inside the ~15s budget
N_SAMPLES = 5                 # intermittent collapse => never trust a single sample
OUT       = Path("out")


def load_model():
    model = AutoModelForCausalLM.from_pretrained(BASE)
    model = PeftModel.from_pretrained(model, ADAPTER)
    return model.cuda().eval()          # .eval() = dropout off (Week 2 measurement bug)


def classify(notes):
    """'empty' | 'collapsed' | 'ok'. Two distinct failure modes, not one."""
    if len(notes) < 2:
        return "empty"
    pitches = [n.pitch for n in notes]
    zeros = sum(1 for a, b in zip(pitches, pitches[1:]) if a == b)
    if len(notes) > 200 or zeros / (len(pitches) - 1) > 0.4:
        return "collapsed"
    return "ok"


def _align_all(chunk_names, seed_notes):
    """Load, truncate, align each retrieved/random chunk to the seed. Returns (phrases, prov)."""
    phrases, prov = [], []
    for name in chunk_names:
        r = load_notes(Path("data/train") / name)
        a, shift = align_to_seed(truncate(r), seed_notes)
        phrases.append(a)
        prov.append({"chunk": name, "piece": piece_name(name), "shift": shift})
    return phrases, prov


def build_prompt(name, seed_notes, system, B, paths, tmp):
    """Returns (events, start_time, prov) or None if over budget."""
    if system == "A":
        phrases, prov = [], []

    elif system == "Aprime":
        # random train chunks, deterministic per seed, re-drawn until under budget
        rng = random.Random(name)                     # seed the RNG from the seed's name
        for _ in range(50):
            picks = rng.sample(paths, K)
            phrases, prov = _align_all(picks, seed_notes)
            _, start_time, ok = build_conditioned_prompt(phrases, seed_notes)
            if ok:
                break
        for p in prov:
            p["sim"], p["random"] = None, True

    else:  # "B"
        hits = retrieve(seed_notes, B, paths, k=K, max_per_piece=1)
        phrases, prov = _align_all([n for n, _ in hits], seed_notes)
        for p, (_, sim) in zip(prov, hits):
            p["sim"] = round(sim, 4)

    notes, start_time, ok = build_conditioned_prompt(phrases, seed_notes)
    if not ok:
        return None

    notes_to_midi(notes, tmp)
    events = midi_to_events(str(tmp))
    return events, start_time + 0.05, prov            # +0.05 = quantization pad


def budget_exclusions(seeds, B, paths, tmp):
    """Which seeds can't be tested under B (prompt exceeds the adapter's range)?
    Computed on B and applied to ALL systems so the seed set stays constant."""
    excluded = set()
    for rec in seeds:
        seed_notes = load_seed(rec["name"])
        if build_prompt(rec["name"], seed_notes, "B", B, paths, tmp) is None:
            excluded.add(rec["name"])
    return excluded


def run(system, model, B, paths, exclude, n_samples=N_SAMPLES):
    seeds = load_seed_set()
    OUT.joinpath(system).mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"_prompt_{system}.mid"

    records = []
    outcomes = Counter()

    for rec in seeds:
        name = rec["name"]
        if name in exclude:
            continue
        seed_notes = load_seed(name)
        prompt = build_prompt(name, seed_notes, system, B, paths, tmp)
        if prompt is None:                            # shouldn't happen: exclude already applied
            continue
        events, start_time, prov = prompt

        for i in range(n_samples):
            torch.manual_seed(i)                      # MATCHED across A / A' / B
            with torch.no_grad():
                out = generate(model, start_time, start_time + GEN_SECS,
                               inputs=events, top_p=TOP_P)

            gen_mid = OUT / system / f"{Path(name).stem}_gen{i:02d}.mid"
            events_to_midi(out).save(str(gen_mid))    # mido: .save, not .write

            notes = clip_and_rezero(load_generated_notes(gen_mid), start_time)
            notes_to_midi(notes, gen_mid)             # overwrite with CLIPPED version

            verdict = classify(notes)
            outcomes[verdict] += 1
            raw = pretty_midi.PrettyMIDI(str(gen_mid))
            if len(raw.instruments) > 1:
                multi_instr += 1        # count it, report it alongside collapse rate
            records.append({"seed": name, "sample": i, "system": system,
                            "n_notes": len(notes), "verdict": verdict,
                            "seed_notes": rec["notes"], "retrieved": prov})

    total = sum(outcomes.values())
    print(f"System {system}: {total} generations on {total//n_samples} seeds  "
          f"| ok {outcomes['ok']}  collapsed {outcomes['collapsed']}  empty {outcomes['empty']}  "
          f"| collapse {100*outcomes['collapsed']/total:.1f}%")
    Path(f"results_{system}.json").write_text(json.dumps(records, indent=2))
    return records


if __name__ == "__main__":
    model = load_model()
    B, paths = load_index()

    tmp = OUT / "_prompt_budget.mid"
    OUT.mkdir(exist_ok=True)
    exclude = budget_exclusions(load_seed_set(), B, paths, tmp)
    print(f"budget-excluded {len(exclude)} seeds (applied to all systems): "
          f"{sorted(exclude) or 'none'}\n")

    for system in ("A", "Aprime", "B"):
        run(system, model, B, paths, exclude)
