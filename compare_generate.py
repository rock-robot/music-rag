"""
compare_generate.py — same seed, two models: pretrained vs. fine-tuned.
Any audible difference is attributable to fine-tuning alone.
"""
import torch
from transformers import AutoModelForCausalLM
from anticipation.convert import midi_to_events, events_to_midi
from anticipation.ops import clip

DEVICE       = "cuda"
PRETRAINED   = "stanford-crfm/music-small-800k"
FINETUNED    = "checkpoints/best_model"
SEED_MIDI    = "midi/fluteSolo1.mid"   # a val piece or a Week-0 seed clip
PROMPT_SECS  = 5                                # how much of the seed to feed as the prompt

# --- load BOTH models (same architecture, different weights) ---
model_pre = AutoModelForCausalLM.from_pretrained(PRETRAINED).to(DEVICE)
model_ft  = AutoModelForCausalLM.from_pretrained(FINETUNED).to(DEVICE)

# --- prepare the seed: tokenize, then clip to the first PROMPT_SECS seconds ---
events = midi_to_events(SEED_MIDI)
prompt = clip(events, 0, PROMPT_SECS)          # the shared starting phrase

# save the prompt itself so you can hear what you fed in
events_to_midi(prompt).save("out_prompt.mid")

print(f"seed piece   : {SEED_MIDI}")
print(f"prompt tokens: {len(prompt)}  ({PROMPT_SECS}s)")
print(f"both models loaded on {DEVICE}")

from anticipation.sample import generate

GEN_SECS  = 15      # generate this many seconds of continuation
RAND_SEED = 42      # the RANDOM seed — pinned identically for both models
PROMPT_END = PROMPT_SECS   # continue from where the prompt ends

def continue_from(model, prompt, label):
    torch.manual_seed(RAND_SEED)            # SAME dice for both models
    events = generate(
        model,
        start_time=PROMPT_END,              # start generating where the prompt ends
        end_time=PROMPT_END + GEN_SECS,     # generate GEN_SECS more
        inputs=prompt,                      # the shared seed phrase
        top_p=0.98,
    )
    events_to_midi(events).save(f"out_{label}.mid")
    print(f"  saved out_{label}.mid  ({len(events)} tokens)")
    return events

print("generating pretrained continuation...")
gen_pre = continue_from(model_pre, prompt, "pretrained")

print("generating fine-tuned continuation...")
gen_ft  = continue_from(model_ft,  prompt, "finetuned")

print("\ndone — listen to:")
print("  out_prompt.mid      (the shared 5s starting phrase)")
print("  out_pretrained.mid  (generic model's continuation)")
print("  out_finetuned.mid   (YOUR model's continuation)")