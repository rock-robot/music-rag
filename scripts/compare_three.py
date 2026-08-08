"""
compare_three.py — pretrained vs. full-fine-tune vs. LoRA, one seed, fair fight.
"""
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from anticipation.convert import midi_to_events, events_to_midi
from anticipation.ops import clip
from anticipation.sample import generate

DEVICE      = "cuda"
PRETRAINED  = "stanford-crfm/music-small-800k"
FULL_FT     = "checkpoints/best_model"      # full fine-tune (complete model)
LORA_ADAPT  = "checkpoints/best_lora"       # LoRA (adapter only)
SEED_MIDI   = "midi/fluteSolo1.mid"      # same seed as before
PROMPT_SECS = 5
GEN_SECS    = 15
RAND_SEED   = 42

# --- load all THREE models ---
model_pre  = AutoModelForCausalLM.from_pretrained(PRETRAINED).to(DEVICE)
model_full = AutoModelForCausalLM.from_pretrained(FULL_FT).to(DEVICE)

# LoRA = base + adapter, loaded in two steps (a fresh base, then the adapter on top)
lora_base  = AutoModelForCausalLM.from_pretrained(PRETRAINED)
model_lora = PeftModel.from_pretrained(lora_base, LORA_ADAPT).to(DEVICE)

# --- shared seed prompt (identical for all three) ---
events = midi_to_events(SEED_MIDI)
prompt = clip(events, 0, PROMPT_SECS)
events_to_midi(prompt).save("out_prompt.mid")

def continue_from(model, label):
    torch.manual_seed(RAND_SEED)                 # SAME dice for all three
    out = generate(model, start_time=PROMPT_SECS,
                   end_time=PROMPT_SECS + GEN_SECS,
                   inputs=prompt, top_p=0.98)
    events_to_midi(out).save(f"out_{label}.mid")
    print(f"  saved out_{label}.mid")


print("generating...")
"""
continue_from(model_pre,  "pretrained")
continue_from(model_full, "fullft")
continue_from(model_lora, "lora")
print("\nlisten in order: out_prompt / out_pretrained / out_fullft / out_lora")
"""

# same fluteSolo1 prompt, same RAND_SEED=42, ONLY top_p changes
for tp in [0.9, 0.95, 0.98]:
    torch.manual_seed(42)
    out = generate(model_lora, start_time=5, end_time=20, inputs=prompt, top_p=tp)
    events_to_midi(out).save(f"out_lora_tp{int(tp*100)}.mid")
    # reuse your overlap/chord check on each file