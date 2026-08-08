import torch
from pathlib import Path
from anticipation.convert import midi_to_events, events_to_midi
from anticipation.ops import clip
from anticipation.sample import generate
from transformers import AutoModelForCausalLM
from peft import PeftModel

DEVICE      = "cuda"
PRETRAINED  = "stanford-crfm/music-small-800k"
FULL_FT     = "checkpoints/best_model"      # full fine-tune (complete model)
LORA_ADAPT  = "checkpoints/best_lora"       # LoRA (adapter only)
SEED_MIDI   = "midi/fluteSolo1.mid"      # same seed as before
PROMPT_SECS = 5
GEN_SECS    = 10          # stay inside the ~11s reliable horizon!
TOP_P       = 0.90        # the value we validated

model_pre  = AutoModelForCausalLM.from_pretrained(PRETRAINED).to(DEVICE)

# LoRA = base + adapter, loaded in two steps (a fresh base, then the adapter on top)
lora_base  = AutoModelForCausalLM.from_pretrained(PRETRAINED)
model_lora = PeftModel.from_pretrained(lora_base, LORA_ADAPT).to(DEVICE)

# --- shared seed prompt (identical for all three) ---
events = midi_to_events(SEED_MIDI)
prompt = clip(events, 0, PROMPT_SECS)
events_to_midi(prompt).save("out_prompt.mid")

def generate_batch(model, prompt, tag, n=20):
    outdir = Path(f"gen_{tag}")
    outdir.mkdir(exist_ok=True)
    for i in range(n):
        torch.manual_seed(i)              # DIFFERENT dice each run, SAME across models
        out = generate(model, start_time=PROMPT_SECS,
                       end_time=PROMPT_SECS + GEN_SECS,
                       inputs=prompt, top_p=TOP_P)
        events_to_midi(out).save(outdir / f"gen_{i:02d}.mid")
    print(f"  wrote {n} generations to {outdir}/")




prompt = clip(midi_to_events("midi/fluteSolo1.mid"), 0, PROMPT_SECS)
generate_batch(model_pre,  prompt, "pretrained")
generate_batch(model_lora, prompt, "lora")