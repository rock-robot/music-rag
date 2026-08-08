import torch
from transformers import AutoModelForCausalLM
from anticipation.sample import generate
from anticipation.convert import events_to_midi

# 1. Load the pretrained model and move it onto the GPU
model = AutoModelForCausalLM.from_pretrained('stanford-crfm/music-small-800k').cuda()

# 2. Generate 10 seconds of music from nothing (unconditional)
events = generate(model, start_time=0, end_time=10, top_p=0.98)

# 3. Turn the model's output back into a MIDI file we can play
mid = events_to_midi(events)
mid.save('first_generation.mid')
print("done — wrote first_generation.mid")