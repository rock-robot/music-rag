import torch
from transformers import AutoModelForCausalLM
from anticipation import ops
from anticipation.sample import generate
from anticipation.convert import midi_to_events, events_to_midi

model = AutoModelForCausalLM.from_pretrained('stanford-crfm/music-small-800k').cuda()

# 1. Load your seed MIDI and turn it into the model's event tokens
SEED = '/home/whamel/music-rag/midi/fluteSolo1.mid'      # <-- change this
events = midi_to_events(SEED)

# 2. Keep only the first 5 seconds of the seed as the prompt ("history")
history = ops.clip(events, 0, 5, clip_duration=False)

# 3. Continue from second 5 out to second 15 (10 seconds of new music)
continuation = generate(model, 5, 15, inputs=history, top_p=0.98)

# 4. Back to MIDI, save, listen
mid = events_to_midi(continuation)
mid.save('continuation.mid')
print("done — wrote continuation.mid")