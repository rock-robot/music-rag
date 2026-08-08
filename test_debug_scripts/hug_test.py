import torch
from transformers import AutoModelForCausalLM

MODEL = "stanford-crfm/music-small-800k"
model = AutoModelForCausalLM.from_pretrained(MODEL)   # first run downloads ~0.5 GB, then caches

# 1. how big is it, really?
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters     : {n_params:,}")

# 2-3. does the model agree with what we packed against?
print(f"vocab size     : {model.config.vocab_size}")
print(f"context length : {model.config.n_positions}")

# 4. the embedding table — the thing we reasoned about two sessions ago
print(f"embedding shape: {tuple(model.get_input_embeddings().weight.shape)}")