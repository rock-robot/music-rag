import torch
from transformers import AutoModelForCausalLM

# --- load the model IN THIS script (it can't carry over from another run) ---
model = AutoModelForCausalLM.from_pretrained("stanford-crfm/music-small-800k")
model.eval()

# --- now feed one packed block ---
with open("packed/train.txt") as f:
    block = list(map(int, f.readline().split()))

input_ids = torch.tensor([block])
print("input shape:", input_ids.shape)

with torch.no_grad():
    out = model(input_ids, labels=input_ids)

print("loss:", out.loss.item())