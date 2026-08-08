"""
smoke_test.py — prove the training loop works by overfitting a tiny slice.
Success = training loss on a handful of blocks crashes toward zero.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
MODEL_NAME    = "stanford-crfm/music-small-800k"
TRAIN_PATH    = "packed/train.txt"
DEVICE        = "cuda"
SMOKE_BLOCKS  = 8       # tiny slice we will memorize
SMOKE_BATCH   = 4
LEARNING_RATE = 5e-5
SMOKE_STEPS   = 50      # how many times we loop over the tiny slice

# ---------------------------------------------------------------------------
# dataset: same two-method contract as before, but only the first N blocks
# ---------------------------------------------------------------------------
class PackedBlocks(Dataset):
    def __init__(self, path, limit=None):
        self.blocks = []
        with open(path) as f:
            for line in f:
                self.blocks.append(list(map(int, line.split())))
                if limit is not None and len(self.blocks) >= limit:
                    break
        print(f"loaded {len(self.blocks)} blocks from {path}")

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        return torch.tensor(self.blocks[i])

# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

smoke_ds     = PackedBlocks(TRAIN_PATH, limit=SMOKE_BLOCKS)
smoke_loader = DataLoader(smoke_ds, batch_size=SMOKE_BATCH, shuffle=True)

# ---------------------------------------------------------------------------
# the training loop — THIS is what you're filling in
# ---------------------------------------------------------------------------
model.train()                          # training mode (NOT eval) — no_grad stays OFF

for step in range(SMOKE_STEPS):
    for batch in smoke_loader:
        batch = batch.to(DEVICE)

        # --- the four moves, in order. fill in blanks 1-4 ---
        out  = model(batch, labels=batch)   # 1. forward pass (given)
        loss = out.loss

        loss.backward()        # blank 1: compute gradients (backward pass)
        optimizer.step()   # blank 2: apply the nudges to the weights
        optimizer.zero_grad()   # blank 3: clear the gradients for next batch

    print(f"step {step:2d}   loss {loss.item():.4f}")