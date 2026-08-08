"""
train.py — fine-tune music-small-800k on the melody corpus.
Save-on-best + early stopping to capture the best-generalizing model.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM

# --- config ---
MODEL_NAME    = "stanford-crfm/music-small-800k"
TRAIN_PATH    = "packed/train.txt"
VAL_PATH      = "packed/val.txt"
DEVICE        = "cuda"
BATCH_SIZE    = 4
LEARNING_RATE = 5e-5
MAX_EPOCHS    = 30
PATIENCE      = 3
CKPT_PATH     = "checkpoints/best_model"

class PackedBlocks(Dataset):
    def __init__(self, path, limit=None):
        self.blocks = []
        with open(path) as f:
            for line in f:
                self.blocks.append(list(map(int, line.split())))
                if limit and len(self.blocks) >= limit:
                    break
        print(f"loaded {len(self.blocks)} blocks from {path}")
    def __len__(self):  return len(self.blocks)
    def __getitem__(self, i):  return torch.tensor(self.blocks[i])

@torch.no_grad()
def evaluate(model, loader, max_batches=None):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        total += model(batch, labels=batch).loss.item()
        n += 1
        if max_batches and n >= max_batches:
            break
    return total / n

# --- setup ---
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

train_loader = DataLoader(PackedBlocks(TRAIN_PATH), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(PackedBlocks(VAL_PATH),   batch_size=BATCH_SIZE, shuffle=False)

best_val          = float("inf")
epochs_since_best = 0

# --- training ---
for epoch in range(MAX_EPOCHS):
    # --- train one epoch (running avg kept only as a live progress readout) ---
    model.train()
    running, n = 0.0, 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        loss = model(batch, labels=batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # NEW: seatbelt
        optimizer.step()
        optimizer.zero_grad()
        running += loss.item(); n += 1
        if n % 200 == 0:
            print(f"    epoch {epoch} batch {n:4d}   batch_loss {loss.item():.4f}")
    train_running = running / n          # noisy, dropout-on — for interest only

    # --- honest, comparable measurements: BOTH through evaluate() ---
    train_loss = evaluate(model, train_loader, max_batches=300)   # sample train
    val_loss   = evaluate(model, val_loader)                      # full val
    gap = val_loss - train_loss
    print(f"epoch {epoch:2d}   train {train_loss:.4f}   val {val_loss:.4f}   "
          f"gap {gap:+.4f}   (live {train_running:.4f})")

    # --- save-on-best + early stopping (unchanged) ---
    if val_loss < best_val:
        best_val = val_loss
        epochs_since_best = 0
        model.save_pretrained(CKPT_PATH)
        print(f"          new best val {best_val:.4f} — saved")
    else:
        epochs_since_best += 1
        print(f"          no improvement ({epochs_since_best}/{PATIENCE})")

    if epochs_since_best >= PATIENCE:
        print(f"stopping — val hasn't improved in {PATIENCE} epochs")
        break

print(f"done. best val loss: {best_val:.4f}   saved to {CKPT_PATH}")