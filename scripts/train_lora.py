"""
train_lora.py — LoRA fine-tune of music-small-800k on the melody corpus.

Identical to train.py EXCEPT for three changes (all marked `# LORA:` below):
  1. model is wrapped with a LoRA adapter (~0.23% of params trainable)
  2. higher learning rate (2e-4 vs 5e-6) — LoRA trains fresh adapters, tolerates bigger steps
  3. separate checkpoint path — so the full-fine-tune champion is not overwritten

Experiment: does constraining trainable params to ~295K (vs 128M) push the
overfitting turn later and/or reach a lower val loss than the full fine-tune's 0.24?
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType   # LORA: PEFT imports

# --- config ---
MODEL_NAME    = "stanford-crfm/music-small-800k"
TRAIN_PATH    = "packed/train.txt"
VAL_PATH      = "packed/val.txt"
DEVICE        = "cuda"
BATCH_SIZE    = 4
LEARNING_RATE = 2e-4                     # LORA: raised from 5e-6
MAX_EPOCHS    = 30
PATIENCE      = 3
CKPT_PATH     = "checkpoints/best_lora"  # LORA: separate from checkpoints/best_model

# LORA: adapter hyperparameters
LORA_R        = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["c_attn"]               # GPT-2 fused Q/K/V attention projection


class PackedBlocks(Dataset):
    def __init__(self, path, limit=None):
        self.blocks = []
        with open(path) as f:
            for line in f:
                self.blocks.append(list(map(int, line.split())))
                if limit and len(self.blocks) >= limit:
                    break
        print(f"loaded {len(self.blocks)} blocks from {path}")

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        return torch.tensor(self.blocks[i])


@torch.no_grad()
def evaluate(model, loader, max_batches=None):
    model.eval()                          # dropout OFF + single-shot: honest, comparable
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        total += model(batch, labels=batch).loss.item()
        n += 1
        if max_batches and n >= max_batches:
            break
    return total / n


# --- setup ---
# LORA: build the base model, then wrap it in a LoRA adapter
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGETS,
)
model = get_peft_model(base, lora_cfg).to(DEVICE)
model.print_trainable_parameters()        # LORA: confirm ~0.23% trainable every run

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

train_loader = DataLoader(PackedBlocks(TRAIN_PATH), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(PackedBlocks(VAL_PATH),   batch_size=BATCH_SIZE, shuffle=False)

best_val          = float("inf")
epochs_since_best = 0

# --- training ---
for epoch in range(MAX_EPOCHS):
    model.train()
    running, n = 0.0, 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        loss = model(batch, labels=batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # seatbelt vs. spikes
        optimizer.step()
        optimizer.zero_grad()
        running += loss.item(); n += 1
        if n % 200 == 0:
            print(f"    epoch {epoch} batch {n:4d}   batch_loss {loss.item():.4f}")
    train_running = running / n

    # honest, comparable measurements: BOTH through evaluate() (eval-mode, single-shot)
    train_loss = evaluate(model, train_loader, max_batches=300)
    val_loss   = evaluate(model, val_loader)
    gap = val_loss - train_loss
    print(f"epoch {epoch:2d}   train {train_loss:.4f}   val {val_loss:.4f}   "
          f"gap {gap:+.4f}   (live {train_running:.4f})")

    # save-on-best + early stopping
    if val_loss < best_val:
        best_val = val_loss
        epochs_since_best = 0
        model.save_pretrained(CKPT_PATH)          # LoRA: saves ONLY the adapter (a few MB)
        print(f"          new best val {best_val:.4f} — saved")
    else:
        epochs_since_best += 1
        print(f"          no improvement ({epochs_since_best}/{PATIENCE})")

    if epochs_since_best >= PATIENCE:
        print(f"stopping — val hasn't improved in {PATIENCE} epochs")
        break

print(f"done. best val loss: {best_val:.4f}   saved to {CKPT_PATH}")