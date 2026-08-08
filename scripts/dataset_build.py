import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

class PackedBlocks(Dataset):
    def __init__(self, path):
        # read the whole file ONCE (option a), parse each line into a list of ints
        self.blocks = []
        with open(path) as f:
            for line in f:
                self.blocks.append(list(map(int, line.split())))
        print(f"loaded {len(self.blocks)} blocks from {path}")

    def __len__(self):
        # QUESTION 1: how many examples do we have?
        return len(self.blocks)

    def __getitem__(self, i):
        # QUESTION 2: return block i as a tensor of token IDs
        return torch.tensor(self.blocks[i])

BATCH_SIZE = 4      # start small — confirm it fits, push up later

train_ds = PackedBlocks("packed/train.txt")
val_ds   = PackedBlocks("packed/val.txt")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)  
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)  

batch = next(iter(train_loader))     # pull the first batch
print("batch shape:", batch.shape)
print("batch dtype:", batch.dtype)

model = AutoModelForCausalLM.from_pretrained("stanford-crfm/music-small-800k")
device = "cuda"                       # your 4090
model = model.to(device)              # move the model's weights to GPU (once)

def evaluate(model, loader):
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)  # move THIS batch to GPU (every iteration)
            out = model(batch, labels=batch)
            total_loss += out.loss.item()
            n_batches += 1
    return total_loss / n_batches


train_loss = evaluate(model, train_loader)
print(f"baseline train loss: {train_loss:.4f}")