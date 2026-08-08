import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM

DEVICE = "cuda"
CKPT   = "checkpoints/best_model"     # the epoch-0 model

class PackedBlocks(Dataset):
    def __init__(self, path):
        self.blocks = [list(map(int, l.split())) for l in open(path)]
    def __len__(self):       return len(self.blocks)
    def __getitem__(self, i): return torch.tensor(self.blocks[i])

model = AutoModelForCausalLM.from_pretrained(CKPT).to(DEVICE)
loader = DataLoader(PackedBlocks("packed/val.txt"), batch_size=4, shuffle=False)
batch  = next(iter(loader)).to(DEVICE)

# SAME model, SAME batch, two modes — the only difference is train vs eval
model.train()
with torch.no_grad():
    loss_train_mode = model(batch, labels=batch).loss.item()

model.eval()
with torch.no_grad():
    loss_eval_mode = model(batch, labels=batch).loss.item()

print(f"train() mode loss : {loss_train_mode:.4f}")
print(f"eval()  mode loss : {loss_eval_mode:.4f}")
print(f"difference        : {abs(loss_train_mode - loss_eval_mode):.4f}")