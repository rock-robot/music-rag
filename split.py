import os
import glob

AUG_DIR = '/home/whamel/music-rag/augmented'

def piece_of(filename):
    """Extract the source piece from an augmented filename.
    'paradise_lost_mov_1_idx07_..._t-03.mid' -> 'paradise_lost_mov_1'"""
    base = os.path.basename(filename)
    return base.split('_idx')[0]        # everything before the '_idx' marker

import random

VAL_FRACTION = 0.15
SEED = 42               # fixed -> same split every run (reproducibility)

def choose_val_pieces(pieces):
    """Pick ~15% of pieces for validation, reproducibly."""
    pieces = sorted(pieces)          # sort FIRST, so input order can't affect the result
    n_val = round(len(pieces) * VAL_FRACTION)
    rng = random.Random(SEED)        # a seeded generator, isolated from global random
    return set(rng.sample(pieces, n_val))

import shutil

TRAIN_DIR = '/home/whamel/music-rag/data/train'
VAL_DIR   = '/home/whamel/music-rag/data/val'

def route_files(files, val_pieces):
    """Copy each file into train/ or val/ based on its source piece."""
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    counts = {'train': 0, 'val': 0}
    for f in files:
        dest_dir = VAL_DIR if piece_of(f) in val_pieces else TRAIN_DIR
        shutil.copy2(f, os.path.join(dest_dir, os.path.basename(f)))
        counts['val' if dest_dir == VAL_DIR else 'train'] += 1
    return counts

if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(AUG_DIR, '*.mid')))
    pieces = set(piece_of(f) for f in files)
    val_pieces = choose_val_pieces(pieces)

    print(f"routing {len(files)} files: {len(pieces)-len(val_pieces)} train pieces, "
          f"{len(val_pieces)} val pieces...")
    counts = route_files(files, val_pieces)
    print(f"\n  train: {counts['train']} files")
    print(f"  val:   {counts['val']} files")
    print(f"  total: {counts['train'] + counts['val']}")