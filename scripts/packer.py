from pathlib import Path

# --- constants, straight from your vocab dump ---
AR      = 55026        # AUTOREGRESS control code
SEP     = 55025        # SEPARATOR
BLOCK   = 1024         # CONTEXT_SIZE
CONTENT = BLOCK - 1    # room left after the AR token = 1023

def build_ribbon(tokenized_dir):
    """Read every token file into one SEP-punctuated stream."""
    files = sorted(Path(tokenized_dir).glob("*.txt"))
    ribbon = []
    for path in files:
        ribbon.extend(list(map(int, path.read_text().split())))
        ribbon.extend([SEP, SEP, SEP])          # boundary wall
    return ribbon, len(files)

def pack(ribbon):
    """Slice the ribbon into [AR] + 1023 blocks, dropping the tail."""
    blocks = []
    for start in range(0, len(ribbon) - CONTENT + 1, CONTENT):
        slice_ = ribbon[start : start + CONTENT]
        blocks.append([AR] + slice_)
    return blocks

def pack_split(tokenized_dir, out_path, expected_files=None):
    ribbon, n_files = build_ribbon(tokenized_dir)
    if expected_files is not None and n_files != expected_files:
        print(f"   !! expected {expected_files} files, found {n_files} — STOP")
        return
    blocks = pack(ribbon)

    # --- sanity checks before writing (measure, don't trust) ---
    lengths = {len(b) for b in blocks}
    print(f"[{Path(tokenized_dir).name}] files={n_files}  "
          f"ribbon={len(ribbon):,}  blocks={len(blocks):,}  "
          f"block_lengths={lengths}")
    assert lengths == {BLOCK}, f"malformed blocks: {lengths}"

    # --- write: one block per line, space-separated ints ---
    with open(out_path, "w") as f:
        for b in blocks:
            f.write(" ".join(map(str, b)) + "\n")
    print(f"   wrote {len(blocks):,} blocks -> {out_path}")

pack_split("tokenized/train", "packed/train.txt", expected_files=42967)
pack_split("tokenized/val",   "packed/val.txt",   expected_files=23155)