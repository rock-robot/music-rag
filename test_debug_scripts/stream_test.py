from pathlib import Path

TOKENIZED_DIR = Path("tokenized/train")

files = sorted(TOKENIZED_DIR.glob("*.txt"))
print(f"reading {len(files)} token files")

stream = []                                   # the growing concatenated list
for i, path in enumerate(files, 1):
    tokens = list(map(int, path.read_text().split()))
    stream.extend(tokens)
    if i % 5000 == 0:
        print(f"   ...{i}/{len(files)}  (so far: {len(stream):,} tokens)")

print(f"total tokens : {len(stream):,}")
print(f"avg / chunk  : {len(stream) / len(files):.1f}")