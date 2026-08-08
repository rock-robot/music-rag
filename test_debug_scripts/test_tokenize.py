from pathlib import Path
from anticipation.convert import midi_to_events

CORPUS_DIR    = Path("data/train")           # your Week 1 output
TOKENIZED_DIR = Path("tokenized/train")      # NEW: parallel output folder
TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)   # NEW: create it if missing

files = sorted(CORPUS_DIR.glob("*.mid"))
print(f"found {len(files)} files")

subset = files[:20]          # STILL a dry run — do not point at 66k yet
written, failures = 0, []

for path in subset:
    try:
        events = midi_to_events(str(path))
        if len(events) == 0:
            failures.append((path.name, "empty token list"))
            continue
        out = TOKENIZED_DIR / (path.stem + ".txt")     # NEW: same stem, .txt
        out.write_text(" ".join(map(str, events)))     # NEW: write the tokens
        written += 1
    except Exception as e:
        failures.append((path.name, repr(e)))

print(f"inputs    : {len(subset)}")
print(f"written   : {written}")
print(f"failed    : {len(failures)}")
print(f"reconciles: {written + len(failures) == len(subset)}")   # the key check
for name, reason in failures:
    print(f"   !! {name}: {reason}")

