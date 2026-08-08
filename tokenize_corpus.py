from pathlib import Path
from anticipation.convert import midi_to_events

def tokenize_corpus(corpus_dir, tokenized_dir, expected=None, report_every=5000):
    corpus_dir, tokenized_dir = Path(corpus_dir), Path(tokenized_dir)
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(corpus_dir.glob("*.mid"))
    print(f"[{corpus_dir.name}] found {len(files)} files")
    if expected is not None and len(files) != expected:      # count guard
        print(f"   !! expected {expected} — STOP and investigate")
        return

    written, failures = 0, []
    for i, path in enumerate(files, 1):
        try:
            events = midi_to_events(str(path))
            if len(events) == 0:
                failures.append((path.name, "empty token list"))
                continue
            (tokenized_dir / (path.stem + ".txt")).write_text(" ".join(map(str, events)))
            written += 1
        except Exception as e:
            failures.append((path.name, repr(e)))
        if i % report_every == 0:                            # heartbeat
            print(f"   ...{i}/{len(files)}  (ok={written}, failed={len(failures)})")

    ok = written + len(failures) == len(files)
    print(f"[{corpus_dir.name}] done: written={written}, failed={len(failures)}, reconciles={ok}")
    if failures:                                             # persist the reasons
        log = tokenized_dir.parent / f"failures_{corpus_dir.name}.txt"
        log.write_text("\n".join(f"{n}\t{r}" for n, r in failures))
        print(f"   failures logged to {log}")
    return written, failures

# one definition, two calls — no drift possible
tokenize_corpus("data/train", "tokenized/train", expected=42967)
tokenize_corpus("data/val",   "tokenized/val",   expected=23155)