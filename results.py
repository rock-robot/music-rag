# results.py -- load the Week 3 experiment records into a measurable form.
import json
from pathlib import Path
from collections import Counter

OUT = Path("out")
SYSTEMS = ("A", "Aprime", "B")


def gen_path(rec):
    """Rebuild the on-disk path from a record. Mirrors generate_ab.py's naming:
        out/{system}/{seed-stem}_gen{sample:02d}.mid"""
    return OUT / rec["system"] / f"{Path(rec['seed']).stem}_gen{rec['sample']:02d}.mid"


def load_results(system, ok_only=True):
    """Records for one system, with `path` attached. Verdicts are read from the
    JSON, not recomputed -- the classifier ran once, at generation time."""
    recs = json.loads(Path(f"results_{system}.json").read_text())
    for r in recs:
        r["path"] = gen_path(r)
    missing = [r for r in recs if not r["path"].exists()]
    if missing:
        raise FileNotFoundError(f"{system}: {len(missing)} records have no file, "
                                f"e.g. {missing[0]['path']}")
    verdicts = Counter(r["verdict"] for r in recs)
    kept = [r for r in recs if r["verdict"] == "ok"] if ok_only else recs
    print(f"{system}: {len(recs)} generations on "
          f"{len({r['seed'] for r in recs})} seeds | "
          f"{dict(verdicts)} | using {len(kept)}")
    return kept


def load_all(ok_only=True):
    return {s: load_results(s, ok_only) for s in SYSTEMS}
