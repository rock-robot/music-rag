import csv, json
from collections import defaultdict

with open("responses.csv", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

# (pair, slot) -> system, from the PRIVATE key
key = {(k["pair"], int(k["slot"])): k["system"]
       for k in json.load(open("PAIRING_KEY.json"))}

def rate(field):
    """Per-seed fraction choosing B on `field`. Per-seed FIRST, never pooled."""
    seed = defaultdict(lambda: {"B": 0, "n": 0})
    for r in rows:
        choice = r[field].strip()
        if choice not in ("1", "2"):
            continue                        # unanswered / malformed
        system = key[(r["pair_id"], int(choice))]
        seed[r["pair_id"]]["n"] += 1
        seed[r["pair_id"]]["B"] += (system == "B")
    return {s: d["B"] / d["n"] for s, d in seed.items() if d["n"]}

style   = rate("style_choice")
quality = rate("quality_choice")

print("seed    style_B%  quality_B%")
for s in sorted(style):
    print(f"  {s}   {style[s]:5.2f}      {quality[s]:5.2f}")

import statistics as st
for name, d in (("STYLE", style), ("QUALITY", quality)):
    vals = list(d.values())
    above = sum(v > 0.5 for v in vals)
    print(f"\n{name}: mean per-seed B-preference {st.mean(vals):.3f} "
          f"({above}/{len(vals)} seeds favour B)")