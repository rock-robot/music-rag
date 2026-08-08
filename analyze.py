import json
from collections import defaultdict

def load(sys):
    return json.loads(open(f"results_{sys}.json").read())

rows = defaultdict(lambda: {"A":0, "Aprime":0, "B":0, "seed_notes":0})
for sysname in ("A", "Aprime", "B"):
    for r in load(sysname):
        rows[r["seed"]]["seed_notes"] = r["seed_notes"]
        if r["verdict"] == "collapsed":
            rows[r["seed"]][sysname] += 1

print(f"{'seed':45s} notes  A  A'  B")
for seed, d in sorted(rows.items(), key=lambda kv: -kv[1]["seed_notes"]):
    print(f"{seed[:44]:45s} {d['seed_notes']:>4}  "
          f"{d['A']}/5 {d['Aprime']}/5 {d['B']}/5")