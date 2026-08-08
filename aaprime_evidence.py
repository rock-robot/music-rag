"""aaprime_evidence.py — does the A->A' context effect show up subjectively?

A->A' is the strongest OBJECTIVE finding (prepending corpus material improved
stylistic fidelity: negative delta on all 3 metrics, 5 references, 10/16 seeds)
but it had NO subjective confirmation. Full-tier participants also rated A-vs-A'
pairs; this pulls those trials out and asks whether listeners preferred A'.

These trials live in the SAME responses.csv, distinguished only by which two
systems their pair maps to in PAIRING_KEY.json. So the entire separation is the
key join -- the database never knew the comparison type.

Run from music-rag.
"""
import csv, json, statistics as st
from collections import defaultdict, Counter
from scipy.stats import wilcoxon

CSV, KEY, DELIM = "responses.csv", "PAIRING_KEY.json", ";"


def load():
    with open(CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=DELIM))
    key_rows = json.load(open(KEY))
    key = {(k["pair"], int(k["slot"])): k["system"] for k in key_rows}
    # which pairs are A-vs-A' pairs vs A'-vs-B pairs?
    systems_of = defaultdict(set)
    for k in key_rows:
        systems_of[k["pair"]].add(k["system"])
    return rows, key, {p: frozenset(s) for p, s in systems_of.items()}


def analyse(rows, key, systems_of, comparison, target):
    """comparison: frozenset like {'A','Aprime'}. target: the system we count wins for."""
    want = frozenset(comparison)
    seed = defaultdict(lambda: {"style_hit":0, "qual_hit":0, "n":0})
    for r in rows:
        if systems_of.get(r["pair_id"]) != want:
            continue                              # not this comparison's pairs
        for field, tag in (("style_choice","style_hit"), ("quality_choice","qual_hit")):
            c = r[field].strip()
            if c not in ("1","2"):
                continue
            if key[(r["pair_id"], int(c))] == target:
                seed[r["pair_id"]][tag] += 1
        # count n once, on any answered trial
        if r["style_choice"].strip() in ("1","2"):
            seed[r["pair_id"]]["n"] += 1
    return seed


def report(seed, target, n_min=3):
    if not seed:
        print("  no trials found for this comparison — was A rendered and included "
              "in the Full tier? Check the systems present in PAIRING_KEY.json.")
        return
    for tag, label in (("style_hit","STYLE"), ("qual_hit","QUALITY")):
        rate = {s: d[tag]/d["n"] for s, d in seed.items() if d["n"]}
        vals = list(rate.values())
        above = sum(v > 0.5 for v in vals)
        thin  = min(d["n"] for d in seed.values())
        print(f"\n  {label}: mean {target}-preference {st.mean(vals):.3f} "
              f"| {above}/{len(vals)} seeds | thinnest n={thin}")
        nz = [v-0.5 for v in vals if v != 0.5]
        if len(nz) >= 6:
            stat, p = wilcoxon(nz)
            print(f"         Wilcoxon p = {p:.4f}")
        else:
            print(f"         only {len(vals)} seeds / {len(nz)} non-tied "
                  "— report descriptively, too few for Wilcoxon")


def main():
    rows, key, systems_of = load()
    kinds = Counter(frozenset(s) for s in systems_of.values())
    print("comparison types present:", {tuple(sorted(k)): v for k, v in kinds.items()})

    print("\n=== A' vs B (headline — cross-check against wilcoxon_ab.py) ===")
    report(analyse(rows, key, systems_of, {"Aprime","B"}, target="B"), "B")

    print("\n=== A vs A' (context effect — target = A', the one with prepended material) ===")
    seed = analyse(rows, key, systems_of, {"A","Aprime"}, target="Aprime")
    report(seed, "Aprime")
    if seed:
        print("\n  Objective A->A' showed A' CLOSER to the corpus (context helps).")
        print("  Subjective mean > 0.5 here would mean listeners AGREE — A' preferred.")
        print("  If < 6 A-vs-A' seeds have data, this is descriptive support, not a test:")
        print("  Full tier was the only tier with these trials, so n is small by design.")


if __name__ == "__main__":
    main()