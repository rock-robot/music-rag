"""wilcoxon_ab.py — significance test for the A'-vs-B listening headline.

Per-seed preference rates (fraction choosing B) tested against 0.5 = no preference.
Wilcoxon signed-rank uses both the direction AND the magnitude of each seed's
departure from 0.5, which the sign test throws away. Analysis is per-seed, never
pooled -- a seed with 10 judgments and one with 4 get equal weight.

Run from music-rag.  Needs responses.csv (semicolon-delimited) + PAIRING_KEY.json.
"""
import csv, json, statistics as st
from collections import defaultdict
from scipy.stats import wilcoxon

CSV   = "responses.csv"
KEY   = "PAIRING_KEY.json"
DELIM = ";"                       # Supabase export is semicolon-delimited


def load():
    with open(CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=DELIM))
    key = {(k["pair"], int(k["slot"])): k["system"] for k in json.load(open(KEY))}
    return rows, key


def per_seed_rate(rows, key, field, target="B", among=("Aprime", "B")):
    """{seed: fraction choosing `target`} over trials whose pair is an `among` pair.
    Also returns judgments-per-seed so thin seeds can be flagged."""
    seed = defaultdict(lambda: {"hit": 0, "n": 0})
    for r in rows:
        c = r[field].strip()
        if c not in ("1", "2"):
            continue
        system = key.get((r["pair_id"], int(c)))
        if system not in among:               # skip trials from the other comparison
            continue
        seed[r["pair_id"]]["n"]  += 1
        seed[r["pair_id"]]["hit"] += (system == target)
    rate = {s: d["hit"] / d["n"] for s, d in seed.items() if d["n"]}
    n_of = {s: d["n"] for s, d in seed.items() if d["n"]}
    return rate, n_of


def test(name, rate, n_of):
    vals = list(rate.values())
    diffs = [v - 0.5 for v in vals]
    mean = st.mean(vals)
    above = sum(v > 0.5 for v in vals)
    ties  = sum(d == 0 for d in diffs)

    print(f"\n{name}")
    print(f"  {len(vals)} seeds | mean B-preference {mean:.3f} "
          f"| {above}/{len(vals)} seeds favour B | thinnest seed n={min(n_of.values())}")

    nonzero = [d for d in diffs if d != 0]
    if len(nonzero) < 6:
        print("  too few non-tied seeds for a meaningful Wilcoxon")
        return
    stat, p = wilcoxon(nonzero)               # drops exact-0.5 seeds automatically
    verdict = ("below 0.5 -> A' preferred" if mean < 0.5 else
               "above 0.5 -> B preferred") if p < 0.05 else "no reliable preference"
    print(f"  Wilcoxon p = {p:.4f}   ({verdict}"
          f"{f', {ties} seed(s) exactly tied and dropped' if ties else ''})")


def main():
    rows, key = load()
    print(f"{len(rows)} answers loaded")
    for field, label in (("style_choice", "STYLE  — 'sounds like the same composer'"),
                         ("quality_choice", "QUALITY — 'the better piece of music'")):
        rate, n_of = per_seed_rate(rows, key, field)
        test(label, rate, n_of)

    print("\nInterpretation: mean < 0.5 means listeners preferred A' (the length-matched"
          "\ncontrol) over B (retrieval). A p >= 0.05 with mean near 0.5 is a NULL that"
          "\nconverges with the objective A'->B result — report it as convergence, not failure.")


if __name__ == "__main__":
    main()