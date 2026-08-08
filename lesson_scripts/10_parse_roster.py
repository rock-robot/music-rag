import pdfplumber
from collections import defaultdict
import re

PDF = '/home/whamel/music-rag/scores/A Letter - Full Score.pdf'

FAMILIES = [
    'contrabassoon', 'bassoon', 'piccolo', 'flute', 'oboe', 'english horn',
    'contrabass clarinet', 'bass clarinet', 'alto clarinet', 'clarinet',
    'soprano saxophone', 'alto saxophone', 'tenor saxophone',
    'baritone saxophone', 'saxophone',
    'trumpet', 'horn', 'bass trombone', 'trombone', 'euphonium', 'tuba',
    'double bass', 'harp', 'piano', 'organ', 'timpani',
    'vibraphone', 'xylophone', 'glockenspiel', 'crotales',
    'suspended cymbal', 'tam-tam',
]
# precompute space-stripped family keys, longest first (specificity)
FAMILY_KEYS = sorted(
    [(f, f.replace(' ', '')) for f in FAMILIES],
    key=lambda t: -len(t[1])
)

QUALIFIERS = ['piccolo', 'contrabass', 'soprano', 'sopranino', 'alto',
              'tenor', 'baritone', 'bass']
TRANSP_RE  = re.compile(r'in[a-g][b#]?$|^[b#]$')   # "inbb","ineb","inf", bare "b"/"#"

def squash(raw):
    """Lowercase and strip ALL spaces + artifacts -> matchable key."""
    s = raw.lower()
    s = re.sub(r'[^a-z#]', '', s)      # keep letters and # only
    return s

def find_family(squashed):
    for orig, key in FAMILY_KEYS:
        if key in squashed:
            return orig
    return None

def parse_rows(rows):
    """rows = list of (top, raw_text). Returns structured records, merged."""
    records = []
    for top, raw in rows:
        sq      = squash(raw)
        is_solo = sq.startswith('solo')
        if is_solo:
            sq = sq[4:]
        family  = find_family(sq)
        voices  = re.findall(r'\d+', raw)

        # continuation row? (transposition tail or bare accidental, no family)
        if family is None and (TRANSP_RE.match(sq) or sq in ('b', '#', '')):
            if records:                       # merge upward
                records[-1]['transp'] = records[-1].get('transp') or sq
            continue
        # pure artifact (brace, page number) — drop
        if family is None and not re.search(r'[a-z]{3,}', sq):
            continue

        # qualifier = size word present in the squashed name
        qual = next((q for q in QUALIFIERS if q in sq and q not in (family or '')), None)
        records.append({'raw': raw, 'family': family, 'qualifier': qual,
                        'voices': voices, 'solo': is_solo, 'transp': None})
    return records

X0_MAX, ROW_TOL = 100, 2.0
def extract_roster(pdf_path, x0_max=X0_MAX, row_tol=ROW_TOL):
    with pdfplumber.open(pdf_path) as pdf:
        def margin_count(p):
            return len([w for w in p.extract_words() if w['x0'] <= p.width*0.18])
        page  = max(pdf.pages[:8], key=margin_count)
        words = [w for w in page.extract_words() if w['x0'] < x0_max]
        rows  = defaultdict(list)
        for w in words:
            rows[round(w['top']/row_tol)*row_tol].append(w)
        out = []
        for top in sorted(rows):
            frags = sorted(rows[top], key=lambda w: w['x0'])
            out.append((round(top,1), ' '.join(f['text'] for f in frags)))
        return out

if __name__ == '__main__':
    recs = parse_rows(extract_roster(PDF))
    print(f"{len(recs)} instruments parsed\n")
    for i, r in enumerate(recs):
        fam  = r['family'] or '???'
        qual = f"[{r['qualifier']}]" if r['qualifier'] else ''
        flag = '' if r['family'] else '   <-- UNMATCHED'
        print(f"  {i:2d}  {fam:<16}{qual:<14} solo={int(r['solo'])} "
              f"transp={r['transp'] or '-':<5} voices={r['voices']}{flag}")