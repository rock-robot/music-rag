import pdfplumber
from collections import defaultdict

PDF = '/home/whamel/music-rag/scores/A Letter - Full Score.pdf'
X0_MAX   = 100      # labels live left of ~100; clefs/timesigs/tempo sit right of it
ROW_TOL  = 2.0      # fragments within this many points of 'top' = same line

def extract_roster(pdf_path, x0_max=X0_MAX, row_tol=ROW_TOL):
    with pdfplumber.open(pdf_path) as pdf:
        # reuse the detector: most left-margin words = first system page
        def margin_count(p):
            return len([w for w in p.extract_words() if w['x0'] <= p.width*0.18])
        page = max(pdf.pages[:8], key=margin_count)

        # 1. FILTER: keep only true left-margin words
        words = [w for w in page.extract_words() if w['x0'] < x0_max]

        # 2. GROUP by vertical position (rounded to tolerance)
        rows = defaultdict(list)
        for w in words:
            key = round(w['top'] / row_tol) * row_tol
            rows[key].append(w)

        # 3. STITCH: within each row, order by x0 and join
        labels = []
        for top in sorted(rows):
            frags = sorted(rows[top], key=lambda w: w['x0'])
            text  = ' '.join(f['text'] for f in frags)
            labels.append((round(top, 1), text))
        return labels

for top, text in extract_roster(PDF):
    print(f"  top={top:7.1f}   {text!r}")