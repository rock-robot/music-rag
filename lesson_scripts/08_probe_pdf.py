import pdfplumber

PDF = '/home/whamel/music-rag/scores/A Letter - Full Score.pdf'

def left_margin_words(page, margin_frac=0.18):
    """Words whose left edge sits in the leftmost `margin_frac` of the page."""
    cutoff = page.width * margin_frac
    words = [w for w in page.extract_words() if w['x0'] <= cutoff]
    return sorted(words, key=lambda w: w['top'])   # top-to-bottom = score order

with pdfplumber.open(PDF) as pdf:
    print(f"{len(pdf.pages)} pages, page size {pdf.pages[0].width:.0f}x{pdf.pages[0].height:.0f}\n")

    # 1. Find the first "music" page: the one with the most left-margin words.
    counts = [(i, len(left_margin_words(p))) for i, p in enumerate(pdf.pages[:8])]
    print("left-margin word counts per page:")
    for i, c in counts:
        print(f"  page {i}: {c} margin words")
    first_system = max(counts, key=lambda t: t[1])[0]
    print(f"\n-> guessing page {first_system} is the first full system\n")

    # 2. Dump that page's left margin, top to bottom — this should be the roster.
    print(f"--- left-margin labels on page {first_system} (top to bottom) ---")
    for w in left_margin_words(pdf.pages[first_system]):
        print(f"  top={w['top']:6.1f}  x0={w['x0']:5.1f}  {w['text']!r}")