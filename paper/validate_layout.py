r"""Measure the rendered page instead of trusting the build log.

wacv.sty line 46 sets

    \hbadness=10000 \vbadness=10000 \vfuzz=30pt \hfuzz=30pt

so LaTeX does not report an overfull box until it exceeds 30pt, which is 12.7% of
the 236.25pt column. A table can hang well into the gutter and the log stays silent.
Verified empirically: widening a table's inter-column padding from 3pt to 9pt, which
pushes it far past the column, still produced zero "Overfull \hbox" lines.

That makes every grep of the log worthless for this template, including the one I
had been running. This measures the PDF itself: for each page it finds the body
text column boundaries, then reports any drawn rule or text span that crosses them.

Run after a build:  python validate_layout.py
Exit status is 1 if anything overflows, so it can gate a commit.
"""
import sys
from pathlib import Path

import pymupdf

PDF = Path(__file__).with_name("crop_alignment_wacv_2027.pdf")
TOL_PT = 1.5          # antialiasing and glyph side-bearing slack
DPI = 72              # work in PDF points directly


def main():
    doc = pymupdf.open(str(PDF))
    problems = []

    for pno, page in enumerate(doc, 1):
        # Body text spans, excluding the header band and the margin line numbers.
        spans = []
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    if s["bbox"][1] < 90:            # running head
                        continue
                    if s["size"] < 6.5:              # line numbers in the margin
                        continue
                    spans.append(s["bbox"])
        if not spans:
            continue

        # Two columns: split on the page midpoint, then take each column's extent
        mid = page.rect.width / 2
        left = [b for b in spans if b[2] <= mid + 20]
        right = [b for b in spans if b[0] >= mid - 20]
        cols = []
        for name, group in (("left", left), ("right", right)):
            if len(group) < 10:
                continue
            x0 = min(b[0] for b in group)
            x1 = max(b[2] for b in group)
            cols.append((name, x0, x1))

        # Rules drawn by booktabs are thin filled rectangles
        # booktabs rules span most of their table; the title block's framed box and
        # the header band draw short rules that are not tables and must be ignored.
        colw = max((x1 - x0) for _, x0, x1 in cols) if cols else page.rect.width
        rules = []
        for d in page.get_drawings():
            r = d["rect"]
            if r.height < 2.0 and r.width > 0.45 * colw:
                rules.append(r)

        for r in rules:
            for name, x0, x1 in cols:
                # a rule belonging to this column
                if r.x0 > x0 - 60 and r.x1 < x1 + 60:
                    if r.x0 < x0 - TOL_PT or r.x1 > x1 + TOL_PT:
                        over = max(x0 - r.x0, r.x1 - x1)
                        problems.append(
                            "p{} {:5s} column: a rule overhangs by {:.1f}pt "
                            "(rule {:.1f}-{:.1f}, column {:.1f}-{:.1f})".format(
                                pno, name, over, r.x0, r.x1, x0, x1))
                    break

    print("=" * 72)
    print("RENDERED LAYOUT CHECK  ({} pages)".format(doc.page_count))
    print("the build log cannot do this: wacv.sty sets hfuzz=30pt")
    print("=" * 72)
    if problems:
        for p in problems:
            print("  " + p)
        print("\n{} overflow(s)".format(len(problems)))
    else:
        print("  no rule or table extends past its column")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
