r"""Measure the rendered page instead of trusting the build log.

wacv.sty line 46 sets

    \hbadness=10000 \vbadness=10000 \vfuzz=30pt \hfuzz=30pt

so LaTeX does not report an overfull box until it exceeds 30pt, which is 12.7% of
the 236.25pt column. A table can hang well into the gutter and the log stays silent.
Verified empirically: widening a table's inter-column padding from 3pt to 9pt, which
pushes it far past the column, still produced zero "Overfull \hbox" lines.

That makes every grep of the log worthless for this template, including the one I
had been running. This measures the PDF itself: for each page it finds the body
text column boundaries, then reports booktabs-style single-column table rules and
their text when they cross a column. It is deliberately not a general figure-boundary
checker; final rendered-page inspection covers that separate case.

Run after a build:  python check_layout.py
Exit status is 1 if anything overflows, so it can gate a commit.
"""
import sys
from collections import Counter
from pathlib import Path

import fitz

PDFS = [Path(__file__).with_name(n) for n in ("main.pdf", "supp.pdf")]
TOL_PT = 1.5          # antialiasing and glyph side-bearing slack
DPI = 72              # work in PDF points directly


def mode(vals):
    """The most common edge, to half a point. Justified text puts nearly every
    full line at the same margin, so this is the margin, and a handful of wider
    lines cannot move it the way a maximum would."""
    return Counter(round(v * 2) / 2 for v in vals).most_common(1)[0][0]


def scan(pdf):
    """Overflows in one document. The supplement was never scanned at all until it
    turned out to contain an overhanging table of its own."""
    doc = fitz.open(str(pdf))
    problems = []

    # Column edges established once over the WHOLE document, as a fallback.
    #
    # They were derived per page from that page's own justified text, and a page
    # made mostly of tables does not have ten such spans, so `cols` came out empty
    # and every check below was skipped in silence. A table overhanging its column
    # by 36pt sat on exactly such a page and this file reported the document clean.
    # A checker that cannot establish its reference must borrow one, not give up.
    doc_spans = []
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for sp in l.get("spans", []):
                    if sp["bbox"][1] < 90 or sp["size"] < 6.5:
                        continue
                    if sp["text"].strip().isdigit():
                        continue
                    doc_spans.append(sp["bbox"])
    doc_cols = []
    if doc_spans:
        _mid = doc[0].rect.width / 2
        for _name, _grp in (("left", [b for b in doc_spans if b[2] <= _mid + 20]),
                            ("right", [b for b in doc_spans if b[0] >= _mid - 20])):
            if len(_grp) >= 40:
                doc_cols.append((_name, mode(b[0] for b in _grp),
                                 mode(b[2] for b in _grp)))
    if {c[0] for c in doc_cols} != {"left", "right"}:
        problems.append(
            "{}: could not establish both document-level column boundaries"
            .format(pdf.name))

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
                    if s["text"].strip().isdigit():  # ditto, when set at body size
                        continue
                    spans.append(s["bbox"])
        if not spans:
            continue

        # Two columns: split on the page midpoint, then take each column's extent
        mid = page.rect.width / 2
        left = [b for b in spans if b[2] <= mid + 20]
        right = [b for b in spans if b[0] >= mid - 20]
        # Column edges come from the justified body text, where nearly every full
        # line shares one left and one right edge, so the mode is the margin.
        #
        # They must not come from min/max, which is what this check did until it was
        # caught. A table wider than its column contributes its own spans to the
        # group, so the maximum became the table's own right edge and the table was
        # then measured against itself. It could not fail. Two tables the build log
        # had flagged passed this check for that reason.
        # The margins are a property of the TEMPLATE, not of the page, so they are
        # taken from the whole document. Computing them per page took the mode over
        # that page's spans, and on a page which is mostly table the table's own cell
        # edges outnumber the running text and win the mode: page 4 of the supplement
        # resolved its left column as 62.5-84.5 instead of 58.5-295.0, against which
        # nothing can overhang. A 36pt overhang on that page passed in silence.
        cols = list(doc_cols)
        if not cols:
            for name, group in (("left", left), ("right", right)):
                if len(group) < 10:
                    continue
                cols.append((name, mode(b[0] for b in group),
                             mode(b[2] for b in group)))

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
            if not cols:
                continue
            doc_x0 = min(x0 for _, x0, _ in cols)
            doc_x1 = max(x1 for _, _, x1 in cols)
            # A table* may be deliberately narrower than the full text block.  It
            # is still unambiguously two-column when its rule extends materially
            # into both columns, across the gutter.  A one-column spill that only
            # nicks the gutter is not exempted.
            by_x = sorted(cols, key=lambda c: c[1])
            crosses_both = (len(by_x) > 1 and
                            r.x0 < by_x[0][2] - TOL_PT and
                            r.x1 > by_x[-1][1] + TOL_PT)
            if ((len(cols) > 1 and r.x0 <= doc_x0 + TOL_PT and
                 r.x1 >= doc_x1 - TOL_PT) or crosses_both):
                continue
            rc = (r.x0 + r.x1) / 2
            name, x0, x1 = min(cols, key=lambda c: abs(rc - (c[1] + c[2]) / 2))
            if r.x0 < x0 - TOL_PT or r.x1 > x1 + TOL_PT:
                over = max(x0 - r.x0, r.x1 - x1)
                problems.append(
                    "p{} {:5s} column: a rule overhangs by {:.1f}pt "
                    "(rule {:.1f}-{:.1f}, column {:.1f}-{:.1f})".format(
                        pno, name, over, r.x0, r.x1, x0, x1))

        # A TABLE'S TEXT can overhang while its rules do not: booktabs draws the rule
        # to the tabular's declared width, and a cell wider than its column simply
        # prints past it. The withdrawn-statistics table did that, its value column
        # landing on the neighbouring column's caption, and this check called the page
        # clean because it only ever looked at rules.
        #
        # Scoped to TABLE BODIES, not to all text. A first attempt measured every span
        # against the column edge and reported eighteen overhangs on page 1 alone: the
        # centred title, the abstract and a full-width figure are all legitimately
        # outside one column, and a check with eighteen false positives is one its
        # reader learns to skip. A table is identified as two or more rules sharing an
        # x-extent, which is what booktabs draws, and only spans between the topmost
        # and bottommost of those rules are measured.
        by_extent = {}
        for r in rules:
            by_extent.setdefault((round(r.x0), round(r.x1)), []).append(r)
        for (rx0, rx1), group in by_extent.items():
            if len(group) < 2:
                continue
            top = min(r.y0 for r in group)
            bot = max(r.y1 for r in group)
            if not cols:
                continue
            doc_x0 = min(x0 for _, x0, _ in cols)
            doc_x1 = max(x1 for _, _, x1 in cols)
            by_x = sorted(cols, key=lambda c: c[1])
            crosses_both = (len(by_x) > 1 and
                            rx0 < by_x[0][2] - TOL_PT and
                            rx1 > by_x[-1][1] + TOL_PT)
            if ((len(cols) > 1 and rx0 <= doc_x0 + TOL_PT and
                 rx1 >= doc_x1 - TOL_PT) or crosses_both):
                continue
            rc = (rx0 + rx1) / 2
            name, x0, x1 = min(cols, key=lambda c: abs(rc - (c[1] + c[2]) / 2))
            if rx0 < x0 - TOL_PT or rx1 > x1 + TOL_PT:
                continue        # the rule itself already overhangs; reported above
            for b in spans:
                if not (top - 2 <= b[1] and b[3] <= bot + 2):
                    continue
                # and it must START inside the table. Filtering on the vertical
                # band alone swept in the NEIGHBOURING column's text at the same
                # height and reported twenty overhangs on one clean page.
                if not (rx0 - 5 <= b[0] <= rx1 + 5):
                    continue
                if b[2] > x1 + TOL_PT:
                    problems.append(
                        "p{} {:5s} column: table text overhangs by {:.1f}pt "
                        "(text ends {:.1f}, column ends {:.1f})".format(
                            pno, name, b[2] - x1, b[2], x1))

    return problems, doc.page_count


def crossdoc():
    """main.tex and supp.tex are separate documents, so a \\cref in one pointing at a
    label defined only in the other resolves to nothing and typesets as ??.

    LaTeX does warn, but the warning is one line among hundreds and it has slipped
    through twice: once for an equation and once for a section. Both were caught by
    reading the built PDF. This makes it a check that fails."""
    import re
    here = Path(__file__).parent
    groups = {
        "main": [here / "main.tex"] + sorted((here / "sec").glob("*.tex")),
        "supp": [here / "supp.tex", here / "supp_perfold.tex"],
    }
    problems = []
    for name, files in groups.items():
        text = ""
        for f in files:
            if f.exists():
                text += f.read_text(encoding="utf-8")
        used = set(re.findall(r"\\[cC]ref\{([^}]*)\}", text))
        used = {k.strip() for grp in used for k in grp.split(",")}
        defined = set(re.findall(r"\\label\{([^}]*)\}", text))
        for miss in sorted(used - defined):
            problems.append(
                "{}: \\cref{{{}}} has no \\label in that document".format(name, miss))
    return problems


def bodylimit(pdf, limit=8):
    """The body must end within `limit` pages. References and the supplement are free.

    WACV's limit is on the body, and nothing here was checking it: page_count was
    read and printed but never asserted, so a spill past page 8 passed this gate
    every time and was caught only by reading the PDF. It spilled repeatedly.

    The body ends where the References heading begins, so that heading's position is
    the measurement. It must be found on the rendered page and not in extracted line
    text: in review mode the heading's line begins with a margin line number and, in a
    two-column layout, extraction merges the facing column onto the same line, so it
    comes out as "624 References [16] A. Author, B. Author, and ...". A line-anchored
    regex therefore never matches, which is how the first version of this function
    passed a document whose body ran to page 12. It is now found as a text span.
    """
    doc = fitz.open(str(pdf))

    # The running head and the review-mode margin line numbers are set in a sans face
    # (NimbusSanL); everything the body contributes is serif. That separates them far
    # more reliably than a y threshold, which is what broke the first version: the
    # References heading sits at y=72.8, inside the band a running-head filter drops.
    def serif(sp):
        return not sp["font"].startswith("NimbusSanL")

    ref = None
    for pno, page in enumerate(doc, 1):
        for blk in page.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                for sp in ln.get("spans", []):
                    if sp["text"].strip() == "References" and sp["size"] >= 11:
                        ref = (pno, sp["bbox"], page.rect.width / 2)
        if ref:
            break
    if ref is None:
        return ["{}: References heading not found; body limit cannot be checked"
                .format(pdf.name)]
    ref_page, rb, mid = ref
    if ref_page <= limit:
        return []
    if ref_page > limit + 1:
        return ["{}: body runs to at least page {}, limit is {}".format(
            pdf.name, ref_page - 1, limit)]

    # References begins on the first page past the limit, which is legal only if it
    # begins at the top of that page. In reading order the left column precedes the
    # right, so anything serif ahead of the heading is body that has spilled.
    left = rb[2] <= mid
    spill = 0
    for blk in doc[ref_page - 1].get_text("dict")["blocks"]:
        for ln in blk.get("lines", []):
            for sp in ln.get("spans", []):
                if not serif(sp) or not sp["text"].strip():
                    continue
                if sp["bbox"][1] < 90:
                    continue
                x1, y1 = sp["bbox"][2], sp["bbox"][3]
                ahead = (y1 <= rb[1] - 1) if (x1 <= mid) == left else (x1 <= mid)
                if ahead:
                    spill += 1
    if spill:
        return ["{}: {} body span(s) sit ahead of the References heading on page {}, "
                "limit is {}".format(pdf.name, spill, ref_page, limit)]
    return []


def main():
    print("=" * 72)
    print("RENDERED LAYOUT CHECK")
    print("the build log cannot do this: wacv.sty sets hfuzz=30pt")
    print("=" * 72)
    missing = [f for f in PDFS if not f.exists()]
    if missing:
        for f in missing:
            print("  NOT BUILT: {}".format(f.name))
        print("")
        print("a document that was not built is not a document that passed")
        return 1
    total = 0
    for f in PDFS:
        problems, pages = scan(f)
        print("  {} ({} pages)".format(f.name, pages))
        for p in problems:
            print("    " + p)
        if not problems:
            print("    no rule or table extends past its column")
        total += len(problems)
    # The limit is a rule about the MAIN paper. The supplement has none, and once it
    # gained its own References section this loop began applying the eight-page limit
    # to it and failing a document that cannot be over length. A checker that fires
    # on a document the rule does not cover teaches its reader to ignore it.
    limited = [f for f in PDFS if f.name == "main.pdf"]
    over = [m for f in limited for m in bodylimit(f)]
    for m in over:
        print("  BODY OVER LIMIT      " + m)
        total += 1
    if not over:
        print("  body ends within the page limit ({} checked, supplement exempt)"
              .format(", ".join(f.name for f in limited)))
    xd = crossdoc()
    print("")
    if xd:
        for x in xd:
            print("  CROSS-DOCUMENT REF  " + x)
    else:
        print("  every cref resolves within its own document")
    total += len(xd)
    if total:
        print("")
        print("{} problem(s)".format(total))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
