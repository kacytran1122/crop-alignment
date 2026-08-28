r"""Presentation faults that survive a clean build, checked on the rendered PDF.

A LaTeX run can finish with zero errors and still ship a page that is wrong to read.
The three faults here have all occurred in this paper:

  - a lost backslash, which typesets a control sequence as a word. This shipped once
    as "extttcv2.imread" and was caught by looking at a page image, not by the log.
  - an unresolved reference or citation, which renders as ?? or [?]. LaTeX warns, but
    the warning is one line among hundreds, and it slipped through twice.
  - a float that is captioned and never referred to, or numbered out of sequence.

Only checks that can be stated exactly are here. An earlier version of this file also
tried to flag sentences ending mid-clause, columns mixing decimal places, and text set
too small. Every one of those fired only on false positives: sentences that legitimately
end in a preposition, a transposed table whose consistency runs along rows rather than
columns, and the prime in $c'$, which is a math superscript and is meant to be small.
A check that reports twelve problems and has twelve false positives trains its reader
to ignore it, which is worse than not having it.

`selftest()` plants a fault and confirms each scan sees it, so a clean run means the
scans ran rather than that the patterns never matched.

Run after a build:  python check_presentation.py
Exit status is 1 if anything is found, so it can gate a commit.
"""
import re
import sys
import os
from pathlib import Path

import fitz

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDFS = [Path(__file__).with_name(n) for n in ("main.pdf", "supp.pdf")]

# Only tokens that are not also ordinary English. "label", "item", "section" and
# "paragraph" are all words this paper uses in prose, and including them made an
# earlier version report thirty hits, every one a false positive.
CMDS = ["texttt", "textt", "emph", "textbf", "textit", "cref", "Cref", "citep",
        "includegraphics", "toprule", "midrule", "bottomrule", "hspace",
        "vspace", "kap", "mathcal", "mathbf", "linewidth", "centering",
        "scriptsize", "tabcolsep", "setlength", "hfuzz", "vfuzz", "noalign",
        "begin{tabular}", "end{tabular}", "textwidth", "captionof"]


def leaked(pages):
    """Control sequences that reached the page as text."""
    out = []
    for c in CMDS:
        pat = re.compile(r"\b" + re.escape(c) + r"\b")
        for i, t in enumerate(pages, 1):
            for m in pat.finditer(t):
                out.append((i, c, t[max(0, m.start() - 25):m.end() + 25]
                            .replace("\n", " ")))
    return out


# A backslash eaten by a shell turns the escape into its control character, so
# a backslash-t before "exttt" becomes TAB + "exttt", and the same before
# typesets the remainder as a word. This shipped as "extttfinalize_labels()" and
# "5.6imes10^-17" on a supplement page, and the leaked-command scan above could not
# see it, because it looks for the command SPELLED CORRECTLY. These are the forms
# that survive the first character being eaten, for every escape a shell knows.
# "op" was here for 	op and matched "optical", "opened", "operational" and
# "opposite" -- seven false positives on a clean build, which is how a checker
# teaches its reader to ignore it. Every token below is one no English word begins
# with, or one that requires a following brace.
TRUNCATED = ["exttt", "extbf", "extit", "imes", "onumber", "ewcommand",
             "ightarrow", "egin{", "ottomrule", "rac{", "ootnotesize", "space{",
             "lpha", "pprox", "elta", "ambda"]


def truncated_commands(pages):
    """A control sequence whose backslash became a control character."""
    out = []
    for c in TRUNCATED:
        # anchored on a word start so "top" inside "stop" is not a hit
        pat = re.compile(r"(?<![A-Za-z])" + re.escape(c))
        for i, t in enumerate(pages, 1):
            for m in pat.finditer(t):
                ctx = t[max(0, m.start() - 20):m.end() + 22].replace(chr(10), " ")
                # a real word can start this way; require the tell-tale, which is
                # that the character before it is whitespace AND what follows runs
                # straight into a brace or an identifier
                nxt = t[m.end():m.end() + 1]
                if nxt and (nxt.isalpha() or nxt in "{0123456789"):
                    out.append((i, c, ctx))
    return out


def raw_control_chars(tex_files):
    """LaTeX source should not contain a raw tab: it is what a lost backslash
    leaves behind, and it is invisible in an editor."""
    out = []
    for f in tex_files:
        try:
            body = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(body.split(chr(10)), 1):
            for ch, name in ((chr(9), "TAB"), (chr(11), "VT"), (chr(12), "FF")):
                if ch in line:
                    out.append((f.name, i, name, line.strip()[:60]))
    return out


def _undefined_in_log_text(body, name):
    """Undefined commands, references and citations in one log string."""
    out = []
    lines = body.split(chr(10))
    for i, line in enumerate(lines):
        if ("Citation" in line and "undefined" in line) or (
                "Reference" in line and "undefined" in line):
            out.append((name, line.strip()[:70]))
            continue
        if line.startswith("! Undefined control sequence."):
            ctx = ""
            for j in range(i + 1, min(i + 4, len(lines))):
                if lines[j].startswith("l."):
                    ctx = lines[j].strip()[:70]
                    break
            out.append((name, ctx))
    return out


def undefined_in_log(logs):
    """Commands LaTeX could not resolve, read from the build log.

    A lost backslash has two fates. If it eats the escape into a control
    character, the remainder is typeset and the page scans above catch it. If the
    backslash survives but the name is truncated, LaTeX raises "Undefined control
    sequence", prints nothing, and finishes with exit status 0 -- so the page looks
    almost right, the PDF carries no leaked text, and every scan above passes. Four
    such commands sat in the supplement while both page scans reported it clean.
    The log named all four, on the run that produced the file we were reading.
    """
    out = []
    for f in logs:
        try:
            body = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        out.extend(_undefined_in_log_text(body, f.name))
    return out


def unresolved(pages):
    """?? for a reference, [?] for a citation."""
    out = []
    # `[?]` was matched literally. LaTeX sets the undefined-citation mark with the
    # bibliography style's inter-entry space inside the brackets, so the page carries
    # "[? ]" and the scan saw nothing while five of them were visible on one
    # supplement page. Whitespace is now optional, and the same for the reference
    # mark, which some styles set as "? ?".
    for pat, what in ((r"\?\s*\?", "unresolved reference"),
                      (r"\[\s*\?\s*\]", "unresolved citation")):
        for i, t in enumerate(pages, 1):
            for _ in re.finditer(pat, t):
                out.append((i, what))
    return out


def floats(pages):
    """Every float captioned must be referred to somewhere, and the numbering of
    each kind must run 1..n with no gap."""
    full = "\n".join(pages)
    out = []
    for kind, short in (("Figure", "Fig."), ("Table", "Tab.")):
        nums = sorted({int(n) for n in re.findall(kind + r"\s+(\d+)\.", full)})
        if nums and nums != list(range(1, len(nums) + 1)):
            out.append("{} numbering is {}, expected 1..{}".format(
                kind, nums, len(nums)))
        for n in nums:
            pat = r"\b(?:" + re.escape(short) + r"|" + kind + r")\s*" + str(n) + r"\b"
            # the caption itself matches once, so a referenced float matches twice
            if len(re.findall(pat, full)) < 2:
                out.append("{} {} is captioned but never referred to".format(kind, n))
    return out


def selftest():
    """Each scan must see a planted fault and must ignore a clean page."""
    clean = ["WACV 2027 Submission #*****. Figure 1. A caption. See Fig. 1 here."]
    cases = [
        ("leaked", leaked, ["a texttt b"], clean),
        ("unresolved", unresolved, ["see Section ?? for this"], clean),
        ("floats", floats, ["Figure 1. A caption with no reference anywhere."], clean),
        ("truncated", truncated_commands, ["a extttfoo b and 5.6imes10"], clean),
        # the exact forms that shipped: a spaced citation mark and a spaced ref mark
        ("spaced [? ]", unresolved, ["architecture U-Net [? ] U-Net"], clean),
        ("spaced ? ?", unresolved, ["see Section ? ? for this"], clean),
    ]
    planted = ("junk" + chr(10) + "! Undefined control sequence."
               + chr(10) + "l.637 $+0.75$. " + chr(92) + "exttt" + chr(10)
               + "Package natbib Warning: Citation `x2015y' on page 1 undefined"
               + chr(10))
    good_log = "Output written on good.pdf (9 pages)." + chr(10)
    hit = len(_undefined_in_log_text(planted, "planted.log"))
    miss = len(_undefined_in_log_text(good_log, "good.log"))
    print("  self-test {:11s} plant={} clean={} -> {}".format(
        "undefined", hit, miss, "OK" if (hit == 2 and miss == 0) else "BROKEN"))
    if hit != 2 or miss != 0:
        return False
    ok = True
    for name, fn, planted, good in cases:
        hit, miss = len(fn(planted)), len(fn(good))
        if hit < 1 or miss != 0:
            ok = False
        print("  self-test {:11s} plant={} clean={} -> {}".format(
            name, hit, miss, "OK" if (hit >= 1 and miss == 0) else "BROKEN"))
    return ok


def main():
    print("=" * 68)
    print("PRESENTATION CHECK")
    print("a clean build is not a readable page")
    print("=" * 68)
    if not selftest():
        print("\nthe scans themselves are broken; fix them before trusting a pass")
        return 1
    missing = [f for f in PDFS if not f.exists()]
    if missing:
        for f in missing:
            print("  NOT BUILT: {}".format(f.name))
        return 1

    total = 0
    tex = (sorted(Path(__file__).parent.glob("*.tex"))
           + sorted((Path(__file__).parent / "sec").glob("*.tex")))
    ctrl = raw_control_chars(tex)
    print("")
    print("  {} LaTeX source file(s) scanned for raw control characters".format(
        len(tex)))
    for name, i, what, line in ctrl:
        print("    RAW {} IN SOURCE  {}:{}  {!r}".format(what, name, i, line))
        total += 1
    if not ctrl:
        print("    none, so no backslash has been eaten into one")

    logs = [f.with_suffix(".log") for f in PDFS]
    missing_logs = [f for f in logs if not f.exists()]
    allow_no_logs = os.environ.get("PRESENTATION_ALLOW_NO_LOGS") == "1"
    for f in missing_logs:
        print("  NOT BUILT: {} is missing, so undefined commands/citations cannot "
              "be checked".format(f.name))
        if not allow_no_logs:
            total += 1
    und = undefined_in_log(logs)
    print("  {} build log(s) scanned for undefined control sequences".format(
        len([f for f in logs if f.exists()])))
    for name, ctx in und:
        print("    UNDEFINED COMMAND  {}  {}".format(name, ctx))
        total += 1
    if not und and not missing_logs:
        print("    none, so every command in the sources resolved")

    for pdf in PDFS:
        doc = fitz.open(str(pdf))
        pages = [p.get_text("text") for p in doc]
        print("\n  {} ({} pages)".format(pdf.name, doc.page_count))
        found = False
        for i, c, ctx in leaked(pages):
            print("    LEAKED COMMAND  p{} {!r}: ...{}...".format(i, c, ctx))
            total += 1
            found = True
        for i, what in unresolved(pages):
            print("    {}  p{}".format(what.upper(), i))
            total += 1
            found = True
        for msg in floats(pages):
            print("    FLOAT  {}".format(msg))
            total += 1
            found = True
        for i, c, ctx in truncated_commands(pages):
            print("    LOST BACKSLASH  p{} {!r}: ...{}...".format(i, c, ctx))
            total += 1
            found = True
        if not found:
            print("    no leaked commands, no unresolved refs, floats numbered "
                  "and referred to")

    if total:
        print("\n{} problem(s)".format(total))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
