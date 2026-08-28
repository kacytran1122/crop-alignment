"""Check every headline number in the draft against the file it came from.

Numbers in a paper go stale silently. Three have already done it here: the mosaic
required-n stayed at 238 after it became 197, the published fusion gain stayed at 65
after it became 70, and figure 1 carried both plus two more while rendering
perfectly. None of those crashed anything, and none would have been caught by
reading the draft.

So this reaggregates claims where per-run outputs exist, re-reads explicitly
summary-only quantities, and checks that each formatted result actually appears in
the submitted text. It cannot verify prose, and it deliberately does not try.

A check that fails means one of three things, and the fix differs:
  - the draft is stale        -> update the draft
  - the analysis was rerun    -> confirm the new value is right, then update
  - the check is wrong        -> fix the check, and say so

Run from the repository root on the machine holding runs/.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

import os

# The repository, found from this file, is the first place to look. The training
# box's absolute path was the ONLY place it looked, so a clone found nothing and
# said so in a way that exited 0.
REPO = Path(__file__).resolve().parents[2]
RELEASE_MODE = (REPO / "MANIFEST.json").exists()
# Cold-release verification must use only the hashed files inside the release.
# Otherwise an ambient SEAICE_ROOT can silently replace a shipped input.
ROOTS = [REPO] if RELEASE_MODE else (
    ([Path(os.environ["SEAICE_ROOT"])] if os.environ.get("SEAICE_ROOT") else [])
    + [REPO])
# Every current input below has one canonical repository-relative path.  A basename
# fallback is unsafe here: several gates contain files called j1_summary.json and
# labeller_survey.json, so a successful lookup could silently audit the wrong run.

# The paper is the LaTeX now. PAPER_DRAFT_V3.md is a superseded Markdown draft, and
# auditing it meant the printed numbers were never the ones being checked.
if len(sys.argv) > 1:
    SOURCES = [Path(sys.argv[1])]
else:
    SOURCES = sorted((REPO / "paper" / "sec").glob("*.tex"))
    for extra in (REPO / "paper" / "main.tex", REPO / "paper" / "supp.tex",
                  REPO / "paper" / "supp_perfold.tex"):
        if extra.exists():
            SOURCES.append(extra)
if not SOURCES:
    raise SystemExit("no source files to audit")
DRAFT = ", ".join(f.name for f in SOURCES)
def rendered_source(path):
    """Return only the source TeX actually reads."""
    text = path.read_text(encoding="utf-8").split("\\end{document}", 1)[0]
    # A correct number in a comment is not a claim and must not rescue a stale
    # rendered occurrence. Remove unescaped comments before token matching.
    return re.sub(r"(?m)(?<!\\)%.*$", "", text)


TEXT = "\n".join(rendered_source(f) for f in SOURCES)
# LaTeX writes numbers as $+0.1106$ and 80{,}215; strip what would defeat matching
TEXT = TEXT.replace("{,}", ",").replace("$", "").replace("\\,", " ")

# The body alone, for the retired-value check below. The supplement is excluded on
# purpose: it describes the errors, so it quotes superseded numbers deliberately.
BODY = chr(10).join(rendered_source(f) for f in SOURCES
                 if f.name not in ("supp.tex", "supp_perfold.tex"))
BODY = BODY.replace("{,}", ",").replace("$", "").replace("\\,", " ")

# README.md and REPRODUCE.md quote the same headline numbers to anyone who reads
# the repository, and they went stale in exactly the same way the paper did: both
# still carried the pre-rescore dial after the paper had been corrected. They are
# scanned for retired values too.
for _doc in ("README.md", "REPRODUCE.md"):
    _f = REPO / _doc
    if _f.exists():
        BODY += chr(10) + _f.read_text(encoding="utf-8")

MIN_CHECKS = int(os.environ.get("VERIFY_MIN_CHECKS", "25"))

ok, bad, skipped, gone = [], [], [], []

# Checks whose INPUTS are documented as not shipping, so they cannot run inside the
# released archive. A skip fails this run by design, which is right for an input that
# should have been there and wrong for one the archive was never meant to carry. So
# these are a separate category: printed loudly, and counted toward the headline
# total in both places so the two runs report the same number.
absent = []


def needs_data(label, why):
    """A check that cannot run here because its input is deliberately not shipped."""
    absent.append((label, why))


missing_inputs = []


def load(rel):
    """Load one exact repository-relative result path. Never fall back by name."""
    cands = [root / rel for root in ROOTS]
    for f in cands:
        try:
            if f.exists():
                return json.loads(f.read_text())
        except OSError:
            continue
    missing_inputs.append(rel)
    return None


def _number_patterns(s):
    """Sign- and decimal-aware patterns for one formatted numeric token."""
    variants = [s, s.replace("-", "−")]
    if s.startswith("+"):
        variants.append(s[1:])
    elif s and s[0].isdigit():
        variants.append("+" + s)
    return [r"(?<![\d.+\-−])" + re.escape(v) + r"(?!\d|\.\d)"
            for v in dict.fromkeys(variants)]


# Executable regressions for two false-pass modes found during audit.
assert not any(re.search(p, "-0.1106") for p in _number_patterns("+0.1106"))
assert not any(re.search(p, "8.5") for p in _number_patterns("8"))
assert any(re.search(p, "+0.1106") for p in _number_patterns("+0.1106"))


def check(label, value, fmt="{:+.4f}", source=""):
    """Assert the formatted value appears in the draft as a standalone number.

    Plain substring matching is useless for short integers: searching for "8" from
    a required-n of 8 matches nearly any page and the check always passes. So the
    search is anchored on digit boundaries -- the match must not have another digit
    immediately before or after it, and must not sit inside a longer decimal.
    """
    s = fmt.format(value)
    for pat in _number_patterns(s):
        if re.search(pat, TEXT):
            ok.append((label, s))
            return
    bad.append((label, s, source))


def retired(label, value, why):
    """A number the paper used to print, which must not appear in the body again.

    Every check above asks whether a CORRECT value is present. None of them can see
    a SUPERSEDED value still sitting in the text, because nothing recomputes a
    number that no longer exists. That gap is not hypothetical: after the rescore
    the body still carried the old dial contrast t, the old control reading, the old
    flood Omega and the old within-event correlation, and all four survived a run in
    which every other number verified and the audit reported no mismatches.

    The supplement is exempt, because it describes these errors and quotes the
    superseded values on purpose.
    """
    pat = "(?<![0-9.])" + re.escape(value) + "(?![0-9])"
    alt = "(?<![0-9.])" + re.escape(value.replace("-", chr(0x2212))) + "(?![0-9])"
    if re.search(pat, BODY) or re.search(alt, BODY):
        bad.append((label + ", retired", value + " is still in the body", why))
    else:
        ok.append((label + ", retired", "absent: " + value))


def skip(label, why):
    skipped.append((label, why))


def withdrawn(label, why):
    """A check that is gone on purpose, because the claim it audited was
    withdrawn. Recorded so a number that stopped being recomputed cannot pass as
    one that still is, but it does not fail the run: five of these made a clean
    exit impossible, which meant a genuine skip had nowhere to show up."""
    gone.append((label, why))


# ---- crop alignment, sea ice ---------------------------------------------
d = load("gate7_rebuild/results/kappa_summary.json")
if d:
    kn, kc = np.array(d["kappa_noisy"]), np.array(d["kappa_clean"])
    diff = kn - kc
    check("kappa noisy mean", kn.mean(), source="kappa_summary.json")
    check("kappa control mean", kc.mean(), source="kappa_summary.json")
    check("kappa difference", diff.mean(), source="kappa_summary.json")
    withdrawn("kappa difference t", "fold-level inference is not reported")
    check("artefact fraction", 100 * np.mean(d["frac_artefact"]), "{:.2f}",
          "kappa_summary.json")
    check("Omega control", np.mean(d["omega_clean"]), "{:.4f}", "kappa_summary.json")
    if (d.get("primary_seed") == 42 and
            d.get("primary_input_mode") == "optical-only"):
        ok.append(("sea-ice primary provenance", "seed 42 optical-only"))
    else:
        bad.append(("sea-ice primary provenance",
                    repr((d.get("primary_seed"), d.get("primary_input_mode"))),
                    "kappa_summary.json"))
    sens = d.get("by_seed", {}).get("7", {})
    if sens.get("input_mode") == "optical+ICESat-2 photon":
        ks = np.array(sens.get("kappa_noisy", []))
        cs = np.array(sens.get("kappa_clean", []))
        if len(ks) == 17 and len(cs) == 17:
            check("photon-enabled sensitivity", np.mean(ks - cs),
                  source="kappa_summary.json")
        else:
            bad.append(("photon-enabled sensitivity", "not 17 paired folds",
                        "kappa_summary.json"))
    else:
        bad.append(("photon-enabled sensitivity provenance",
                    repr(sens.get("input_mode")), "kappa_summary.json"))
else:
    skip("crop alignment", "kappa_summary.json missing")

# ---- calibration, rebuilt (J1) ---------------------------------------------
d = load("gate8_crossdomain/results/j1_summary.json")
if d:
    k, g = d["kappa_by_event"], d["damage_by_event"]
    ev = sorted(set(k["a000"]) & set(g["a000"]))
    ARMS = ["a000", "a025", "a050", "a075", "a100", "perm", "offc"]
    for a in ARMS:
        check("kappa arm {}".format(a),
              float(np.mean([k[a][e] for e in ev])), source="j1_summary.json")
        check("mIoU arm {}".format(a),
              float(np.mean([g[a][e] for e in ev])), "{:.3f}", "j1_summary.json")
    c = np.array([k["a100"][e] - k["a000"][e] for e in ev])
    check("dial contrast", c.mean(), source="j1_summary.json")
    withdrawn("dial contrast t", "fold-level inference is not reported")
    # the permutation control: the paper's strongest claim, previously unchecked
    pc = np.array([k["a100"][e] - k["perm"][e] for e in ev])
    withdrawn("scramble aggregate difference",
              "the paper reports seed-specific contrasts and the retained share")
    withdrawn("scramble difference t", "fold-level inference is not reported")
    cost = np.array([g["a100"][e] - g["a000"][e] for e in ev])
    costp = np.array([g["perm"][e] - g["a000"][e] for e in ev])
    check("cost a100", cost.mean(), "{:+.4f}", "j1_summary.json")
    check("cost perm", costp.mean(), "{:+.4f}", "j1_summary.json")
    share = costp.mean() / cost.mean()
    if "{:.0f}\\%".format(100 * share) in TEXT or "{:.0f} per".format(100 * share) in TEXT:
        ok.append(("permutation keeps this share of the cost",
                   "{:.0f}%".format(100 * share)))
    else:
        bad.append(("permutation keeps this share of the cost",
                    "{:.0f}%".format(100 * share), "j1_summary.json"))
else:
    skip("calibration", "runs/j1_summary.json missing")

# ---- arm construction and scramble geometry -------------------------------
_arm_meta = load("gate8_crossdomain/results/j1_arms_meta.json")
if _arm_meta:
    _pub = _arm_meta["arms"]["pub"]
    _a0 = _arm_meta["arms"]["a000"]
    if _pub != _a0:
        bad.append(("pub equals a000", "metadata rows differ",
                    "j1_arms_meta.json"))
    else:
        ok.append(("pub equals a000", "identical metadata"))
    _shift = (_arm_meta["arms"]["a100"]["mean_db"]
              - _arm_meta["arms"]["a000"]["mean_db"])
    check("dial mean-threshold shift", _shift, "{:.2f}", "j1_arms_meta.json")
    _sg = _arm_meta["scramble_geometry"]
    _ta = _arm_meta["threshold_accounting"]
    check("threshold positions total", _ta["total_positions"], "{:,d}",
          "j1_arms_meta.json")
    check("direct crop fits", _ta["direct_crop_fits"], "{:,d}",
          "j1_arms_meta.json")
    check("finite chip fallbacks", _ta["finite_chip_fallbacks"], "{:,d}",
          "j1_arms_meta.json")
    check("unresolved threshold positions", _ta["unresolved_positions"], "{:,d}",
          "j1_arms_meta.json")
    check("common usable chips", _arm_meta["n_usable_chips"], "{:d}",
          "j1_arms_meta.json")
    check("scramble assignments", _sg["assignments"], "{:,d}",
          "j1_arms_meta.json")
    check("scramble fixed points", _sg["fixed_points"], "{:d}",
          "j1_arms_meta.json")
    check("scramble fixed fraction",
          100 * _sg["fixed_points"] / _sg["assignments"], "{:.2f}",
          "j1_arms_meta.json")
    check("scramble overlapping fraction", 100 * _sg["overlapping_fraction"],
          "{:.2f}", "j1_arms_meta.json")
else:
    skip("arm construction", "j1_arms_meta.json missing")

# ---- cost against human truth -------------------------------------------
# Superseded by the J1 calibration block above, which recomputes the cost
# from the rebuilt arms. The old damage_summary.json is the pre-J1 dial.
withdrawn("cost (pre-J1)", "superseded by the J1 calibration block")

# ---- estimator verification ----------------------------------------------
thr = []
for _root in ROOTS:
    thr = sorted((_root / "runs_s1fc" / "kappa").glob("*_thr.json"))
    if thr:
        break
if not thr:
    # The four per-event results are also released as one file, so the headline
    # |kappa| < 1.4e-15 is verifiable from the repository rather than only on
    # the box that produced it. It was the one abstract number nothing checked.
    en = load("gate8_crossdomain/results/estimator_null.json")
    if en:
        worst = max(abs(v["kappa"]) for v in en.values())
        if worst < 1.4e-15 and all(v["omega"] == 0.0 for v in en.values()):
            ok.append(("estimator |kappa| under 1.4e-15",
                       "max {:.1e} over {} events, omega exactly 0".format(
                           worst, len(en))))
        else:
            bad.append(("estimator |kappa| under 1.4e-15",
                        "max {:.1e}".format(worst), "estimator_null.json"))
if thr:
    worst = max(abs(json.loads(f.read_text())["kappa"]) for f in thr)
    if worst < 1.4e-15:
        ok.append(("estimator |kappa| under 1.4e-15",
                   "max {:.1e} over {} events".format(worst, len(thr))))
    else:
        bad.append(("estimator |kappa| under 1.4e-15",
                    "max {:.1e}".format(worst), "runs_s1fc/kappa/*_thr.json"))
else:
    pass  # covered by estimator_null.json above

# ---- the seed table, recomputed from the per-run outputs -------------------
# This table replaced the flood t and sign-test p, so it is now carrying the weight
# those numbers used to carry. It is recomputed here from the per-run scorer files
# rather than trusted, because a typed table is exactly the kind of number that
# survives a rescore unnoticed.
_prk = None
for _r in list(ROOTS) + [REPO]:
    _c = _r / "gate8_crossdomain" / "results" / "per_run_kappa"
    if _c.exists():
        _prk = _c
        break
if _prk:
    _by = {}
    for _f in _prk.glob("j1_*.json"):
        if "_thr" in _f.name or "_step" in _f.name:
            continue
        _rec = json.loads(_f.read_text())
        _k = (_rec.get("arm"), _rec.get("seed"))
        _by.setdefault(_k, {})[_rec.get("event")] = _rec.get(
            "kappa_pixel", _rec.get("kappa"))
    for _a1, _a0, _name in (("a100", "a000", "dial"), ("a100", "perm", "permutation")):
        for _seed in (7, 42, 123):
            _x, _y = _by.get((_a1, _seed)), _by.get((_a0, _seed))
            if not _x or not _y:
                skip("seed table {} s{}".format(_name, _seed), "per-run arm missing")
                continue
            _ev = sorted(set(_x) & set(_y))
            if len(_ev) != 11:
                bad.append(("seed table {} s{}".format(_name, _seed),
                            "{} events, expected 11".format(len(_ev)),
                            "per_run_kappa"))
                continue
            check("seed table, {} s{}".format(_name, _seed),
                  float(np.mean([_x[e] - _y[e] for e in _ev])),
                  source="per_run_kappa")
else:
    skip("seed table", "per-run kappa outputs absent")

# ---- the qualitative figure's caption, recomputed from the imagery ----------
# The caption prints an artefact fraction and a scan summary. Those are numbers like
# any other, and a figure regenerated on a different region would leave them behind.
# Recomputed here when the imagery is present, skipped honestly when it is not, since
# the archive does not ship it.
_fig = REPO / "gate7_rebuild" / "src" / "fig_mechanism_real.py"
_tifs = REPO / "S2_tiff" / "S2_tiff"
if _fig.exists() and _tifs.exists() and any(_tifs.glob("*.tif")):
    _sys_path = str(REPO / "gate7_rebuild" / "src")
    if _sys_path not in sys.path:
        sys.path.insert(0, _sys_path)
    try:
        import fig_mechanism_real as _fm
        _tif = sorted(_tifs.glob("*.tif"))[4]
        _reg = _fm.read_region(_tif, 3985, 7906, _fm.REGION)
        _f_crop, _, _, _ = _fm.artefact_fraction(_reg)
        from s2_published_survey import scene_context as _sc
        _f_scene, _, _, _ = _fm.artefact_fraction(
            _reg, _sc(_reg), ("bg", "otsu", "n1", "n2"))
        check("figure 5, artefact fraction per crop", 100 * _f_crop, "{:.1f}",
              "S2 imagery")
        # Re-run the published scan rather than trusting its stored summary. The
        # stored JSON is then checked against the fresh per-site aggregation.
        _sv = REPO / "gate7_rebuild" / "results" / "mechanism_survey.json"
        if _sv.exists():
            _d = json.loads(_sv.read_text(encoding="utf-8"))
            _rows = _fm.scan(12, 10, seed=0)
            _fr = np.asarray([x[0] for x in _rows], dtype=float)
            _fresh = {"n_sites": len(_rows),
                      "n_nonempty": int((_fr > 0).sum()),
                      "median": float(np.median(_fr)),
                      "max": float(np.max(_fr))}
            for _key in _fresh:
                if not np.isclose(_fresh[_key], _d[_key], rtol=0, atol=1e-12):
                    bad.append(("survey stored {}".format(_key),
                                "stored {!r}, fresh {!r}".format(
                                    _d[_key], _fresh[_key]),
                                "mechanism_survey.json"))
            check("survey, sites scanned", _fresh["n_sites"], "{:d}", "fresh scan")
            check("survey, sites with a nonempty A", _fresh["n_nonempty"], "{:d}",
                  "fresh scan")
            check("survey, median artefact fraction", 100 * _fresh["median"],
                  "{:.2f}", "fresh scan")
            check("survey, largest artefact fraction", 100 * _fresh["max"],
                  "{:.2f}", "fresh scan")
        else:
            bad.append(("figure 5 survey", "mechanism_survey.json not written",
                        "fig_mechanism_real.py --scan"))
        if _f_scene != 0.0:
            bad.append(("figure 5, scene-fitted arm",
                        "reads {:.4f}, must be exactly 0".format(_f_scene),
                        "S2 imagery"))
        else:
            ok.append(("figure 5, scene-fitted arm is exactly empty", "0.0"))
    except Exception as _e:                      # noqa: BLE001
        skip("figure 5 caption", "could not recompute: {}".format(_e))
else:
    # one entry per check the block would have run, so the headline total is the
    # same number in the repository and in the archive. A block that contributed two
    # checks here and one absence there made the two runs disagree by exactly one,
    # which is the kind of drift this guard exists to catch.
    for _lab in ("figure 5, artefact fraction per crop",
                 "figure 5, scene-fitted arm is exactly empty",
                 "survey, sites scanned",
                 "survey, sites with a nonempty A",
                 "survey, median artefact fraction",
                 "survey, largest artefact fraction"):
        needs_data(_lab, "S2 imagery, public and documented as not shipped")

# The forensic error ledger was removed from the submitted supplement. Its history
# remains in version control, but a submission number checker should not require or
# legitimise a catalogue that is no longer part of the paper.
withdrawn("error-ledger counts", "forensic history removed from the submission")

# ---- the released archive, measured rather than remembered -----------------
# The supplement stated a size the archive had not had for some time: nothing
# recomputed it, because it is a property of files rather than of a result. Both
# facts are read from MANIFEST.json, which exists in the repository and in the
# extracted archive, so this runs in both places rather than skipping in one --
# a check that quietly does not run in the release is the fault this file exists
# to catch.
_man = None
_cands = [r / "MANIFEST.json" for r in ROOTS]
_cands += [REPO / "MANIFEST.json", REPO / "release" / "MANIFEST.json"]
for _c in _cands:
    if _c.exists():
        _man = json.loads(_c.read_text(encoding="utf-8"))
        break
if _man:
    # The paper deliberately avoids brittle archive counts. Integrity is checked by
    # verify.py against the manifest, not by matching a typed count in prose.
    if not _man.get("files"):
        bad.append(("release manifest", "contains no payload rows", "MANIFEST.json"))
    else:
        ok.append(("release manifest", "{} payload rows".format(len(_man["files"]))))
else:
    skip("release archive", "MANIFEST.json not built")

# ---- withdrawn exploratory premium ---------------------------------------
withdrawn("artefact premium",
          "removed from the submission: its estimator was underdefined")
withdrawn("required sample sizes", "independence assumption is not available")

# ---- medians; fold intervals are not reported ------------------------------
# Added after a read-through found a stale median (+0.0307) the audit had not
# covered: it checked means and t statistics and never the order statistics beside
# them. A number audit only protects the numbers it was told about.
withdrawn("premium medians", "artefact premium removed from the submission")

# ---- epoch budget ---------------------------------------------------------
withdrawn("budget comparison",
          "removed: schedule and architecture also change with epoch cap")

# ---- transfer collapse ----------------------------------------------------
d = load("gate8_crossdomain/results/mechanism.json")
if d:
    col = np.array(d["collapse"])
    check("transfer collapse", col.mean(), source="mechanism.json")
    withdrawn("transfer-collapse t", "fold-level inference is not reported")
    # The correlation between label agreement and collapse survives as a number and
    # is still printed in the supplement, but the slope's t was only ever quoted in
    # support of calling it a mechanism check. It is not one: an arm scored against
    # expert labels is bounded by the agreement between the two label sets, so the
    # correlation is expected under any hypothesis. The claim is withdrawn, so the
    # check goes with it rather than failing forever against a sentence we removed.
    check("mechanism correlation",
          float(np.corrcoef(d["agreement"], d["collapse"])[0, 1]), "{:.3f}",
          "mechanism.json")
    withdrawn("mechanism slope t", "withdrawn: the correlation is not a mechanism check")
else:
    skip("transfer collapse", "mechanism.json missing")

# ---- dataset scale --------------------------------------------------------
cm = load("gate8_crossdomain/results/s1f_cache_meta.json")
if cm:
    if "{}".format(cm["n"]) in TEXT:
        ok.append(("flood chips", str(cm["n"])))
    else:
        bad.append(("flood chips", str(cm["n"]), "s1f_cache_meta.json"))

# ---- report ---------------------------------------------------------------
# ---- how many chips each survey actually surveyed -------------------------
# The draft said 446 for a table computed on 445, because one chip carries no valid
# backscatter and is skipped. The top-level n_chips is len(files), not the number
# surveyed, and nothing here was comparing the two.
d = load("gate8_crossdomain/results/labeller_survey.json")
if d:
    per_rule = {v["n_chips"] for v in d["rules"].values()}
    if len(per_rule) != 1:
        bad.append(("S1 survey rules agree on chip count",
                    "|".join(str(n) for n in sorted(per_rule)),
                    "labeller_survey.json"))
    else:
        ok.append(("S1 survey rules agree on chip count", str(per_rule.pop())))
    check("S1 chips actually surveyed",
          min(v["n_chips"] for v in d["rules"].values()), "{:d}",
          "labeller_survey.json")

# ---- does kappa survive a different architecture? --------------------------
d = load("gate8_crossdomain/results/arch_kappa.json")
if d:
    for enc, r in sorted(d.items()):
        if r["n"] < 11:
            skip("architecture {}".format(enc),
                 "only {} of 11 events scored".format(r["n"]))
            continue
        check("arch {} contrast".format(enc), r["contrast"], "{:+.4f}",
              "arch_kappa.json")
        withdrawn("arch {} t".format(enc), "fold-level inference is not reported")
        check("arch {} events".format(enc), r["n"], "{:d}", "arch_kappa.json")
        # the pre-registered falsification was the contrast collapsing to zero
        if not (r["contrast"] > 0 and r["positive"] == r["n"]):
            bad.append(("arch {} contrast survives".format(enc),
                        "{:+.4f}, positive {}/{}".format(
                            r["contrast"], r["positive"], r["n"]),
                        "arch_kappa.json"))
        else:
            ok.append(("arch {} contrast survives".format(enc),
                       "{:+.4f}, {}/{}".format(r["contrast"], r["positive"], r["n"])))

# ---- values the paper has retired, which must not reappear -----------------
# Listed by hand, because only a person knows a number was replaced rather than
# simply dropped. Each was printed in an earlier version and is now superseded.
# Removed from the body because the eleven flood models are leave-one-event-out
# folds sharing training data, so no test over them is licensed. Retired rather
# than deleted: a superseded number that walks back into the text is a fault this
# project has had four times.
retired("flood dial t", "58.24", "no test over dependent folds; supplement only")
retired("flood sign test p", "0.00098", "no test over dependent folds")
retired("flood permutation t", "35.02", "no test over dependent folds")
retired("flood damage t", "-10.49", "no test over dependent folds")
retired("flood slope CI low", "-1.24", "event bootstrap withdrawn from the body")
retired("flood slope CI high", "-0.85", "event bootstrap withdrawn from the body")
retired("pre-rescore dial contrast", "+0.1030", "rescored to +0.2477")
retired("pre-rescore dial contrast t", "9.44", "rescored to 58.75")
retired("pre-rescore a100 kappa", "+0.1029", "rescored to +0.2468")
retired("pre-rescore a000 kappa", "-0.0001", "rescored to -0.0008")
retired("pre-rescore a000 Omega", "0.0040", "rescored to 0.0108")
retired("pre-rescore within-event r", "-0.797", "rescored to -0.928")
retired("pre-rescore within-event slope", "-2.12", "rescored to -1.11")
retired("pre-rescore slope CI low", "-3.45", "rescored to -1.31")
retired("pre-rescore fixed-alpha mean r", "+0.147", "rescored to -0.194")
retired("pre-rescore grid contrast", "+0.1774", "rescored to +0.4286")
retired("pre-rescore grid Spearman", "+0.991", "rescored to +0.936")

# Superseded a second time, by the change from crop-read to pixel weighting. The
# supplement still reports the crop-read values on purpose and is exempt from this
# scan; the body must carry only the pixel-weighted ones.
retired("crop-read dial contrast", "+0.2477", "pixel-weighted is +0.2613")
retired("crop-read a100", "+0.2468", "pixel-weighted is +0.2605")
retired("crop-read perm", "+0.0653", "pixel-weighted is +0.0672")
retired("crop-read paired difference", "+0.1816", "pixel-weighted is +0.1933")
retired("crop-read dial t", "58.75", "pixel-weighted is 58.24")
retired("crop-read permutation t", "34.74", "pixel-weighted is 35.02")
retired("crop-read grid base", "+0.2524", "pixel-weighted is +0.2666")
retired("crop-read grid thinned", "+0.4286", "pixel-weighted is +0.4587")
retired("crop-read grid Spearman", "+0.936", "pixel-weighted is +0.918")
retired("crop-read transformer", "+0.2797", "pixel-weighted is +0.2966")
retired("crop-read second CNN", "+0.2776", "pixel-weighted is +0.2943")
retired("crop-read within-event slope", "-1.11", "pixel-weighted is -1.05")
retired("crop-read fixed-alpha mean", "-0.194", "pixel-weighted is -0.196")
retired("crop-read fixed-alpha largest", "+0.797", "pixel-weighted is +0.774")
retired("crop-read sea-ice treated", "+0.1260", "source-pixel primary is +0.1518")
retired("crop-read sea-ice control", "+0.0154", "source-pixel primary is +0.0097")
retired("crop-read sea-ice difference", "+0.1106", "source-pixel primary is +0.1421")
retired("crop-read sea-ice difference t", "13.01", "pixel-weighted is 10.02")
retired("crop-read sea-ice treated t", "13.97", "pixel-weighted is 10.79")
retired("two-mode sea-ice treated average", "+0.1535",
        "seed 42 optical-only is the primary, not an average across input modes")
retired("two-mode sea-ice control average", "+0.0098",
        "seed 42 optical-only is the primary, not an average across input modes")
retired("two-mode sea-ice contrast average", "+0.1437",
        "seed 42 optical-only is the primary, not an average across input modes")

# ---- what the artefact costs against expert labels -------------------------
# Also unaudited, and also already drifted: Section 6 carried the within-event r,
# slope and interval from the superseded scoring.
d = load("gate8_crossdomain/results/j1_fixed_alpha.json")
if d:
    check("cost mean", d["cost_mean"], "{:+.4f}", "j1_fixed_alpha.json")
    check("cost sd", d["cost_sd"], "{:.4f}", "j1_fixed_alpha.json")
    withdrawn("cost t", "fold-level inference is not reported")
    check("cost worse events", d["cost_worse_events"], "{:d}", "j1_fixed_alpha.json")
    check("within-event r", d["r_within"], "{:+.3f}", "j1_fixed_alpha.json")
    check("within-event slope", d["slope_within"], "{:+.2f}", "j1_fixed_alpha.json")
    withdrawn("slope interval", "fold-level inference is not reported")
    check("fixed-alpha mean r", d["fixed_alpha_mean_r"], "{:+.3f}",
          "j1_fixed_alpha.json")
    check("fixed-alpha largest r", max(d["fixed_alpha_r"]), "{:+.3f}",
          "j1_fixed_alpha.json")
    # the paper withdraws this argument, and the reason it gives is a sign
    # disagreement. If the five ever do agree in sign, that prose is wrong.
    if all(x < 0 for x in d["fixed_alpha_r"]):
        bad.append(("fixed-alpha correlations disagree in sign",
                    "all five negative, so the withdrawal's stated reason is gone",
                    "j1_fixed_alpha.json"))
    else:
        ok.append(("fixed-alpha correlations disagree in sign",
                   "{} of 5 negative".format(sum(x < 0 for x in d["fixed_alpha_r"]))))

# ---- the crop grid kappa is relative to ------------------------------------
# Unaudited until now, and it had already drifted: the supplement carried the
# rescored grid numbers while the results file in the repository still held the
# pre-rescore ones. Both were internally consistent, so nothing caught it.
d = load("gate8_crossdomain/results/j1_grid.json")
if d:
    check("grid events", d["n_events"], "{:d}", "j1_grid.json")
    check("grid contrast stride32", d["contrast"]["stride32"], "{:+.4f}",
          "j1_grid.json")
    check("grid contrast stride64", d["contrast"]["stride64"], "{:+.4f}",
          "j1_grid.json")
    check("grid A frac stride32", 100 * d["a_frac"]["stride32"], "{:.2f}",
          "j1_grid.json")
    check("grid A frac stride64", 100 * d["a_frac"]["stride64"], "{:.2f}",
          "j1_grid.json")
    check("grid Spearman", d["spearman"], "{:+.3f}", "j1_grid.json")
    # the claim the supplement makes about the grid is directional, not just numeric
    if not (d["contrast"]["stride64"] > d["contrast"]["stride32"] > 0):
        bad.append(("grid thinning raises the contrast",
                    "{:+.4f} -> {:+.4f}".format(d["contrast"]["stride32"],
                                                d["contrast"]["stride64"]),
                    "j1_grid.json"))
    else:
        ok.append(("grid thinning raises the contrast",
                   "{:+.4f} -> {:+.4f}".format(d["contrast"]["stride32"],
                                               d["contrast"]["stride64"])))

# ---- cloud masking on a third dataset -------------------------------------
d = load("gate8_crossdomain/results/s3_cloudsen12_survey.json")
if d:
    r = {k.strip(): v for k, v in d["rules"].items()}
    check("S3 patches", d["n_patches"], "{:d}", "s3_cloudsen12_survey.json")
    for label, name in (
            ("S3 fixed B02", "fixed B02 constant"),
            ("S3 fixed whiteness", "fixed whiteness and brightness"),
            ("S3 percentile per scene", "clear-sky percentile, per scene"),
            ("S3 percentile per crop", "clear-sky percentile, per crop"),
            ("S3 Otsu on HOT", "Otsu on HOT, per crop"),
            ("S3 min-max per crop", "min-max per crop, then fixed")):
        check(label + " P(A)", 100 * r[name]["mean"], "{:.2f}",
              "s3_cloudsen12_survey.json")
        check(label + " bal acc", r[name]["balanced_accuracy"], "{:.3f}",
              "s3_cloudsen12_survey.json")
    # the three rules that fit nothing to the crop are per-pixel functions of the
    # scene, so they must be exactly zero on every patch, not merely small
    for name in ("fixed B02 constant", "fixed whiteness and brightness",
                 "clear-sky percentile, per scene"):
        if r[name]["max"] != 0.0 or r[name]["n_nonzero"] != 0:
            bad.append((name + " is exactly zero",
                        "max {!r}, nonzero {}".format(r[name]["max"],
                                                      r[name]["n_nonzero"]),
                        "s3_cloudsen12_survey.json"))
        else:
            ok.append((name + " is exactly zero", "0.00% on all patches"))
    # every rule must be averaged over the same pixels or the columns are not
    # comparable; neither earlier survey checked this
    if not d.get("denominators_agree"):
        bad.append(("S3 rules share a denominator",
                    str(d.get("patches_with_unequal_denominators"))[:40],
                    "s3_cloudsen12_survey.json"))
    else:
        ok.append(("S3 rules share a denominator", "identical"))
    # the pool must actually have been a pool
    if d.get("workers", 1) > 1 and d.get("distinct_worker_pids", 1) < 2:
        bad.append(("S3 parallel run used more than one process",
                    str(d.get("distinct_worker_pids")),
                    "s3_cloudsen12_survey.json"))
    else:
        ok.append(("S3 parallel run used more than one process",
                   str(d.get("distinct_worker_pids"))))
    # the data the numbers came from must be identifiable
    src = d.get("source", {})
    if not (src.get("revision") and src.get("sha256")):
        bad.append(("S3 data revision recorded", "missing",
                    "s3_cloudsen12_survey.json"))
    else:
        ok.append(("S3 data revision recorded", src["revision"][:12]))
    # the dose-response the supplement reports, lowest and highest cloud-fraction bins
    dose = d.get("by_cloud_fraction", {})
    if dose:
        keys = sorted(dose)
        lo_vals = [v for k, v in dose[keys[0]].items() if k != "n"]
        hi_vals = [v for k, v in dose[keys[-1]].items() if k != "n"]
        check("S3 lowest cloud bin min", 100 * min(lo_vals), "{:.1f}",
              "s3_cloudsen12_survey.json")
        check("S3 highest cloud bin max", 100 * max(hi_vals), "{:.1f}",
              "s3_cloudsen12_survey.json")

# ---- a foundation-model labeller with a fixed semantic rule ---------------
d = load("gate8_crossdomain/results/s1_sam_survey.json")
if d:
    a = {k.strip(): v for k, v in d["arms"].items()}
    sam = a["SAM per crop"]
    check("SAM per crop mean P(A)", 100 * sam["mean"], "{:.2f}",
          "s1_sam_survey.json")
    check("SAM per crop max P(A)", 100 * sam["max"], "{:.2f}", "s1_sam_survey.json")
    check("SAM chips", sam["n_chips"], "{:d}", "s1_sam_survey.json")
    # the distribution is strongly right-skewed, so the paper reports the median
    # beside the mean and the count of chips above zero, which is the claim the
    # exact null makes meaningful. Both were printed from a log before the survey
    # recorded per-chip values, and nothing checked them.
    if "per_chip" in sam:
        check("SAM per crop median", 100 * float(np.median(sam["per_chip"])),
              "{:.2f}", "s1_sam_survey.json")
    if sam.get("n_nonzero") != sam["n_chips"]:
        bad.append(("SAM nonzero on every chip",
                    "{} of {}".format(sam.get("n_nonzero"), sam["n_chips"]),
                    "s1_sam_survey.json"))
    else:
        ok.append(("SAM nonzero on every chip",
                   "{}/{}".format(sam["n_nonzero"], sam["n_chips"])))
    # both controls must be exactly zero, not merely small: they are per-pixel
    # functions of the scene, so anything else is a fault in the survey
    for name in ("fixed threshold, per pixel", "SAM per scene"):
        if a[name]["max"] != 0.0:
            bad.append((name + " is exactly zero", repr(a[name]["max"]),
                        "s1_sam_survey.json"))
        else:
            ok.append((name + " is exactly zero", "0.00%"))

d = load("gate8_crossdomain/results/s1_sam_determinism.json")
if d:
    # if the mask generator were not deterministic the survey would be measuring
    # run-to-run variation instead of crop-dependence
    if not d["deterministic"]:
        bad.append(("SAM is deterministic",
                    "{}/{} identical".format(d["identical_labels"], d["pairs"]),
                    "s1_sam_determinism.json"))
    else:
        ok.append(("SAM is deterministic",
                   "{}/{}".format(d["identical_labels"], d["pairs"])))

# ---- the published labeller, run per crop on public optical chips ---------
d = load("gate8_crossdomain/results/s2_published_survey.json")
if d:
    arms = d["arms"]

    def arm(name):
        for k, v in arms.items():
            if k.strip() == name:
                return v
        raise KeyError(name)

    full = arm("full pipeline, fitted per crop")
    repaired = arm("scene background and Otsu")
    for label, name, fmt in (
            ("S2 stage 1 only P(A)", "stage 1 only, fixed HSV constants", "mean"),
            ("S2 published per crop P(A)", "full pipeline, fitted per crop", "mean"),
            ("S2 published per crop max", "full pipeline, fitted per crop", "max"),
            ("S2 scene background P(A)", "scene background", "mean"),
            ("S2 scene Otsu P(A)", "scene Otsu", "mean"),
            ("S2 background and Otsu P(A)", "scene background and Otsu", "mean"),
            ("S2 background and Otsu max", "scene background and Otsu", "max"),
            ("S2 all fitted per scene P(A)", "everything fitted per scene", "mean")):
        check(label, 100 * arm(name)[fmt], "{:.2f}", "s2_published_survey.json")
    # the share of the artefact that repairing the background and the Otsu removes
    check("S2 share removed by repairing two steps",
          100 * (full["mean"] - repaired["mean"]) / full["mean"], "{:.0f}",
          "s2_published_survey.json")
    # Both normalisations are inert in the published configuration. Assert the
    # equality rather than trusting that the printed digits happen to agree.
    for name in ("scene first normalisation", "scene second normalisation"):
        if arm(name)["mean"] != full["mean"]:
            bad.append((name + " equals published arm", "not equal",
                        "s2_published_survey.json"))
        else:
            ok.append((name + " equals published arm", "exact"))
    # nothing may abstain, which is what lets the survey restrict to fully labelled
    # pixels; the restriction was documented as a safeguard before it was measured
    if d.get("unlabelled_pixels", 0) != 0:
        bad.append(("S2 no pixel abstains", str(d["unlabelled_pixels"]),
                    "s2_published_survey.json"))
    else:
        ok.append(("S2 no pixel abstains", "0"))

# ---- the same survey under two other reflectance ceilings -----------------
for tag in ("2000", "5000"):
    d = load("gate8_crossdomain/results/s2_survey_s{}.json".format(tag))
    if d:
        a = {k.strip(): v for k, v in d["arms"].items()}
        check("S2 scale {} published".format(tag),
              100 * a["full pipeline, fitted per crop"]["mean"], "{:.2f}",
              "s2_survey_s{}.json".format(tag))
        check("S2 scale {} residual".format(tag),
              100 * a["scene background and Otsu"]["mean"], "{:.2f}",
              "s2_survey_s{}.json".format(tag))
        check("S2 scale {} clipped".format(tag), 100 * d["clipped_fraction"],
              "{:.1f}", "s2_survey_s{}.json".format(tag))

d = load("gate8_crossdomain/results/s2_mechanism.json")
if d:
    check("S2 crops examined", d["n_crops"], "{:,d}", "s2_mechanism.json")
    check("S2 crops where range differs", d["first"]["range_differs"], "{:d}",
          "s2_mechanism.json")
    check("S2 constant arrays", d["first"]["constant_array"], "{:d}",
          "s2_mechanism.json")
    check("S2 first normalisation is the identity",
          100 * d["first"]["identity_fraction"], "{:.1f}", "s2_mechanism.json")
    check("S2 second normalisation is the identity",
          100 * d["second"]["identity_fraction"], "{:.1f}", "s2_mechanism.json")
    for k in ("first", "second"):
        if d[k]["pixels_changed"] != 0:
            bad.append(("S2 {} normalisation changes no label".format(k),
                        "{} pixels".format(d[k]["pixels_changed"]),
                        "s2_mechanism.json"))
        else:
            ok.append(("S2 {} normalisation changes no label".format(k), "0 pixels"))

print("=" * 78)
print("DRAFT NUMBER AUDIT: {}".format(DRAFT))
print("=" * 78)
# The live total is printed below. Documentation deliberately does not duplicate it,
# because adding or withdrawing one check otherwise makes a correct release stale.

if absent:
    print("\n{} check(s) NOT RUN HERE, inputs deliberately not shipped:".format(
        len(absent)))
    for _lab, _why in absent:
        print("  {:36s} {}".format(_lab, _why))

print("\n{} checks passed{}".format(
    len(ok), ", {} not run here".format(len(absent)) if absent else ""))
if bad:
    print("\n{} MISMATCHES -- the draft does not contain these computed values:".format(
        len(bad)))
    for label, val, src in bad:
        print("  {:36s} computed {:>14s}   from {}".format(label, val, src))
else:
    print("\nno mismatches")
if gone:
    print("\n{} withdrawn, by design:".format(len(gone)))
    for label, why in gone:
        print("  {:36s} {}".format(label, why))
if skipped:
    print("\n{} skipped:".format(len(skipped)))
    for label, why in skipped:
        print("  {:36s} {}".format(label, why))
if missing_inputs:
    print("\n{} INPUTS NOT FOUND -- these checks never ran:".format(
        len(missing_inputs)))
    for rel in missing_inputs:
        print("  {}".format(rel))
    print("  looked in: " + ", ".join(str(r) for r in ROOTS))

print()
fail = []
if bad:
    fail.append("{} mismatches".format(len(bad)))
if skipped:
    fail.append("{} skipped".format(len(skipped)))
if missing_inputs:
    fail.append("{} inputs not found".format(len(missing_inputs)))
if len(ok) < MIN_CHECKS:
    fail.append("only {} checks ran, expected at least {}".format(len(ok), MIN_CHECKS))
if fail:
    print("VERIFICATION FAILED: " + "; ".join(fail))
    print("A run that checks nothing is not a run that passed.")
    sys.exit(1)
print("VERIFIED: {} stored-value tokens checked against exact source inputs and found in {}"
      .format(len(ok), DRAFT))
sys.exit(0)
