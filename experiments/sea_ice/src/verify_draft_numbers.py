"""Check every headline number in the draft against the file it came from.

Numbers in a paper go stale silently. Three have already done it here: the mosaic
required-n stayed at 238 after it became 197, the published fusion gain stayed at 65
after it became 70, and figure 1 carried both plus two more while rendering
perfectly. None of those crashed anything, and none would have been caught by
reading the draft.

So this recomputes each claim from its source and checks the formatted result
actually appears in the draft text. It cannot verify prose, and it deliberately does
not try: it verifies the numbers, which is the part that drifts.

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

W = Path(__file__).resolve().parents[1]
DRAFT = Path(sys.argv[1] if len(sys.argv) > 1 else "PAPER_DRAFT_V3.md")
TEXT = DRAFT.read_text(encoding="utf-8")

ok, bad, skipped = [], [], []


def load(rel):
    f = W / rel
    return json.loads(f.read_text()) if f.exists() else None


def check(label, value, fmt="{:+.4f}", source=""):
    """Assert the formatted value appears in the draft as a standalone number.

    Plain substring matching is useless for short integers: searching for "8" from
    a required-n of 8 matches nearly any page and the check always passes. So the
    search is anchored on digit boundaries -- the match must not have another digit
    immediately before or after it, and must not sit inside a longer decimal.
    """
    s = fmt.format(value)
    # the draft uses a unicode minus in tables and an ascii one in code blocks
    variants = {s, s.replace("-", "−"), s.lstrip("+")}
    for v in variants:
        pat = r"(?<![\d.])" + re.escape(v) + r"(?![\d])"
        if re.search(pat, TEXT):
            ok.append((label, s))
            return
    bad.append((label, s, source))


def skip(label, why):
    skipped.append((label, why))


# ---- crop alignment, sea ice ---------------------------------------------
d = load("runs/kappa_summary.json")
if d:
    kn, kc = np.array(d["kappa_noisy"]), np.array(d["kappa_clean"])
    diff = kn - kc
    check("kappa noisy mean", kn.mean(), source="kappa_summary.json")
    check("kappa control mean", kc.mean(), source="kappa_summary.json")
    check("kappa difference", diff.mean(), source="kappa_summary.json")
    check("kappa difference t", diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))),
          "{:.2f}", "kappa_summary.json")
    check("artefact fraction", 100 * np.mean(d["frac_artefact"]), "{:.2f}",
          "kappa_summary.json")
    check("Omega control", np.mean(d["omega_clean"]), "{:.3f}", "kappa_summary.json")
else:
    skip("crop alignment", "kappa_summary.json missing")

# ---- calibration ----------------------------------------------------------
d = load("runs_s1fc/summary.json")
if d:
    al = d["alphas"]
    for a in al:
        check("kappa at alpha={}".format(a), np.mean(d["kappa"][str(a)]),
              source="s1fc/summary.json")
        check("Omega at alpha={}".format(a), np.mean(d["omega"][str(a)]), "{:.4f}",
              "s1fc/summary.json")
    lo, hi = np.array(d["kappa"][str(al[0])]), np.array(d["kappa"][str(al[-1])])
    con = hi - lo
    check("calibration contrast", con.mean(), source="s1fc/summary.json")
    check("calibration contrast t", con.mean() / (con.std(ddof=1) / np.sqrt(len(con))),
          "{:.2f}", "s1fc/summary.json")
    check("calibration artefact %", 100 * d["frac_artefact"], "{:.2f}",
          "s1fc/summary.json")
    check("calibration slope", d["slope"], source="s1fc/summary.json")
else:
    skip("calibration", "s1fc/summary.json missing")

# ---- cost against human truth --------------------------------------------
d = load("runs_s1fc/damage_summary.json")
dk = load("runs_s1fc/summary.json")
if d and dk:
    for a in d["alphas"]:
        check("mIoU vs expert at alpha={}".format(a),
              np.mean(d["miou_vs_expert"][str(a)]), "{:.4f}", "damage_summary.json")
    check("cost mean", d["cost_mean"], source="damage_summary.json")
    check("cost t", d["cost_t"], "{:.2f}", "damage_summary.json")
    ev = [e for e in dk["events"] if e in d["events"]]
    ik = [dk["events"].index(e) for e in ev]
    idd = [d["events"].index(e) for e in ev]
    K = np.array([[dk["kappa"][str(a)][j] for j in ik] for a in d["alphas"]])
    M = np.array([[d["miou_vs_expert"][str(a)][j] for j in idd] for a in d["alphas"]])
    kc_, mc_ = (K - K.mean(0)).ravel(), (M - M.mean(0)).ravel()
    check("within-event r", float(np.corrcoef(kc_, mc_)[0, 1]), "{:.3f}",
          "damage + kappa")
    rs = [float(np.corrcoef(K[i], M[i])[0, 1]) for i in range(len(d["alphas"]))]
    for a, r in zip(d["alphas"], rs):
        check("fixed-alpha r at {}".format(a), r, "{:.3f}", "damage + kappa")
    check("fixed-alpha mean r", float(np.mean(rs)), "{:.3f}", "damage + kappa")
else:
    skip("cost", "damage_summary.json missing")

# ---- estimator verification ----------------------------------------------
thr = sorted((W / "runs_s1fc" / "kappa").glob("*_thr.json"))
if thr:
    worst = max(abs(json.loads(f.read_text())["kappa"]) for f in thr)
    if worst < 1.4e-15:
        ok.append(("estimator |kappa| under 1.4e-15",
                   "max {:.1e} over {} events".format(worst, len(thr))))
    else:
        bad.append(("estimator |kappa| under 1.4e-15",
                    "max {:.1e}".format(worst), "runs_s1fc/kappa/*_thr.json"))
else:
    skip("estimator check", "no *_thr.json")

# ---- premium and power ----------------------------------------------------
d = load("runs/power.json")
if d:
    check("patch premium", d["patch"]["d"], source="power.json")
    check("mosaic premium", d["mosaic_vote"]["d"], source="power.json")
    check("mosaic draw premium", d["mosaic_draw"]["d"], source="power.json")
    for k, name in (("patch", "patch"), ("mosaic_vote", "mosaic vote"),
                    ("mosaic_draw", "mosaic draw")):
        check("required n, " + name, d[k]["n_power80"], "{:.0f}", "power.json")
else:
    skip("power", "power.json missing")

# ---- medians and their distribution-free intervals ------------------------
# Added after a read-through found a stale median (+0.0307) the audit had not
# covered: it checked means and t statistics and never the order statistics beside
# them. A number audit only protects the numbers it was told about.
d = load("runs/mosaic_premium.json")
if d and "loao_s42" in d and "nanmean" in d["loao_s42"]:
    from math import comb
    r = d["loao_s42"]["nanmean"]
    v = np.sort(np.array(r["adv_original"]) - np.array(r["adv_scene"]))
    m = len(v)
    check("mosaic premium median", float(np.median(v)), source="mosaic_premium.json")
    kbest = None
    for k in range(0, m // 2 + 1):
        cov = 1 - 2 * sum(comb(m, i) for i in range(0, k)) / 2 ** m
        if cov >= 0.95:
            kbest = (k, cov)
    if kbest:
        k, cov = kbest
        check("median CI coverage", 100 * cov, "{:.1f}", "mosaic_premium.json")
        check("median CI low", float(v[k]), source="mosaic_premium.json")
        check("median CI high", float(v[m - 1 - k]), source="mosaic_premium.json")
else:
    skip("median", "mosaic_premium.json missing")

# ---- epoch budget ---------------------------------------------------------
d = load("runs/epoch_control.json")
if d:
    p12, p60 = np.array(d["premium_12"]), np.array(d["premium_60"])
    check("premium at 12 epochs", p12.mean(), source="epoch_control.json")
    check("premium at 60 epochs", p60.mean(), source="epoch_control.json")
    check("sd at 12 epochs", p12.std(ddof=1), "{:.4f}", "epoch_control.json")
    check("sd at 60 epochs", p60.std(ddof=1), "{:.4f}", "epoch_control.json")
    dd = p60 - p12
    check("paired change", dd.mean(), source="epoch_control.json")
    check("paired change t", dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))),
          "{:.2f}", "epoch_control.json")
    if "{} of {}".format(d["runs_at_cap"], d["n_long_runs"]) in TEXT or \
       "{} of 34".format(d["runs_at_cap"]) in TEXT:
        ok.append(("runs at cap", "{}/{}".format(d["runs_at_cap"], d["n_long_runs"])))
    else:
        bad.append(("runs at cap", "{} of {}".format(d["runs_at_cap"],
                    d["n_long_runs"]), "epoch_control.json"))
else:
    skip("epoch budget", "epoch_control.json missing")

# ---- transfer collapse ----------------------------------------------------
d = load("runs_s1f/mechanism.json")
if d:
    col = np.array(d["collapse"])
    check("transfer collapse", col.mean(), source="mechanism.json")
    check("transfer collapse t", col.mean() / (col.std(ddof=1) / np.sqrt(len(col))),
          "{:.2f}", "mechanism.json")
    check("mechanism slope t", d["slope_t"], "{:.2f}", "mechanism.json")
    check("mechanism correlation",
          float(np.corrcoef(d["agreement"], d["collapse"])[0, 1]), "{:.3f}",
          "mechanism.json")
else:
    skip("transfer collapse", "mechanism.json missing")

# ---- dataset scale --------------------------------------------------------
cm = load("data/sen1floods11/cache/meta.json")
if cm:
    if "{}".format(cm["n"]) in TEXT:
        ok.append(("flood chips", str(cm["n"])))
    else:
        bad.append(("flood chips", str(cm["n"]), "cache/meta.json"))

# ---- report ---------------------------------------------------------------
print("=" * 78)
print("DRAFT NUMBER AUDIT: {}".format(DRAFT))
print("=" * 78)
print("\n{} checks passed".format(len(ok)))
if bad:
    print("\n{} MISMATCHES -- the draft does not contain these computed values:".format(
        len(bad)))
    for label, val, src in bad:
        print("  {:36s} computed {:>14s}   from {}".format(label, val, src))
else:
    print("\nno mismatches")
if skipped:
    print("\n{} skipped:".format(len(skipped)))
    for label, why in skipped:
        print("  {:36s} {}".format(label, why))
print()
sys.exit(1 if bad else 0)
