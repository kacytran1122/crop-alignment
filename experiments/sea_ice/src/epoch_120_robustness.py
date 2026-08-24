r"""Adversarial checks on the 120-epoch premium before it goes in the paper.

The 120-epoch number is the first result in this project that makes our own claim
look GOOD: the premium rises to +0.0796 while the spread falls to 0.0747, so it
resolves at n = 17 where every earlier budget did not. That is exactly the direction
in which we should be least willing to trust ourselves, and the paper's own thesis is
that effects like this one are usually artefacts of small samples.

So, before it is written down:

  1. Leave-one-out. One fold (20191124191133_08990) carries a delta of +0.2185, more
     than three times the next largest. If the result rests on it, this is
     KNOWN_ISSUES item 2 ("the 0.1164 is three outliers") happening again.
  2. Distribution-free. A t-test on 17 paired folds assumes something about the tails
     that 17 points cannot establish. Sign test and Wilcoxon make weaker assumptions.
  3. Required n, computed the same way the paper computes it everywhere else.
  4. The trend across all three budgets, since the paper reports all three.
"""
import json
import glob
from pathlib import Path

import numpy as np
from scipy import stats as st

W = Path(__file__).resolve().parents[1]
MOS = W / "runs" / "mosaic"


def grab(prefix):
    out = {"scene": {}, "original": {}}
    for f in MOS.glob(prefix + "_*.json"):
        r = json.loads(f.read_text())
        if r["seed"] != 42 or r["labels"] not in out:
            continue
        out[r["labels"]][r["acq"]] = r["advantage_nanmean"]
    return out


budgets = [("loao", 12), ("ep60", 60), ("ep120", 120)]
data = {}
for p, cap in budgets:
    d = grab(p)
    acqs = sorted(set(d["scene"]) & set(d["original"]))
    data[cap] = (acqs, np.array([d["original"][a] - d["scene"][a] for a in acqs]))

common = sorted(set.intersection(*[set(data[c][0]) for _, c in budgets]))
print("folds common to all three budgets:", len(common))

prem = {}
for _, cap in budgets:
    acqs, v = data[cap]
    idx = [acqs.index(a) for a in common]
    prem[cap] = v[idx]

print("\n" + "=" * 78)
print("1. THE THREE BUDGETS ON THE SAME FOLDS")
print("=" * 78)
print("{:>8s} {:>10s} {:>9s} {:>7s} {:>8s} {:>9s} {:>10s}".format(
    "budget", "premium", "sd", "t", "d/s", "pos", "N for 80%"))
for _, cap in budgets:
    v = prem[cap]
    n = len(v)
    t = v.mean() / (v.std(ddof=1) / np.sqrt(n))
    need = (2.80 * v.std(ddof=1) / v.mean()) ** 2 if v.mean() > 0 else float("nan")
    print("{:>8d} {:>+10.4f} {:>9.4f} {:>7.2f} {:>8.3f} {:>6d}/{:<2d} {:>10.0f}".format(
        cap, v.mean(), v.std(ddof=1), t, v.mean() / v.std(ddof=1),
        int((v > 0).sum()), n, need))
print("\nresolution at n=17 needs d/s > {:.3f}".format(2.80 / np.sqrt(17)))

v = prem[120]
n = len(v)
print("\n" + "=" * 78)
print("2. LEAVE-ONE-OUT AT 120 EPOCHS  (does one fold carry it?)")
print("=" * 78)
order = np.argsort(-v)
print("largest folds: " + ", ".join(
    "{} {:+.4f}".format(common[i][:8], v[i]) for i in order[:3]))
worst_t, worst_i = None, None
for i in range(n):
    x = np.delete(v, i)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    if worst_t is None or t < worst_t:
        worst_t, worst_i = t, i
print("\nmin t over all leave-one-out subsets: {:.2f}  (dropping {})".format(
    worst_t, common[worst_i]))
x = np.delete(v, order[0])
print("dropping the single largest fold: premium {:+.4f}  sd {:.4f}  t {:.2f}  "
      "d/s {:.3f}".format(x.mean(), x.std(ddof=1), x.mean() / (x.std(ddof=1) / np.sqrt(len(x))),
                          x.mean() / x.std(ddof=1)))
x2 = np.delete(v, order[:2])
print("dropping the two largest:         premium {:+.4f}  sd {:.4f}  t {:.2f}".format(
    x2.mean(), x2.std(ddof=1), x2.mean() / (x2.std(ddof=1) / np.sqrt(len(x2)))))

print("\n" + "=" * 78)
print("3. DISTRIBUTION-FREE, 120 EPOCHS")
print("=" * 78)
k = int((v > 0).sum())
p_sign = st.binomtest(k, n, 0.5, alternative="two-sided").pvalue
w = st.wilcoxon(v, alternative="two-sided")
print("sign test      {}/{} positive, p = {:.5f}".format(k, n, p_sign))
print("Wilcoxon       W = {:.1f}, p = {:.5f}".format(w.statistic, w.pvalue))
print("paired t       t = {:.2f}, p = {:.6f}".format(
    v.mean() / (v.std(ddof=1) / np.sqrt(n)),
    st.ttest_1samp(v, 0).pvalue))
med = np.median(v)
lo, hi = np.sort(v)[3], np.sort(v)[-4]
print("median {:+.4f}   order-statistic interval [{:+.4f}, {:+.4f}]".format(med, lo, hi))

print("\n" + "=" * 78)
print("4. PAIRED CHANGE 12 -> 120, AND 60 -> 120")
print("=" * 78)
for a, b in ((12, 120), (60, 120)):
    d = prem[b] - prem[a]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print("{:>3d} -> {:<3d}  change {:+.4f}  sd {:.4f}  t {:.2f}  positive {}/{}".format(
        a, b, d.mean(), d.std(ddof=1), t, int((d > 0).sum()), len(d)))
