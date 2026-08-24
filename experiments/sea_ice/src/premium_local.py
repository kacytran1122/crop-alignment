"""
The artefact premium, tested as a paired interaction.

The paper's most novel claim is that a deep model's advantage over a two-parameter
threshold is inflated when the labels carry a crop-normalisation artefact. So far
that has only been shown as a RATIO of two separately-computed means (0.1224 /
0.0736 = 1.66), which has no interval and no test.

The right object is a difference-in-differences, computed within acquisition so
that scene difficulty cancels twice over:

    premium(a) = [U-Net(a,orig) - threshold(a,orig)]
               - [U-Net(a,scene) - threshold(a,scene)]

Positive premium means depth is paid more on the artefact-laden labels than on the
repaired ones, on that same held-out ground. Testing it paired across the 17
acquisitions is the whole claim, and it is a stronger test than the ratio because
every nuisance that is common to an acquisition drops out.

Bootstrap over acquisitions as well, since n=17 and the t interval leans on
normality of the fold differences.
"""
import json
from pathlib import Path

import numpy as np

R = Path("gate6_scale/results")
thr = json.loads((R / "threshold_expanded.json").read_text())


def unet(labels):
    out = {}
    for f in sorted(R.glob(f"loao_{labels}_*.json")):
        m = json.loads(f.read_text())
        out[m.get("holdout_acq") or f.stem.split("_", 2)[2]] = m["miou"]
    return out


us, uo = unet("scene"), unet("original")
acqs = sorted(set(thr) & set(us) & set(uo))

adv_s = np.array([us[a] - thr[a]["scene"]["test_miou"] for a in acqs])
adv_o = np.array([uo[a] - thr[a]["original"]["test_miou"] for a in acqs])
prem = adv_o - adv_s
n = len(acqs)

print(f"{'acquisition':24s} {'adv_scene':>10s} {'adv_orig':>10s} {'premium':>9s}")
for a, s, o, p in zip(acqs, adv_s, adv_o, prem):
    print(f"{a:24s} {s:+10.4f} {o:+10.4f} {p:+9.4f}")

se = prem.std(ddof=1) / np.sqrt(n)
print(f"\n{'':24s} {adv_s.mean():+10.4f} {adv_o.mean():+10.4f} {prem.mean():+9.4f}")
print(f"\npaired premium   mean {prem.mean():+.4f}   sd {prem.std(ddof=1):.4f}   "
      f"se {se:.4f}   t {prem.mean()/se:.2f}")
print(f"t 95% CI         [{prem.mean()-2.12*se:+.4f}, {prem.mean()+2.12*se:+.4f}]")
print(f"folds positive   {int((prem>0).sum())}/{n}")

rng = np.random.RandomState(0)
bs = np.array([prem[rng.randint(0, n, n)].mean() for _ in range(20000)])
print(f"bootstrap 95% CI [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]"
      f"   P(premium<=0) = {(bs<=0).mean():.4f}")

# sign test, which assumes nothing about the shape of the distribution
k = int((prem > 0).sum())
from math import comb
p2 = 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
print(f"sign test        {k}/{n} positive, two-sided p = {min(p2,1.0):.5f}")

# the ratio, with an interval, for readers who want it in the paper's old units
rb = np.array([(adv_o[i].mean() if False else adv_o[j].mean() / adv_s[j].mean())
               for j in [rng.randint(0, n, n) for _ in range(20000)]])
print(f"\nratio adv_orig/adv_scene = {adv_o.mean()/adv_s.mean():.2f}x   "
      f"bootstrap 95% CI [{np.percentile(rb,2.5):.2f}x, {np.percentile(rb,97.5):.2f}x]")

# Is the premium just the collapsing fold?
drop = np.argsort([us[a] for a in acqs])[0]
keep = [i for i in range(n) if i != drop]
pk = prem[keep]
sek = pk.std(ddof=1) / np.sqrt(len(pk))
print(f"\ndropping the 99.99%-thick fold ({acqs[drop]}):")
print(f"  premium {pk.mean():+.4f}  t {pk.mean()/sek:.2f}  "
      f"ratio {adv_o[keep].mean()/adv_s[keep].mean():.2f}x")
