r"""Does kappa survive changing the encoder?

Every primary-table and primary-figure kappa comes from a U-Net with a ResNet-18
encoder. Section 3.3 explains the floor in terms of zero padding and truncated
receptive fields, which are properties of convolutions. So the obvious attack is that
kappa reads the architecture rather than the labels. run_arch.sh reruns both ends of the dial on two other
encoders, changing nothing else; this summarises the result.

Written as a script because the JSON it produces was previously built by an inline
one-off with no file behind it. A headline number in the paper with no script to
regenerate it cannot be checked by anyone, including us: after the estimator was
reweighted, this was the one result that had no way to follow.

Reports the pixel-weighted statistic, matching the rest of the paper.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

import os
W = os.environ.get("SEAICE_ROOT",
                   os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K = "kappa_pixel"


def main():
    per = defaultdict(lambda: defaultdict(dict))     # enc -> arm -> event -> kappa
    omg = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(W, "runs_s1fc", "kappa", "arch_*.json")):
        base = os.path.basename(f)[:-5]
        if any(t in base for t in ("_step", "_thr")):
            continue
        r = json.loads(open(f).read())
        if K not in r:
            raise SystemExit("{} has no {}: rescore before summarising".format(base, K))
        # arch_<encoder>_<arm>_<event>_s<seed>, and the encoder name may contain "-"
        parts = base.split("_")
        seed = parts[-1]
        event = parts[-2]
        arm = parts[-3]
        enc = "_".join(parts[1:-3])
        per[enc][arm][event] = r[K]
        omg[enc][arm][event] = r.get("omega", float("nan"))

    out = {}
    for enc in sorted(per):
        arms = per[enc]
        if "a000" not in arms or "a100" not in arms:
            print("  {}: incomplete, skipping".format(enc))
            continue
        ev = sorted(set(arms["a000"]) & set(arms["a100"]))
        d = np.array([arms["a100"][e] - arms["a000"][e] for e in ev])
        se = d.std(ddof=1) / np.sqrt(len(d))
        out[enc] = {
            "n": len(ev),
            "kappa_a000": float(np.mean([arms["a000"][e] for e in ev])),
            "kappa_a100": float(np.mean([arms["a100"][e] for e in ev])),
            "contrast": float(d.mean()),
            "t": float(d.mean() / se) if se else float("nan"),
            "positive": int((d > 0).sum()),
            "omega_a000": float(np.mean([omg[enc]["a000"][e] for e in ev])),
            "omega_a100": float(np.mean([omg[enc]["a100"][e] for e in ev])),
            "per_event_contrast": {e: float(arms["a100"][e] - arms["a000"][e])
                                   for e in ev},
        }
        print("  {:18s} n={:2d}  a000 {:+.4f}  a100 {:+.4f}  contrast {:+.4f}  "
              "t {:.2f}  positive {}/{}".format(
                  enc, out[enc]["n"], out[enc]["kappa_a000"], out[enc]["kappa_a100"],
                  out[enc]["contrast"], out[enc]["t"], out[enc]["positive"],
                  out[enc]["n"]))

    p = os.path.join(W, "runs", "arch_kappa.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote " + p)


if __name__ == "__main__":
    main()
