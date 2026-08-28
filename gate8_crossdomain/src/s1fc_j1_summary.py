r"""J1 stage 3: what the rebuilt calibration says.

The dial endpoint and scramble arm carry the same multiset of thresholds inside each
chip. The scramble randomly reassigns their positions, but allows fixed points,
overlapping donor crops and same-chip predictability. It is therefore a descriptive
sensitivity arm rather than a link-breaking control. This summary reports means and
standard deviations only: leave-one-event-out models share training data, so folds
are not independent replicates for inference.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

import os
W = os.environ.get("SEAICE_ROOT",
                   os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = ["a000", "a025", "a050", "a075", "a100", "perm", "offc"]
# The design, stated once, so absence is detectable. Every downstream test takes
# an intersection somewhere, and an intersection cannot tell you what is missing
# from both sides of it.
EVENTS = ["Bolivia", "Ghana", "India", "Mekong", "Nigeria", "Pakistan",
          "Paraguay", "Somalia", "Spain", "Sri-Lanka", "USA"]
SEEDS = {"s42", "s7", "s123"}   # the run name carries the s, and so does the key
NICE = {"a000": "alpha 0 (published)", "a025": "alpha 0.25", "a050": "alpha 0.50",
        "a075": "alpha 0.75", "a100": "alpha 1 (predominantly crop-fitted)",
        "perm": "permuted thresholds", "offc": "chip-constant offset"}


def load():
    # kap is the PIXEL-weighted statistic, which is what the paper reports: each
    # source pixel in the artefact set counts once. kcr is the crop-read weighting,
    # which counts a pixel once per covering crop and so weights densely covered
    # pixels more. Both are carried so the paper can show the choice does not
    # decide anything.
    kap = defaultdict(dict)      # arm -> (event, seed) -> kappa, pixel-weighted
    kcr = defaultdict(dict)      # arm -> (event, seed) -> kappa, crop-read
    omg = defaultdict(dict)
    dmg = defaultdict(dict)      # arm -> (event, seed) -> mIoU vs expert
    # j1_*.json also matches the stride-64 grid controls j1_*_step2.json and any
    # _thr variant. Parsing those gives seed="step2" and event="Spain_s42", a
    # phantom event under the same arm that inflates every mean and paired summary.
    # The summary predates those files, so it was right by luck, not by filter.
    for f in glob.glob(os.path.join(W, "runs_s1fc", "kappa", "j1_*.json")):
        if any(t in os.path.basename(f) for t in ("_step", "_thr", "_foldnorm")):
            continue
        r = json.loads(open(f).read())
        nm = os.path.basename(f)[:-5]
        parts = nm.split("_")
        arm, seed = parts[1], parts[-1]
        ev = "_".join(parts[2:-1])
        # a run without kappa_pixel predates the weighting fix; refuse rather than
        # silently fall back, because a mixed set would average two estimands
        if "kappa_pixel" not in r:
            raise SystemExit("{} has no kappa_pixel: rescore before summarising"
                             .format(os.path.basename(f)))
        # a cell written twice under two names overwrites in silence, which is the
        # "two seeds in one slot" fault this project has now had three times
        if (ev, seed) in kap[arm]:
            raise SystemExit(
                "arm {} event {} seed {} is scored more than once; a duplicate "
                "moves the mean without changing the count".format(arm, ev, seed))
        kap[arm][(ev, seed)] = r["kappa_pixel"]
        kcr[arm][(ev, seed)] = r["kappa"]
        omg[arm][(ev, seed)] = r.get("omega", float("nan"))
    # accuracy against the EXPERT labels comes from the damage scorer, not from
    # the trainer: the trainer only ever scores a model against its own label set,
    # which is the quantity this whole paper says not to trust
    for f in glob.glob(os.path.join(W, "runs_s1fc", "damage", "j1_*.json")):
        r = json.loads(open(f).read())
        nm = os.path.basename(f)[:-5]
        parts = nm.split("_")
        arm, seed = parts[1], parts[-1]
        ev = "_".join(parts[2:-1])
        if (ev, seed) in dmg[arm]:
            raise SystemExit(
                "damage arm {} event {} seed {} is scored more than once".format(
                    arm, ev, seed))
        dmg[arm][(ev, seed)] = float(r["miou_vs_expert"])
    return kap, kcr, omg, dmg


def by_event(d):
    """Average over seeds within a held-out event (a descriptive unit)."""
    per = defaultdict(list)
    for (ev, _), v in d.items():
        per[ev].append(v)
    return {e: float(np.mean(v)) for e, v in per.items()}


def paired(a, b):
    """Event-level paired difference, only on events present in both."""
    ev = sorted(set(a) & set(b))
    return np.array([a[e] - b[e] for e in ev]), ev


def line(name, d):
    n = len(d)
    if n < 2:
        return "{:24s} n={} too few".format(name, n)
    return "{:24s} n={:2d}  mean {:+.4f}  descriptive sd {:.4f}".format(
        name, n, d.mean(), d.std(ddof=1))


def main():
    kap, kcr, omg, dmg = load()

    # Exact product, before any intersection: 11 events x 7 arms x 3 seeds.
    expected = {(a, e, s) for a in ARMS for e in EVENTS for s in SEEDS}
    got = {(a, e, s) for a in kap for (e, s) in kap[a]}
    missing, extra = sorted(expected - got), sorted(got - expected)
    if missing or extra:
        for m in missing[:6]:
            print("  MISSING CELL:  arm {} event {} seed {}".format(*m))
        for x in extra[:6]:
            print("  UNEXPECTED CELL: arm {} event {} seed {}".format(*x))
        raise SystemExit(
            "the scored set is not {} arms x {} events x {} seeds: {} missing, "
            "{} unexpected".format(len(ARMS), len(EVENTS), len(SEEDS),
                                   len(missing), len(extra)))
    damage_got = {(a, e, s) for a in dmg for (e, s) in dmg[a]}
    damage_missing = sorted(expected - damage_got)
    damage_extra = sorted(damage_got - expected)
    if damage_missing or damage_extra:
        for m in damage_missing[:6]:
            print("  MISSING DAMAGE CELL: arm {} event {} seed {}".format(*m))
        for x in damage_extra[:6]:
            print("  UNEXPECTED DAMAGE CELL: arm {} event {} seed {}".format(*x))
        raise SystemExit(
            "the damage set is not {} arms x {} events x {} seeds: {} missing, "
            "{} unexpected".format(len(ARMS), len(EVENTS), len(SEEDS),
                                   len(damage_missing), len(damage_extra)))
    print("  complete: {} arms x {} events x {} seeds = {} cells, none missing, "
          "none unexpected".format(len(ARMS), len(EVENTS), len(SEEDS),
                                   len(expected)))
    have = [a for a in ARMS if kap.get(a)]
    print("=" * 78)
    print("J1: THE REBUILT FLOOD CALIBRATION")
    print("=" * 78)
    print("arms scored:", ", ".join(have))
    ke = {a: by_event(kap[a]) for a in have}
    kc = {a: by_event(kcr[a]) for a in have}
    de = {a: by_event(dmg[a]) for a in have if dmg.get(a)}
    oe = {a: by_event(omg[a]) for a in have}

    print("\n1. kappa and accuracy by arm, averaged over seeds within each event")
    print("{:24s} {:>7s} {:>10s} {:>11s} {:>9s} {:>10s}".format(
        "arm", "events", "kappa", "(crop-read)", "Omega", "mIoU vs expert"))
    for a in have:
        k = np.array(list(ke[a].values()))
        kk = np.array(list(kc[a].values()))
        o = np.array(list(oe[a].values()))
        dd = np.array(list(de[a].values())) if a in de else np.array([np.nan])
        print("{:24s} {:7d} {:+10.4f} {:+11.4f} {:9.4f} {:10.4f}".format(
            NICE[a], len(k), k.mean(), kk.mean(), np.nanmean(o), np.nanmean(dd)))
    print("  kappa is weighted uniformly over source pixels; the second column"
          " weights")
    print("  each pixel once per covering crop. Reported together because the"
          " choice")
    print("  changes the magnitude and not the ordering or any conclusion.")

    print("\n" + "=" * 78)
    print("2. SENSITIVITY TO ONE WITHIN-CHIP THRESHOLD SCRAMBLE")
    print("=" * 78)
    if "perm" in ke and "a100" in ke and "a000" in ke:
        d1, _ = paired(ke["a100"], ke["a000"])
        d2, _ = paired(ke["perm"], ke["a000"])
        d3, _ = paired(ke["a100"], ke["perm"])
        print(line("a100 - a000  (dial)", d1))
        print(line("perm - a000", d2))
        print(line("a100 - perm", d3))
        share = d2.mean() / d1.mean() if d1.mean() else float("nan")
        print("\n  this scramble retains {:.0%} of the dial's excess "
              "alignment".format(share))
        print("  this is one unconstrained reassignment, not a causal decomposition")
        if de.get("perm") and de.get("a000") and de.get("a100"):
            c1, _ = paired(de["a100"], de["a000"])
            c2, _ = paired(de["perm"], de["a000"])
            print("\n  accuracy cost against expert labels:")
            print(line("  a100 - a000", c1))
            print(line("  perm - a000", c2))
            print("\n  retained loss and reduced alignment are descriptive properties")
            print("  of this mapping; overlap and same-chip dependence remain.")
    else:
        print("  scramble or dial arms missing; cannot summarize the contrast")

    print("\n" + "=" * 78)
    print("3. CROP-INVARIANT ARMS MUST SIT AT THE FLOOR")
    print("=" * 78)
    for a in ("a000", "offc"):
        if a in ke:
            k = np.array(list(ke[a].values()))
            print("{:24s} kappa {:+.4f}  (crop-invariant labels: expect the "
                  "architectural floor)".format(NICE[a], k.mean()))

    out = {"arms": have,
           "kappa_by_event": {a: ke[a] for a in have},
           "kappa_cropread_by_event": {a: kc[a] for a in have},
           "damage_by_event": {a: de[a] for a in de},
           "omega_by_event": {a: oe[a] for a in have}}
    p = os.path.join(W, "runs_s1fc", "j1_summary.json")
    open(p, "w").write(json.dumps(out, indent=1))
    print("\nwrote", p)


if __name__ == "__main__":
    main()
