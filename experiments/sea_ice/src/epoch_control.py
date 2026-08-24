"""Did a short training budget manufacture the artefact premium?

Compares the premium on the same folds across training budgets, under mosaic
scoring every time so nothing else changes. Also reports where training actually
stopped, because the control is only meaningful if the long runs converged: if
best_epoch is still hitting the cap, the objection stands and the paper must say
so rather than claim it was answered.

WHY THIS FILE CHANGED. It had "ep60" and CAP = 60 written into main(). The ep120
block is now running, and its driver ends by calling mosaic_premium.py, which only
ever reads the "loao" and "dj" prefixes -- so the driver's closing lines print the
TWELVE-epoch premium no matter which budget just finished. Reading that line as
"the ep120 result" would put a number in the paper that belongs to a different
experiment. Budgets are now named on the command line as prefix:cap pairs and
printed in every header, so the output cannot be read as a budget it is not.

  python src/epoch_control.py                      # loao:12 vs ep60:60, as before
  python src/epoch_control.py ep60:60 ep120:120    # all three budgets

The cap is passed in because test_metrics.json does not record it, but it is
checked against the runs rather than trusted: a best_epoch above the stated cap
means the pair is wrong and the script stops instead of reporting.
"""
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
MOS = W / "runs" / "mosaic"
RUNS = W / "runs"

BASE = ("loao", 12)          # the budget every other one is compared against
PATIENCE = 15


def grab(prefix):
    out = {"scene": {}, "original": {}}
    for f in MOS.glob(prefix + "_*.json"):
        r = json.loads(f.read_text())
        if r["seed"] != 42 or r["labels"] not in out:
            continue
        out[r["labels"]][r["acq"]] = r["advantage_nanmean"]
    return out


def stop_epochs(prefix):
    """best epoch, epochs actually run, and minutes, per (labels, acquisition).

    Epochs actually run comes from the length of history.csv, not from
    best_epoch + PATIENCE. They agree when a run early-stops, but only the
    history length is direct evidence, and it is the thing that says whether the
    cap was reached.
    """
    rows = {}
    for d in RUNS.glob(prefix + "_*_s42"):
        f = d / "test_metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        h = d / "history.csv"
        ran = None
        if h.exists():
            ran = sum(1 for _ in h.open()) - 1      # minus the header
        rows[(m["labels"], m["holdout_acq"])] = (m.get("best_epoch"), ran,
                                                 m.get("minutes"))
    return rows


def parse_budgets(argv):
    if not argv:
        return [("ep60", 60)]
    out = []
    for a in argv:
        if ":" not in a:
            raise SystemExit(
                "budgets are given as prefix:cap, for example ep120:120; got " + a)
        p, c = a.rsplit(":", 1)
        out.append((p, int(c)))
    return out


def summarise(v, name, n):
    se = v.std(ddof=1) / np.sqrt(n)
    print("{:14s} premium {:+.4f}  sd {:.4f}  t {:5.2f}  positive {}/{}".format(
        name, v.mean(), v.std(ddof=1), v.mean() / se if se else float("nan"),
        int((v > 0).sum()), n))


def check_cap(prefix, cap, ep):
    """Stop before printing anything if the stated cap contradicts the runs.

    This has to run before the first print. A cap that does not match the runs
    puts a wrong budget in every column header, and a header is exactly what a
    reader takes on trust.
    """
    best = [v[0] for v in ep.values() if v[0] is not None]
    if best and max(best) > cap:
        raise SystemExit(
            "{}: best_epoch {} exceeds the stated cap {}. The prefix:cap pair is "
            "wrong, so nothing below would mean what it says.".format(
                prefix, max(best), cap))


def convergence(prefix, cap, ep):
    """How many runs hit the cap with early stopping never firing.

    A run early-stops after PATIENCE epochs without improvement. So a run whose
    history is exactly `cap` epochs long never got that far: it was still inside
    its patience window when the budget ran out, and was therefore still
    improving. The criterion this replaced asked whether best_epoch >= cap, which
    nothing can satisfy, and so announced that every long run had converged over
    best epochs of 59, 56 and 51.
    """
    best = [v[0] for v in ep.values() if v[0] is not None]
    ran = [v[1] for v in ep.values() if v[1] is not None]
    if not ran:
        print("  no history.csv found for {}, cannot report convergence".format(prefix))
        return
    at_cap = sum(1 for c in ran if c >= cap)
    print("\n{} runs that hit the {}-epoch cap without early stopping: {}/{}".format(
        prefix, cap, at_cap, len(ran)))
    print("  best epochs: " + ", ".join(str(c) for c in sorted(best)))
    if at_cap:
        print("  -> {} of {} were still improving at {} epochs, so {} is closer to".format(
            at_cap, len(ran), cap, cap))
        print("     converged than 12 but is NOT converged. Report this premium as a")
        print("     lower bound on the converged-budget premium, not as its value.")
    else:
        print("  -> every run early-stopped before the cap")


def main():
    budgets = parse_budgets(sys.argv[1:])
    base = grab(BASE[0])

    for prefix, cap in budgets:
        long_ = grab(prefix)
        ep = stop_epochs(prefix)
        check_cap(prefix, cap, ep)          # before the first line of output
        acqs = sorted(set(long_["scene"]) & set(long_["original"])
                      & set(base["scene"]) & set(base["original"]))
        print("=" * 96)
        print("CONVERGED-BUDGET CONTROL: {}:{} against {}:{}".format(
            prefix, cap, BASE[0], BASE[1]))
        print("=" * 96)
        if not acqs:
            print("no folds have a mosaic result under both budgets, skipped\n")
            continue

        print("{:24s} {:>9s} {:>9s} {:>9s}  {:>14s}".format(
            "held-out acquisition", "prem@" + str(BASE[1]), "prem@" + str(cap),
            "delta", "epochs run"))
        pS, pL = [], []
        for a in acqs:
            x = base["original"][a] - base["scene"][a]
            y = long_["original"][a] - long_["scene"][a]
            pS.append(x); pL.append(y)
            rs = ep.get(("scene", a), (None, None, None))[1]
            ro = ep.get(("original", a), (None, None, None))[1]
            print("{:24s} {:+9.4f} {:+9.4f} {:+9.4f}  {:>6s} / {:<6s}".format(
                a, x, y, y - x, str(rs), str(ro)))

        pS, pL = np.array(pS), np.array(pL)
        n = len(pS)
        print()
        summarise(pS, "{} epochs".format(BASE[1]), n)
        summarise(pL, "{} epochs".format(cap), n)

        d = pL - pS
        se = d.std(ddof=1) / np.sqrt(n)
        print("\npaired change {:+.4f}  sd {:.4f}  t {:5.2f}   on {} folds".format(
            d.mean(), d.std(ddof=1), d.mean() / se if se else float("nan"), n))

        # the paper states resolution as a ratio of effect to spread against the
        # 0.679 that n = 17 would need, so print the ratio rather than leaving it
        # to be recomputed by hand from two numbers in different paragraphs
        for nm, v in (("{} ep".format(BASE[1]), pS), ("{} ep".format(cap), pL)):
            print("  d/s at {:7s} {:.3f}".format(nm, v.mean() / v.std(ddof=1)))

        convergence(prefix, cap, ep)
        print()


if __name__ == "__main__":
    main()
