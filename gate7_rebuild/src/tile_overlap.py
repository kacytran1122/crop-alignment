"""How much ground is shared across the leave-one-acquisition-out boundary?

If removing the overlap also removes most of the training data, then a
tile-disjoint result is confounded with training-set size and cannot be compared
against the standard one. Check that before spending GPU on it.
"""
import numpy as np
import pandas as pd

t = pd.read_parquet("work/expansion/tokens_subsample.parquet")
acqs = sorted(t.acq_id.unique())
print("{:,} patches, {} tiles, {} acquisitions\n".format(
    len(t), t.tile.nunique(), len(acqs)))
print("{:24s} {:>6s} {:>11s} {:>11s} {:>7s}".format(
    "held-out acquisition", "tiles", "train+val", "disjoint", "kept%"))

rows = []
for a in acqs:
    held = set(t.loc[t.acq_id == a, "tile"].unique())
    rest = t[t.acq_id != a]
    keep = rest[~rest.tile.isin(held)]
    pct = 100.0 * len(keep) / max(len(rest), 1)
    rows.append((a, len(held), len(rest), len(keep), pct))
    print("{:24s} {:6d} {:11,d} {:11,d} {:6.1f}%".format(
        a, len(held), len(rest), len(keep), pct))

k = np.array([r[4] for r in rows])
print("\nkept fraction: min {:.1f}%  median {:.1f}%  max {:.1f}%".format(
    k.min(), np.median(k), k.max()))
print("acquisitions with <50% of training data left: {}/{}".format(
    int((k < 50).sum()), len(k)))

tpa = t.groupby("acq_id").tile.nunique()
apt = t.groupby("tile").acq_id.nunique()
print("\ntiles per acquisition   min {} median {} max {}".format(
    tpa.min(), int(tpa.median()), tpa.max()))
print("acquisitions per tile   min {} median {} max {}".format(
    apt.min(), int(apt.median()), apt.max()))
print("\ntiles appearing on more than one acquisition: {}/{}".format(
    int((apt > 1).sum()), len(apt)))
