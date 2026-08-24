"""Time offsets between the Sentinel-2 and ICESat-2 halves of each pair.

The photon branch is reported as inert: five conditions statistically equivalent to
optical-only. A reviewer has an obvious alternative explanation -- the co-location
is loose, the ice drifted between the two overpasses, and the descriptors are
attached to the wrong pixels. Under that reading the null says nothing about
altimetry and everything about our matching.

The literature makes this concrete rather than hypothetical. Farooq et al. (2022,
Climate Dynamics 59:227-241) measure western Ross Sea drift at 12.03 +- 6.22 cm/s
with subregion maxima near 70 cm/s, and Kwok et al. (2017) put Southern Ocean drift
about 50% above Arctic. At 10 m pixels:

    12 cm/s  -> 43 px per hour
    20 cm/s  -> 72 px per hour
    70 cm/s  -> 252 px per hour

So the offset distribution is not a footnote, it is the difference between a null
about altimetry and a null about our own matching. Compute it, report it per pair,
and state the implied displacement so the reader can judge rather than trust.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(__file__).resolve().parents[1]
SUB = W / "work" / "expansion" / "tokens_subsample.parquet"
DRIFT = [("Ross mean, Farooq 2022", 0.1203),
         ("Ross subregion max", 0.2024),
         ("Ross extreme", 0.70)]


def main():
    t = pd.read_parquet(SUB)
    pairs = t[["acq_id", "s2_datetime", "tile"]].drop_duplicates()

    rows = []
    for acq, grp in pairs.groupby("acq_id"):
        is2 = datetime.strptime(acq.split("_")[0], "%Y%m%d%H%M%S")
        for s2s in sorted(grp["s2_datetime"].unique()):
            s2 = datetime.strptime(str(s2s), "%Y%m%dT%H%M%S")
            dt = (is2 - s2).total_seconds()
            rows.append({"acq_id": acq, "s2": str(s2s), "offset_s": dt,
                         "n_tiles": int((grp["s2_datetime"] == s2s).sum())})

    df = pd.DataFrame(rows)
    print("{:24s} {:>16s} {:>9s} {:>7s}   displacement at 12 cm/s".format(
        "acquisition", "S2 time", "offset", "tiles"))
    for r in df.sort_values("offset_s").itertuples():
        mins = r.offset_s / 60.0
        px = abs(r.offset_s) * 0.1203 / 10.0
        print("{:24s} {:>16s} {:+7.1f}m {:7d}   {:6.1f} px".format(
            r.acq_id, r.s2, mins, r.n_tiles, px))

    o = df["offset_s"].abs().to_numpy()
    print("\n  n pairs {}   |offset| min {:.1f} m   median {:.1f} m   max {:.1f} m".format(
        len(o), o.min() / 60, np.median(o) / 60, o.max() / 60))
    print("  pairs within 15 minutes: {}/{}".format(int((o <= 900).sum()), len(o)))
    print("  pairs within 30 minutes: {}/{}".format(int((o <= 1800).sum()), len(o)))

    print("\n  implied displacement in 10 m pixels, at the median offset "
          "({:.1f} min):".format(np.median(o) / 60))
    for name, v in DRIFT:
        print("    {:26s} {:6.1f} px".format(name, np.median(o) * v / 10.0))
    print("\n  and at the worst pair ({:.1f} min):".format(o.max() / 60))
    for name, v in DRIFT:
        print("    {:26s} {:6.1f} px".format(name, o.max() * v / 10.0))

    df.to_json(W / "work" / "coloc_offsets.json", orient="records", indent=1)
    print("\n  wrote {}".format(W / "work" / "coloc_offsets.json"))


if __name__ == "__main__":
    main()
