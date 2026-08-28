"""Fetch the Sen1Floods11 hand-labelled subset: the cross-domain replication.

Why this dataset and not another. It ships, for the SAME 446 chips, both an
expert label (LabelHand) and an algorithmically generated one (S1OtsuLabelHand),
and the algorithmic generator is a structural clone of the sea-ice one we audited:

  sea ice   Otsu + two min-max normalisations, per 128x128 crop, on the V channel
  floods    Otsu on a histogram built from high-variance 1 km cells, applied to a
            focal-mean-smoothed VH band

Both are single-channel Otsu with a data-adaptive threshold whose value depends on
which sub-windows were selected, applied to a normalised version of the same
channel the model reads. Both are therefore closed loops when the model input is
the channel the label came from.

Two things make it worth more than a replication.

First, the dataset publishes the per-event threshold the generator actually used,
in Sen1Floods11_Metadata.geojson. Recovering it by grid search is an EXTERNAL
validation of the method that recovered t_water=30 in 17 of 17 sea-ice folds --
there we could only check the recovered constant against itself.

Second, the expert labels let us compute something sea ice could not support at
all, because sea ice has no reference:

  collapse = mIoU(model trained on algorithmic labels, scored against expert)
           - mIoU(model trained on expert labels,      scored against expert)

which asks directly how much of a deep model's apparent advantage survives contact
with a human.

Caveat to carry into the design: the hand labels were produced by analysts
CORRECTING a Sentinel-2 index classification, not by annotating from scratch. So
the reference arm is only open-loop if the model reads Sentinel-1. Feed S1, not S2.
"""
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import os
_ROOT = Path(os.environ.get("SEAICE_ROOT",
                            Path(__file__).resolve().parents[1]))

DEST = (_ROOT / "data/sen1floods11")
API = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"
MEDIA = "https://storage.googleapis.com/sen1floods11/"
DIRS = ["LabelHand", "S1OtsuLabelHand", "S1Hand"]
PREFIX = "v1.1/data/flood_events/HandLabeled/"


def listing(prefix):
    out, tok = [], None
    while True:
        u = API + "?prefix=" + urllib.parse.quote(prefix) + "&maxResults=1000"
        if tok:
            u += "&pageToken=" + tok
        d = json.load(urllib.request.urlopen(u, timeout=120))
        out += [(i["name"], int(i["size"])) for i in d.get("items", [])
                if i["name"].endswith(".tif")]
        tok = d.get("nextPageToken")
        if not tok:
            break
    return out


def grab(args):
    name, size = args
    dest = DEST / name.split("HandLabeled/")[-1]
    if dest.exists() and dest.stat().st_size == size:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MEDIA + urllib.parse.quote(name), dest)
    return size


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    total = 0
    for d in DIRS:
        files = listing(PREFIX + d + "/")
        print("{:20s} {:4d} files".format(d, len(files)), flush=True)
        with ThreadPoolExecutor(max_workers=16) as ex:
            got = sum(ex.map(grab, files))
        total += got
        print("{:20s} fetched {:.1f} MB".format("", got / 1e6), flush=True)

    # the published per-event thresholds, which are the external validation
    for extra in ["v1.1/Sen1Floods11_Metadata.geojson",
                  "v1.1/splits/flood_handlabeled/flood_train_data.csv",
                  "v1.1/splits/flood_handlabeled/flood_valid_data.csv",
                  "v1.1/splits/flood_handlabeled/flood_test_data.csv",
                  "v1.1/splits/flood_handlabeled/flood_bolivia_data.csv"]:
        dest = DEST / Path(extra).name
        try:
            urllib.request.urlretrieve(MEDIA + urllib.parse.quote(extra), dest)
            print("got {}  {:,} bytes".format(dest.name, dest.stat().st_size))
        except Exception as e:
            print("MISSING {}: {}".format(extra, e))

    print("\ntotal {:.1f} MB -> {}".format(total / 1e6, DEST))
    mf = DEST / "Sen1Floods11_Metadata.geojson"
    if mf.exists():
        g = json.loads(mf.read_text())
        print("\nthe generator constants the dataset publishes:")
        for f in g.get("features", []):
            p = f.get("properties", {})
            k = [x for x in p if "thresh" in x.lower()]
            if k:
                print("  {:14s} {}".format(
                    str(p.get("location", p.get("Location", "?"))),
                    "  ".join("{}={}".format(x, p[x]) for x in k)))


if __name__ == "__main__":
    main()
