"""Add a tile-disjoint option to leave-one-acquisition-out.

The existing LOAO comment asserts that whole acquisitions "differ in date and
orbit, so nothing is shared across the boundary". That is false: the tile grid is
fixed, so the same ground appears on many dates. Three tiles (T02CNA, T03CWP,
T03CWQ) sit in both train and test, and T03CWP is trained on 20191122 and tested
on 20191126 -- four days apart over the same water.

Whether that matters is an empirical question, so make it one. --disjoint-tiles
additionally removes from train and val every patch whose tile appears in the
held-out acquisition, which turns LOAO from a temporal holdout into a
temporal-and-spatial one. Reporting both is the honest move: the gap between them
IS the leak, measured rather than argued.
"""
import re
from pathlib import Path

p = Path(__file__).resolve().with_name("train.py")
s = p.read_text()

if "disjoint_tiles" in s:
    raise SystemExit("already patched")

s = s.replace(
    'def load_all(dev, feat_cols, label_set="scene", split_col="split", holdout_acq=None,\n'
    '             n_train_acq=None):',
    'def load_all(dev, feat_cols, label_set="scene", split_col="split", holdout_acq=None,\n'
    '             n_train_acq=None, disjoint_tiles=False):')

old = '''        # leave-one-acquisition-out: test = holdout, val = next two in sorted
        # order, train = the rest. Whole acquisitions differ in date and orbit,
        # so nothing is shared across the boundary.'''
new = '''        # leave-one-acquisition-out: test = holdout, val = next two in sorted
        # order, train = the rest. Acquisitions differ in date and orbit, but the
        # tile grid is FIXED, so by default the same ground can appear on both
        # sides of the boundary on different dates. --disjoint-tiles removes that
        # overlap so the two designs can be compared and the leak measured.'''
assert old in s
s = s.replace(old, new)

old = '''        col = np.where(tok["acq_id"].to_numpy() == holdout_acq, "test",
                       np.where(tok["acq_id"].isin(val_acq).to_numpy(), "val", "train"))
        idx = {s: torch.from_numpy(np.where(col == s)[0]).to(dev)
               for s in ("train", "val", "test")}'''
new = '''        col = np.where(tok["acq_id"].to_numpy() == holdout_acq, "test",
                       np.where(tok["acq_id"].isin(val_acq).to_numpy(), "val", "train"))
        if disjoint_tiles:
            held = set(tok.loc[tok["acq_id"] == holdout_acq, "tile"].unique())
            shared = tok["tile"].isin(held).to_numpy() & (col != "test")
            print(json.dumps({"disjoint_tiles": True, "held_tiles": sorted(held),
                              "dropped_patches": int(shared.sum()),
                              "kept_train_val": int((~shared & (col != "test")).sum())}),
                  flush=True)
            col = np.where(shared, "unused", col)
        idx = {s: torch.from_numpy(np.where(col == s)[0]).to(dev)
               for s in ("train", "val", "test")}'''
assert old in s
s = s.replace(old, new)

s = s.replace(
    '''    ap.add_argument("--split-col", default="split", choices=["split", "split_rev"],''',
    '''    ap.add_argument("--disjoint-tiles", action="store_true",
                    help="LOAO: also drop train/val patches on tiles present in the holdout")
    ap.add_argument("--split-col", default="split", choices=["split", "split_rev"],''')

s = s.replace(
    'args.labels, args.split_col, args.holdout_acq,',
    'args.labels, args.split_col, args.holdout_acq,\n                                   ')
old = 'imgs, labs, X, idx, _ = load_all(dev, feat_cols, args.labels, args.split_col, args.holdout_acq,\n                                   '
assert old in s, "call-site anchor moved"
s = s.replace(old, old)

# extend the metrics record so a result file says which design produced it
s = s.replace('"holdout_acq": args.holdout_acq,',
              '"holdout_acq": args.holdout_acq, "disjoint_tiles": args.disjoint_tiles,')

p.write_text(s)
print("patched", p)
