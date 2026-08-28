r"""J1: build every labelling arm as an explicit per-crop threshold field.

WHY THIS EXISTS, stated before any of it is trained.

Three things were wrong with the calibration as it stood.

1. alpha = 0 was not the shipped labels. The docstring of s1f_crop_labels.py says
   Sen1Floods11 "ships an Otsu-on-VH label set computed once per chip". It does not:
   the dataset publishes ONE threshold per event, and our own 9x9 recovery in
   section 7 is built on exactly that fact. The alpha = 0 arm was a per-chip Otsu we
   computed ourselves, so neither end of the dial was the published label set and
   "the cost of crop-dependence" was measured against a baseline of our own making.

2. A crop-fitted threshold can also be a WORSE threshold. Per-crop Otsu on a 128 px
   window that contains little or no water invents water. So the accuracy lost
   across the dial is consistent with plain label damage, and the design could not
   separate "the labels became crop-dependent" from "the labels became bad".

3. Nothing checked whether the result was sensitive to where a fixed multiset of
   thresholds was assigned within each chip.

The scramble arm addresses only (3). Within each chip it keeps the exact multiset of
per-crop thresholds and shuffles their positions. The random permutation is not a
derangement and does not exclude overlapping donor crops; every donor also comes
from the same spatially correlated chip. It therefore attenuates, but does not break,
the threshold--input correspondence. The arm is a sensitivity analysis, not a causal
decomposition of label damage.

Arms written (all shape (n_chips, ns, ns), dB, on the 9x9 focal-mean band):

  pub    published per-event threshold, broadcast to every crop   (crop-invariant)
  a000   == pub, the dial's zero                                  (crop-invariant)
  a025 a050 a075   (1-alpha)*pub + alpha*per-crop Otsu
  a100   predominantly per-crop Otsu, with recorded fallbacks
  perm   the a100 threshold multiset, randomly reassigned within each chip
  offc   pub + one N(0, s) draw per chip, s matched to the within-chip spread of
         the per-crop thresholds                                  (chip-varying,
                                                                   crop-invariant)

Determinism: every random draw comes from a seed passed in and is written to disk,
so an arm is a file, not a procedure that has to be rerun the same way twice.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import os
W = Path(os.environ.get("SEAICE_ROOT",
                        Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(W / "src"))
C = W / "data" / "sen1floods11" / "cache"
CR = C / "crop"
OUT = CR / "arms"

from recover_vh import published, published_for, assert_all_published  # noqa: E402

DIAL = {"a000": 0.0, "a025": 0.25, "a050": 0.50, "a075": 0.75, "a100": 1.0}


def main(seed=42):
    OUT.mkdir(parents=True, exist_ok=True)
    ev = np.load(C / "events.npy")
    ev = np.array([str(e).lower() for e in ev])
    th_chip = np.load(CR / "th_chip.npy")
    th_crop = np.load(CR / "th_crop.npy")
    n, ns, _ = th_crop.shape
    crop_meta = json.loads((CR / "meta.json").read_text())

    # --- the published per-event threshold, which is what alpha = 0 must be ------
    pub = published()
    events = sorted(set(ev.tolist()))
    assert_all_published(pub, events)          # stops rather than dropping an event
    th_ev = np.array([published_for(pub, e) for e in ev], np.float64)
    assert np.isfinite(th_ev).all(), "an event resolved to a non-finite threshold"
    print("published thresholds: {} events, {:.2f} to {:.2f} dB".format(
        len(events), th_ev.min(), th_ev.max()))

    pubf = np.broadcast_to(th_ev[:, None, None], th_crop.shape).copy()

    # how far the per-crop thresholds move inside a chip: the scale the offset arm
    # has to match so that "crop-varying" and "chip-varying" are comparable
    within = th_crop.reshape(n, -1)
    s_within = float(np.nanmean(np.nanstd(within, axis=1)))
    print("within-chip sd of per-crop thresholds: {:.3f} dB".format(s_within))

    rng = np.random.default_rng(seed)
    # Construct the zero endpoint by identity, not by arithmetic.  Some chips have
    # no finite crop threshold at all; IEEE 754 makes 0 * NaN remain NaN, so the
    # algebraically equivalent expression below would corrupt a000 and make the
    # pub == a000 invariant fail.  Nonzero dial arms intentionally retain those
    # NaNs, and every trainer/scorer excludes the same unusable chips.
    arms = {"pub": pubf, "a000": pubf.copy()}
    for name, a in DIAL.items():
        if a == 0.0:
            continue
        arms[name] = (1.0 - a) * pubf + a * th_crop

    # --- scramble: same thresholds, randomly reassigned positions ----------------
    perm = th_crop.reshape(n, -1).copy()
    orders = np.empty((n, perm.shape[1]), dtype=np.int16)
    for i in range(n):
        orders[i] = rng.permutation(perm.shape[1])
        perm[i] = perm[i][orders[i]]
    perm = perm.reshape(th_crop.shape)
    # Count moved positions from the index map. NaN != NaN, so comparing threshold
    # values would incorrectly call every unresolved position moved.
    moved = float((orders != np.arange(orders.shape[1])[None, :]).mean())
    print("permutation arm: {:.1%} of crops received a different threshold".format(moved))
    # The multiset per chip must be identical, or this is not a permutation.
    # equal_nan because chips whose histogram was degenerate carry NaN across the
    # whole row; the trainer drops those chips, and NaN != NaN would fail the check
    # for a reason that has nothing to do with the permutation.
    assert np.allclose(np.sort(perm.reshape(n, -1), axis=1),
                       np.sort(th_crop.reshape(n, -1), axis=1),
                       equal_nan=True), \
        "permutation changed a chip's threshold multiset"
    # Geometry diagnostics make the control's remaining coupling explicit. Crops
    # are 128 px on a stride-32 ns-by-ns grid.
    pos_r = np.repeat(np.arange(ns), ns)
    pos_c = np.tile(np.arange(ns), ns)
    dr = np.abs(pos_r[orders] - pos_r[None, :])
    dc = np.abs(pos_c[orders] - pos_c[None, :])
    overlaps = (dr * 32 < 128) & (dc * 32 < 128)
    shared_area = np.maximum(0, 128 - dr * 32) * np.maximum(0, 128 - dc * 32)
    perm_diag = {
        "assignments": int(orders.size),
        "fixed_points": int((orders == np.arange(orders.shape[1])).sum()),
        "overlapping_pairs": int(overlaps.sum()),
        "overlapping_fraction": float(overlaps.mean()),
        "mean_shared_area_fraction": float(shared_area.mean() / (128 * 128)),
        "mapping_sha256_int16_c_order": hashlib.sha256(orders.tobytes()).hexdigest(),
        "population": "all 446 serialized rows, including six excluded chips",
    }
    total_positions = int(th_crop.size)
    crop_fit_failures = int(crop_meta["n_crop_threshold_fallbacks"])
    nan_positions = int(np.isnan(th_crop).sum())
    nan_chips = int(np.isnan(th_crop.reshape(n, -1)).all(axis=1).sum())
    finite_fallbacks = crop_fit_failures - nan_positions
    direct_crop_fits = total_positions - crop_fit_failures
    if min(direct_crop_fits, finite_fallbacks, nan_positions) < 0:
        raise SystemExit("inconsistent crop-threshold accounting")
    if direct_crop_fits + finite_fallbacks + nan_positions != total_positions:
        raise SystemExit("crop-threshold accounting does not sum to the grid")
    usable = np.isfinite(th_chip) & np.isfinite(th_crop).all(axis=(1, 2))
    print("chips with no usable threshold at all (dropped downstream): {}".format(
        nan_chips))
    arms["perm"] = perm

    # --- chip-constant random offset, matched spread ----------------------------
    off = rng.normal(0.0, s_within, size=n)
    arms["offc"] = pubf + off[:, None, None]

    assert np.array_equal(arms["pub"], arms["a000"]), "pub and a000 must be identical"
    meta = {"seed": seed, "s_within_db": s_within, "n_chips": int(n),
            "n_usable_chips": int(usable.sum()),
            "n_crops_per_chip": int(ns * ns), "dial": DIAL,
            "events": events, "scramble_geometry": perm_diag,
            "threshold_accounting": {
                "total_positions": total_positions,
                "direct_crop_fits": direct_crop_fits,
                "finite_chip_fallbacks": finite_fallbacks,
                "unresolved_positions": nan_positions,
                "unusable_chips": nan_chips,
            },
            "summary_population": "common usable chips",
            "arms": {k: {"mean_db": float(np.mean(v[usable])),
                         "within_chip_sd_db": float(np.nanmean(np.nanstd(
                             v[usable].reshape(int(usable.sum()), -1), axis=1)))}
                     for k, v in arms.items()}}

    for k, v in arms.items():
        np.save(OUT / ("th_" + k + ".npy"), v.astype(np.float64))
    np.save(OUT / "th_event.npy", th_ev)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))

    print("\n{:6s} {:>12s} {:>18s}".format("arm", "mean dB", "within-chip sd"))
    for k in ("pub", "a000", "a025", "a050", "a075", "a100", "perm", "offc"):
        m = meta["arms"][k]
        print("{:6s} {:12.3f} {:18.3f}".format(k, m["mean_db"], m["within_chip_sd_db"]))
    print("\nperm and a100 must share a within-chip sd; pub, a000 and offc must be 0")
    print("wrote {}".format(OUT))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 42)
