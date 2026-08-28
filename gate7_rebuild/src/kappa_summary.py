"""Descriptive crop alignment across dependent held-out folds.

The algebraic identity applies to a crop-invariant predictor, not to a trained
model, and these leave-one-acquisition-out folds share tiles, geography and training
data. This summary therefore reports magnitudes, descriptive spread and sign counts,
without fold-level tests, intervals or power calculations. Seed 42 is the optical-
only primary arm. Seed 7 used the photon branch and is an input-mode sensitivity
check, not a repeat seed to average into the primary estimate.
"""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

import os
W = Path(os.environ.get("SEAICE_ROOT",
                        Path(__file__).resolve().parents[1]))
K = W / "runs" / "kappa"
LIVE_SCORES = K.exists()
if not K.exists():
    K = W / "results" / "per_run_kappa"
SUMMARY_DIR = W / ("runs" if (W / "runs").exists() else "results")


def authenticated_release(root):
    """Require the packaged legacy fallback to belong to an intact release."""
    manifest_path, id_path = root / "MANIFEST.json", root / "ARCHIVE_ID.txt"
    if not manifest_path.exists() or not id_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = manifest["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    seen = set()
    for row in rows:
        rel, want = row.get("path"), row.get("sha256")
        if not rel or rel in seen:
            return False
        seen.add(rel)
        path = root / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != want:
            return False
    digest = hashlib.sha256()
    for row in sorted((r for r in rows if r["path"] != "ARCHIVE_ID.txt"),
                      key=lambda r: r["path"]):
        digest.update((row["path"] + " " + row["sha256"] + "\n").encode("utf-8"))
    got = digest.hexdigest()
    return (manifest.get("archive_content_id") == got and
            got in id_path.read_text(encoding="utf-8"))

# The separately documented summary command must not bypass run_kappa.sh's cache
# binding. Live scores are accepted only when all 68 are tied to the current
# checkpoints, training metadata, arrays, masks and code. Packaged historical
# outputs predate those hashes; verify.py checks their frozen design and values,
# and the supplement explicitly discloses that limitation.
if LIVE_SCORES:
    from validate_primary_runs import EXPECTED, expected_score_path, score_problems
    binding_failures = []
    for run_name in sorted(EXPECTED):
        binding_failures.extend(score_problems(expected_score_path(run_name)))
    if binding_failures:
        for item, why in binding_failures[:12]:
            print("INVALID SCORE {}: {}".format(item, why))
        raise SystemExit("refusing to summarize {} unbound/stale score condition(s)"
                         .format(len(binding_failures)))
    print("validated 68 live scores against checkpoint, data and code hashes")
else:
    if not authenticated_release(W.parent):
        raise SystemExit("live runs/kappa is absent and this tree is not an "
                         "authenticated anonymous release; refusing legacy fallback")
    print("using disclosed legacy packaged scores; archive verify.py checks their "
          "exact design and stored values, not missing historical run hashes")

# The design, read from a frozen file rather than stated here or, worse, inferred.
#
# The previous version fixed N_ACQ = 17 and then built the expected set from the
# acquisitions it FOUND, so it pinned the count and not the identities: swap one
# acquisition for another and seventeen is still seventeen. DESIGN.json carries the
# seventeen identifiers themselves, and is compared for set equality below.
def _design():
    here = Path(__file__).resolve()
    for base in (Path(os.environ["SEAICE_ROOT"]) if os.environ.get("SEAICE_ROOT")
                 else here.parents[2], here.parents[2], here.parents[1]):
        f = Path(base) / "DESIGN.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))["sea_ice"]
    raise SystemExit("DESIGN.json not found; the completeness gate has no design "
                     "to check against and will not guess one")


DESIGN = _design()
ACQS = set(DESIGN["acquisitions"])
N_ACQ = len(ACQS)
SEEDS = set(DESIGN["seeds"])
ARMS = tuple(DESIGN["arms"])
PRIMARY_SEED = int(DESIGN["primary_seed"])
INPUT_MODE = {int(k): v for k, v in DESIGN["input_mode_by_seed"].items()}
if set(INPUT_MODE) != SEEDS or PRIMARY_SEED not in SEEDS:
    raise SystemExit("DESIGN.json must map every scored seed to an input mode and "
                     "name one of them as primary")


def load():
    """Retain one record per acquisition, arm and seed.

    The two seeds must not be averaged: their training records show that seed 42 is
    optical-only while seed 7 uses the photon branch. The exact product is still
    required so the latter can serve as a separately labelled sensitivity check.
    """
    treat, ctrl = collections.defaultdict(list), collections.defaultdict(list)
    for f in K.glob("*.json"):
        r = json.loads(f.read_text())
        # eval_labels was added to crop_alignment.py partway through the work, so
        # files written before that change do not carry the key. A run scored on
        # its own label set has eval_labels == labels by definition, and that is
        # the right default: .get(...) with no default silently dropped a fold.
        if r.get("eval_labels", r["labels"]) != "original":
            continue
        if r.get("labels") not in ARMS:
            raise SystemExit("{} has unexpected training-label arm {!r}; expected {}"
                             .format(f.name, r.get("labels"), ARMS))
        seed = r.get("seed")
        want_mode = INPUT_MODE.get(seed)
        want_photon = "none" if want_mode == "optical-only" else "true"
        want_photon_norm = ("not used" if want_photon == "none" else
                            "fold-local non-test-acquisition mean/std")
        provenance = {
            "input_mode": want_mode,
            "photon_mode": want_photon,
            "optical_normalisation": "fixed ImageNet RGB mean/std",
            "photon_normalisation": want_photon_norm,
        }
        wrong = {k: (r.get(k), v) for k, v in provenance.items() if r.get(k) != v}
        if seed not in SEEDS or wrong:
            raise SystemExit("{} has seed/input provenance outside DESIGN.json: {}"
                             .format(f.name, wrong or {"seed": seed}))
        (ctrl if r["labels"] == "scene" else treat)[r["acq"]].append(r)
    return treat, ctrl


def cell(recs, key, seed):
    """One acquisition/arm/seed value, required to exist exactly once."""
    vals = [r.get(key) for r in recs if r.get("seed") == seed]
    if len(vals) != 1 or vals[0] is None:
        raise SystemExit("expected exactly one non-null {} value at seed {}"
                         .format(key, seed))
    return float(vals[0])


def arrays_for_seed(treat, ctrl, acqs, seed):
    """All claim-bearing per-fold arrays for one fixed input mode."""
    return {
        "input_mode": INPUT_MODE[seed],
        "kappa_noisy": [cell(treat[a], "kappa_pixel", seed) for a in acqs],
        "kappa_clean": [cell(ctrl[a], "kappa_pixel", seed) for a in acqs],
        "kappa_cropread_noisy": [cell(treat[a], "kappa", seed) for a in acqs],
        "kappa_cropread_clean": [cell(ctrl[a], "kappa", seed) for a in acqs],
        "omega_noisy": [cell(treat[a], "omega", seed) for a in acqs],
        "omega_clean": [cell(ctrl[a], "omega", seed) for a in acqs],
        "frac_artefact": [cell(treat[a], "frac_artefact", seed) for a in acqs],
    }


def describe(v, name):
    n = len(v)
    if n < 3:
        print("  {:30s} n={} too few".format(name, n))
        return
    k = int((v > 0).sum())
    print("  {:30s} n={:2d}  mean {:+.4f}  descriptive sd {:.4f}  "
          "median {:+.4f}  positive {}/{}".format(
              name, n, v.mean(), v.std(ddof=1), np.median(v), k, n))


def main():
    treat, ctrl = load()

    # THE CANONICAL PRODUCT, asserted before anything is intersected.
    #
    # This used to take set(treat) & set(ctrl) and then check seeds on whatever
    # survived, which is not a completeness gate at all: an acquisition missing from
    # BOTH arms never enters the intersection, so sixteen folds pass in silence, and
    # one missing from a single arm is discarded by the intersection itself. The
    # previous commit claimed this gate was exact. It was not. The expected set is
    # now stated up front and compared for equality in both directions, so a missing
    # cell and an unexpected cell both fail.
    found = set(treat) | set(ctrl)
    if found != ACQS:
        raise SystemExit(
            "acquisition identities do not match the design: {} missing, {} "
            "unexpected. missing={} unexpected={}".format(
                len(ACQS - found), len(found - ACQS),
                sorted(ACQS - found)[:4], sorted(found - ACQS)[:4]))
    acqs = sorted(ACQS)
    expected = {(a, arm, s) for a in acqs for arm in ("treated", "control")
                for s in SEEDS}
    # counted as a LIST first, so a cell scored twice is caught. Collapsing straight
    # into a set hid duplicates: the same record twice looked exactly like once.
    got_list = [(a, "treated", r.get("seed")) for a in treat for r in treat[a]]
    got_list += [(a, "control", r.get("seed")) for a in ctrl for r in ctrl[a]]
    dupes = sorted({c for c in got_list if got_list.count(c) > 1})
    if dupes:
        for d in dupes[:6]:
            print("  DUPLICATE CELL: acquisition {} arm {} seed {}".format(*d))
        raise SystemExit(
            "{} cell(s) scored more than once; a duplicate moves the mean without "
            "changing the count".format(len(dupes)))
    got = set(got_list)
    missing, extra = sorted(expected - got), sorted(got - expected)
    if missing or extra:
        for m in missing[:6]:
            print("  MISSING CELL:  acquisition {} arm {} seed {}".format(*m))
        for e in extra[:6]:
            print("  UNEXPECTED CELL: acquisition {} arm {} seed {}".format(*e))
        raise SystemExit(
            "the scored set is not {} acquisitions x 2 arms x {}: {} missing, "
            "{} unexpected".format(N_ACQ, sorted(SEEDS), len(missing), len(extra)))

    print("=" * 92)
    print("CROP ALIGNMENT: does the model reproduce the label of the crop it reads?")
    print("  kappa = P(pred = own crop label | A) - P(pred = other crop label | A)")
    print("  structural null: kappa == 0 exactly for any crop-invariant predictor")
    print("=" * 92)
    print("  {:24s} {:>8s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
        "held-out acquisition", "A frac", "k_noisy", "k_clean", "diff", "Omega_r"))

    print("  complete: {} acquisitions x 2 arms x seeds {} = {} cells, "
          "none missing, none unexpected".format(
              N_ACQ, sorted(SEEDS), N_ACQ * 2 * len(SEEDS)))
    print("  primary: seed {} ({})".format(PRIMARY_SEED, INPUT_MODE[PRIMARY_SEED]))

    by_seed = {str(seed): arrays_for_seed(treat, ctrl, acqs, seed)
               for seed in sorted(SEEDS)}
    primary = by_seed[str(PRIMARY_SEED)]
    kt = np.array(primary["kappa_noisy"])
    kc = np.array(primary["kappa_clean"])
    ktc = np.array(primary["kappa_cropread_noisy"])
    kcc = np.array(primary["kappa_cropread_clean"])
    om_t = np.array(primary["omega_noisy"])
    om_c = np.array(primary["omega_clean"])
    frac = np.array(primary["frac_artefact"])
    for i, a in enumerate(acqs):
        print("  {:24s} {:8.2%} {:+9.4f} {:+9.4f} {:+9.4f} {:9.2f}".format(
            a, frac[i], kt[i], kc[i], kt[i] - kc[i],
            om_t[i] / om_c[i] if om_c[i] else float("nan")))

    print()
    describe(kt, "kappa, crop-noisy training")
    describe(kc, "kappa, crop-invariant (control)")
    describe(kt - kc, "difference (treated - control)")
    print()
    describe(ktc, "kappa, crop-read weighted")
    describe(kcc, "kappa, crop-read control")
    describe(ktc - kcc, "difference, crop-read weighted")
    print()
    describe(om_t, "Omega, crop-noisy training")
    describe(om_c, "Omega, crop-invariant (control)")
    print("\n  Omega ratio: mean {:.2f}x".format(float(np.mean(om_t / om_c))))
    print("  Omega is 0 exactly for the two-parameter threshold, by construction.")

    print("  artefact set A: {:.2%} of covered pixels on average, "
          "{:,} instances per fold".format(
              frac.mean(),
              int(np.mean([cell(treat[a], "n_instances", PRIMARY_SEED)
                           for a in acqs]))))
    print("\n  input-mode sensitivity (not averaged into the primary result):")
    for seed in sorted(SEEDS):
        row = by_seed[str(seed)]
        a = np.array(row["kappa_noisy"]) - np.array(row["kappa_clean"])
        print("    seed {} {:28s} mean {:+.4f}, positive {}/{}".format(
            seed, row["input_mode"], a.mean(), int((a > 0).sum()), len(a)))

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / "kappa_summary.json").write_text(json.dumps(
        {"acqs": acqs,
         "scored_seeds": sorted(SEEDS),
         "primary_seed": PRIMARY_SEED,
         "primary_input_mode": INPUT_MODE[PRIMARY_SEED],
         "input_mode_by_seed": {str(k): v for k, v in sorted(INPUT_MODE.items())},
         "kappa_noisy": kt.tolist(), "kappa_clean": kc.tolist(),
         "kappa_cropread_noisy": ktc.tolist(),
         "kappa_cropread_clean": kcc.tolist(),
         "omega_noisy": om_t.tolist(), "omega_clean": om_c.tolist(),
         "frac_artefact": frac.tolist(),
         "by_seed": by_seed}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
