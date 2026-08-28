"""Fail-closed validation for cached sea-ice primary/sensitivity checkpoints.

Training launchers used to skip any directory containing ``test_metrics.json``.
That lets a stale photon-enabled checkpoint occupy an optical-only name. This gate
derives every expected run from DESIGN.json and checks both the checkpoint and the
configuration recorded by training before a run is reused or scored.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from data_fingerprint import (code_digest, scoring_data_digest,
                              training_data_digest)


HERE = Path(__file__).resolve()
W = Path(os.environ.get("SEAICE_ROOT", HERE.parents[1]))
CACHE = Path(os.environ.get("SEAICE_CACHE", "/dev/shm/seaice_expanded"))
TOKENS = Path(os.environ.get(
    "SEAICE_TOKENS", str(W / "work" / "expansion" / "tokens_subsample.parquet")))
FEATURES = Path(os.environ.get(
    "SEAICE_FEATS", str(W / "work" / "expansion" / "feature_cols.json")))
SCENE = W / "work" / "scene_all"


def load_design():
    for path in (HERE.parents[2] / "DESIGN.json", W / "DESIGN.json",
                 W.parent / "DESIGN.json"):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["sea_ice"]
    raise SystemExit("DESIGN.json not found; cached runs will not be trusted")


DESIGN = load_design()
MODE = {int(k): v for k, v in DESIGN["input_mode_by_seed"].items()}


def expected_runs():
    out = {}
    for acq in DESIGN["acquisitions"]:
        for labels in DESIGN["arms"]:
            for seed in DESIGN["seeds"]:
                mode = MODE[seed]
                name = "loao_{}_{}_s{}".format(labels, acq, seed)
                out[name] = {
                    "holdout_acq": acq,
                    "labels": labels,
                    "seed": seed,
                    "photon": "none" if mode == "optical-only" else "true",
                    "disjoint_tiles": False,
                    "fusion": "deep",
                    "encoder": "resnet18",
                    "loss": "focal",
                    "split_col": "split",
                    "n_train_acq": None,
                    "train_acquisition_scope": "all",
                    "optimizer": "AdamW",
                    "scheduler": "CosineAnnealingLR",
                    "epochs": 12,
                    "batch": 128,
                    "lr": 1e-4,
                    "wd": 1e-4,
                    "patience": 5,
                    "amp": "fp16",
                }
    return out


EXPECTED = expected_runs()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def problems(name):
    if name not in EXPECTED:
        return ["run name is outside the frozen design"]
    run = W / "runs" / name
    metrics = run / "test_metrics.json"
    checkpoint = run / "best.pt"
    bad = []
    if not metrics.exists():
        bad.append("test_metrics.json missing")
        return bad
    if not checkpoint.exists():
        bad.append("best.pt missing")
    try:
        record = json.loads(metrics.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        bad.append("test_metrics.json unreadable: {}".format(exc))
        return bad
    for key, want in EXPECTED[name].items():
        if record.get(key) != want:
            bad.append("{}={!r}, expected {!r}".format(key, record.get(key), want))
    # Do the cheap configuration checks before hashing multi-gigabyte arrays.
    if bad:
        return bad
    for key, source in (("token_table_sha256", TOKENS),
                        ("feature_schema_sha256", FEATURES)):
        if not source.exists():
            bad.append("{} source missing: {}".format(key, source))
        elif record.get(key) != sha256_file(source):
            bad.append("{} does not match {}".format(key, source))
    if bad:
        return bad
    want_data = training_data_digest(
        str(CACHE), str(TOKENS), str(FEATURES), EXPECTED[name]["labels"])
    if record.get("training_data_sha256") != want_data:
        bad.append("training_data_sha256 does not match current image/label arrays")
    want_code = code_digest(
        str(HERE.with_name("train.py")), str(HERE.with_name("data_fingerprint.py")))
    if record.get("training_code_sha256") != want_code:
        bad.append("training_code_sha256 does not match the current trainer")
    if record.get("checkpoint_sha256") != sha256_file(checkpoint):
        bad.append("checkpoint_sha256 does not bind best.pt to its training metadata")
    else:
        try:
            import torch
            checkpoint_run_id = torch.load(
                checkpoint, map_location="cpu", weights_only=False).get(
                    "training_run_id")
        except Exception as exc:
            bad.append("best.pt cannot be inspected for training_run_id: {}".format(exc))
        else:
            if not record.get("training_run_id") or (
                    checkpoint_run_id != record.get("training_run_id")):
                bad.append("best.pt training_run_id does not match test_metrics.json")
    return bad


def expected_score_path(name):
    suffix = "_vsorig" if EXPECTED[name]["labels"] == "scene" else ""
    return W / "runs" / "kappa" / (name + suffix + ".json")


def score_problems(path):
    path = Path(path)
    if not path.is_absolute():
        path = W / path
    if not path.exists():
        return [(str(path), "score JSON missing")]
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return [(str(path), "score JSON unreadable: {}".format(exc))]
    name = record.get("run")
    bad = [(name or str(path), why) for why in problems(name)]
    if name not in EXPECTED or bad:
        return bad
    want_path = expected_score_path(name)
    if path.resolve() != want_path.resolve():
        bad.append((name, "score path is {}, expected {}".format(path, want_path)))
    expected_record = {
        "acq": EXPECTED[name]["holdout_acq"],
        "labels": EXPECTED[name]["labels"],
        "eval_labels": "original",
        "seed": EXPECTED[name]["seed"],
        "photon_mode": EXPECTED[name]["photon"],
        "input_mode": MODE[EXPECTED[name]["seed"]],
        "checkpoint_sha256": sha256_file(W / "runs" / name / "best.pt"),
        "training_metadata_sha256": sha256_file(
            W / "runs" / name / "test_metrics.json"),
        "token_table_sha256": sha256_file(TOKENS),
        "feature_schema_sha256": sha256_file(FEATURES),
        "scoring_data_sha256": scoring_data_digest(
            str(CACHE), str(TOKENS), str(FEATURES), str(SCENE), "original"),
        "scoring_code_sha256": code_digest(
            str(HERE.with_name("crop_alignment.py")),
            str(HERE.with_name("train.py")),
            str(HERE.with_name("data_fingerprint.py"))),
    }
    for key, want in expected_record.items():
        if record.get(key) != want:
            bad.append((name, "score {}={!r}, expected {!r}".format(
                key, record.get(key), want)))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="validate one cached run")
    ap.add_argument("--complete", action="store_true",
                    help="require the exact 17 x 2 x 2 checkpoint product")
    ap.add_argument("--score", help="validate one cached scorer JSON")
    ap.add_argument("--scores-complete", action="store_true",
                    help="require all 68 score files and bind each to its model")
    args = ap.parse_args()
    modes = sum((bool(args.run), bool(args.complete), bool(args.score),
                 bool(args.scores_complete)))
    if modes != 1:
        raise SystemExit("choose exactly one validation mode")
    if args.score:
        failures = score_problems(args.score)
        names = [args.score]
    elif args.scores_complete:
        names = [str(expected_score_path(name)) for name in sorted(EXPECTED)]
        failures = []
        for path in names:
            failures.extend(score_problems(path))
    else:
        names = [args.run] if args.run else sorted(EXPECTED)
        failures = []
        for name in names:
            for why in problems(name):
                failures.append((name, why))
    for name, why in failures[:12]:
        print("INVALID {}: {}".format(name, why))
    if failures:
        print("{} invalid/missing condition(s) across {} expected run(s)".format(
            len(failures), len(names)))
        return 1
    noun = "score(s)" if args.score or args.scores_complete else "cached run(s)"
    print("validated {} {} against DESIGN.json and source hashes"
          .format(len(names), noun))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
