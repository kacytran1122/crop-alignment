"""Validate the two new matched optical-only sea-ice replication seeds."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from data_fingerprint import code_digest, training_data_digest


HERE = Path(__file__).resolve()
W = Path(os.environ.get("SEAICE_ROOT", HERE.parents[1]))
CACHE = Path(os.environ.get("SEAICE_CACHE", "/dev/shm/seaice_expanded"))
TOKENS = Path(os.environ.get(
    "SEAICE_TOKENS", str(W / "work" / "expansion" / "tokens_subsample.parquet")))
FEATURES = Path(os.environ.get(
    "SEAICE_FEATS", str(W / "work" / "expansion" / "feature_cols.json")))


def load_design():
    for path in (HERE.parents[2] / "DESIGN.json", W / "DESIGN.json",
                 W / "audit_r2" / "DESIGN.json"):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["sea_ice"]
    raise SystemExit("DESIGN.json not found")


DESIGN = load_design()
SEEDS = [int(seed) for seed in DESIGN["optical_replication_seeds"]]


def expected_runs():
    out = {}
    for acq in DESIGN["acquisitions"]:
        for labels in DESIGN["arms"]:
            for seed in SEEDS:
                name = "loao_optical_{}_{}_s{}".format(labels, acq, seed)
                out[name] = {
                    "holdout_acq": acq,
                    "labels": labels,
                    "seed": seed,
                    "photon": "none",
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
        return ["run name is outside the frozen optical-replication design"]
    run = W / "runs" / name
    metrics, checkpoint = run / "test_metrics.json", run / "best.pt"
    if not metrics.exists():
        return ["test_metrics.json missing"]
    if not checkpoint.exists():
        return ["best.pt missing"]
    try:
        record = json.loads(metrics.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return ["test_metrics.json unreadable: {}".format(exc)]
    bad = []
    for key, want in EXPECTED[name].items():
        if record.get(key) != want:
            bad.append("{}={!r}, expected {!r}".format(key, record.get(key), want))
    if bad:
        return bad
    for key, source in (("token_table_sha256", TOKENS),
                        ("feature_schema_sha256", FEATURES)):
        if not source.exists() or record.get(key) != sha256_file(source):
            bad.append("{} does not match {}".format(key, source))
    if bad:
        return bad
    want_data = training_data_digest(
        str(CACHE), str(TOKENS), str(FEATURES), EXPECTED[name]["labels"])
    if record.get("training_data_sha256") != want_data:
        bad.append("training_data_sha256 does not match current arrays")
    want_code = code_digest(
        str(HERE.with_name("train.py")), str(HERE.with_name("data_fingerprint.py")))
    if record.get("training_code_sha256") != want_code:
        bad.append("training_code_sha256 does not match the replication trainer")
    if record.get("checkpoint_sha256") != sha256_file(checkpoint):
        bad.append("checkpoint_sha256 does not bind best.pt")
    else:
        try:
            import torch
            checkpoint_run_id = torch.load(
                checkpoint, map_location="cpu", weights_only=False).get(
                    "training_run_id")
        except Exception as exc:
            bad.append("best.pt cannot be inspected: {}".format(exc))
        else:
            if not record.get("training_run_id") or (
                    checkpoint_run_id != record.get("training_run_id")):
                bad.append("checkpoint and metadata training_run_id differ")
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    if bool(args.run) == bool(args.complete):
        raise SystemExit("choose exactly one of --run or --complete")
    names = [args.run] if args.run else sorted(EXPECTED)
    failures = [(name, why) for name in names for why in problems(name)]
    for name, why in failures[:12]:
        print("INVALID {}: {}".format(name, why))
    if failures:
        print("{} invalid/missing condition(s) across {} expected run(s)".format(
            len(failures), len(names)))
        return 1
    print("validated {} matched optical-only run(s)".format(len(names)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
