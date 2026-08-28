"""Backfill authenticated input-mode provenance into 68 sea-ice score records.

The original scorer wrote the seed but not the training run's ``photon`` field.
The release now carries the exact bytes and SHA-256 of all 68 historical
``test_metrics.json`` files. This script derives the annotation from those bytes,
not from a hardcoded seed-to-mode assertion, and refuses any cell outside the
frozen design.
"""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESIGN = json.loads((ROOT / "DESIGN.json").read_text(encoding="utf-8"))["sea_ice"]
RUNS = ROOT / "gate7_rebuild" / "results" / "per_run_kappa"
METADATA = ROOT / "gate7_rebuild" / "results" / "seaice_training_metadata.json"
EXPECTED = {(a, arm, seed) for a in DESIGN["acquisitions"]
            for arm in DESIGN["arms"] for seed in DESIGN["seeds"]}


def authenticated_metadata():
    payload = json.loads(METADATA.read_text(encoding="utf-8"))
    out = {}
    for item in payload.get("records", []):
        raw = item["test_metrics_text"].encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != item["source_sha256"]:
            raise SystemExit("training metadata hash mismatch for {}".format(
                item.get("run")))
        rec = json.loads(item["test_metrics_text"])
        cell = (rec.get("holdout_acq"), rec.get("labels"), rec.get("seed"))
        if cell in out:
            raise SystemExit("duplicate training metadata cell {!r}".format(cell))
        out[cell] = rec
    if set(out) != EXPECTED:
        raise SystemExit("training metadata is not the exact frozen 17 x 2 x 2 design")
    return out


def main():
    metadata = authenticated_metadata()
    rows = []
    for path in sorted(RUNS.glob("loao_*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("eval_labels", rec.get("labels")) != "original":
            continue
        cell = (rec.get("acq"), rec.get("labels"), rec.get("seed"))
        rows.append((path, rec, cell))
    got = [cell for _, _, cell in rows]
    if len(got) != len(set(got)) or set(got) != EXPECTED:
        raise SystemExit("retained scores are not the exact frozen 17 x 2 x 2 design")

    changed = 0
    for path, rec, cell in rows:
        photon_mode = metadata[cell].get("photon")
        if photon_mode not in ("none", "true"):
            raise SystemExit("unsupported historical photon mode {!r}".format(
                photon_mode))
        input_mode = ("optical-only" if photon_mode == "none" else
                      "optical+ICESat-2 photon")
        rec.update({
            "photon_mode": photon_mode,
            "input_mode": input_mode,
            "optical_normalisation": "fixed ImageNet RGB mean/std",
            "photon_normalisation": (
                "not used" if photon_mode == "none" else
                "fold-local non-test-acquisition mean/std"),
        })
        rendered = json.dumps(rec, indent=1) + "\n"
        if path.read_text(encoding="utf-8") != rendered:
            path.write_text(rendered, encoding="utf-8")
            changed += 1
    print("authenticated {} exact design cells; {} annotation(s) changed".format(
        len(rows), changed))


if __name__ == "__main__":
    main()
