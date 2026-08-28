"""Export the exact retained sea-ice training metadata without cluster paths.

The historical scorer outputs did not record whether the photon branch was used.
The source ``test_metrics.json`` files do. This exporter preserves their exact
bytes and SHA-256 in one anonymous JSON payload, after requiring the frozen
17-acquisition x 2-label-arm x 2-seed product and checking the identifying fields
against the run directory. Keeping the bytes, rather than only a derived table,
lets the release verify the provenance statement it relies on.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    design = json.loads(args.design.read_text(encoding="utf-8"))["sea_ice"]
    modes = {int(k): value for k, value in design["input_mode_by_seed"].items()}
    expected = {
        (acq, arm, int(seed))
        for acq in design["acquisitions"]
        for arm in design["arms"]
        for seed in design["seeds"]
    }
    records = []
    seen = set()
    for acq, arm, seed in sorted(expected):
        run = "loao_{}_{}_s{}".format(arm, acq, seed)
        path = args.root / "runs" / run / "test_metrics.json"
        raw = path.read_bytes()
        meta = json.loads(raw.decode("utf-8-sig"))
        cell = (meta.get("holdout_acq"), meta.get("labels"), meta.get("seed"))
        if cell != (acq, arm, seed):
            raise SystemExit("{} identifies {!r}, expected {!r}".format(
                path, cell, (acq, arm, seed)))
        want_photon = "none" if modes[seed] == "optical-only" else "true"
        if meta.get("photon") != want_photon:
            raise SystemExit("{} photon={!r}, expected {!r}".format(
                path, meta.get("photon"), want_photon))
        if cell in seen:
            raise SystemExit("duplicate metadata cell {!r}".format(cell))
        seen.add(cell)
        records.append({
            "run": run,
            "source_sha256": sha256(raw),
            "test_metrics_text": raw.decode("utf-8-sig"),
        })
    if seen != expected:
        raise SystemExit("metadata export is not the exact frozen design")

    payload = {
        "schema": 1,
        "description": "Exact bytes of the 68 retained historical sea-ice test_metrics.json files, exported without a machine path.",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print("exported {} authenticated training records".format(len(records)))


if __name__ == "__main__":
    main()
