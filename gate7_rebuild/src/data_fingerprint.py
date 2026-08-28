"""Content fingerprints for the exact arrays and code used by sea-ice runs.

Logical names, rather than absolute paths, enter each digest so fingerprints do
not disclose a machine path and remain stable when the dataset is relocated.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path


def sha256_file(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_digest(entries):
    rows = [{"name": name, "sha256": sha256_file(path)}
            for name, path in sorted(entries)]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def training_data_digest(cache, tokens, features, label_set):
    cache, tokens, features = Path(cache), Path(tokens), Path(features)
    label_file = {"scene": "labels.npy", "original": "labels_original.npy",
                  "kmeans": "labels_kmeans.npy"}[label_set]
    return manifest_digest([
        ("cache/images.npy", cache / "images.npy"),
        ("cache/" + label_file, cache / label_file),
        ("tokens/table.parquet", tokens),
        ("tokens/feature_cols.json", features),
    ])


@lru_cache(maxsize=None)
def scoring_data_digest(cache, tokens, features, scene_dir, eval_labels):
    cache, tokens, features, scene_dir = (Path(cache), Path(tokens),
                                          Path(features), Path(scene_dir))
    label_file = {"scene": "labels.npy", "original": "labels_original.npy",
                  "kmeans": "labels_kmeans.npy"}[eval_labels]
    valid = sorted(scene_dir.glob("*_valid.npy"))
    if not valid:
        raise FileNotFoundError("no scene-valid masks found under {}".format(scene_dir))
    entries = [
        ("cache/images.npy", cache / "images.npy"),
        ("cache/" + label_file, cache / label_file),
        ("tokens/table.parquet", tokens),
        ("tokens/feature_cols.json", features),
    ]
    entries.extend(("scene_valid/" + path.name, path) for path in valid)
    return manifest_digest(entries)


@lru_cache(maxsize=None)
def code_digest(*paths):
    return manifest_digest([("code/" + Path(path).name, Path(path))
                            for path in paths])
