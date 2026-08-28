r"""One command that checks this archive, on a CPU, from a fresh extraction.

An audit found that the previous release shipped `verify_draft_numbers.py` without
the LaTeX sources it reads, so running it printed "no source files to audit". A
verifier that cannot reach the claims it verifies is not evidence of anything, and
shipping one is worse than shipping none: it invites the reader to assume a check
happened.

So the archive now carries the paper sources, the per-run scored outputs, and this.
It runs every CPU-only gate, reports all failures, and exits nonzero if any do not
hold:

    python verify.py

  1. the estimator's own controls, including the case that separates the two
     weightings
  2. the lifecycle regressions
  3. stored-value claims plus exact reaggregation of the primary kappa tables
  4. the completeness of the scored set, against the canonical product rather than
     against a count
  5. rendered layout and presentation gates
  6. the manifest, content ID and SHA-256 for every shipped payload file

What it does NOT do is retrain anything or re-run inference; that needs the imagery
and a GPU, and REPRODUCE.md documents it separately.
"""
import collections
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(label, args, cwd=None):
    print("-" * 68)
    print(label)
    print("-" * 68)
    env = dict(**__import__("os").environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("SEAICE_ROOT", None)
    r = subprocess.run([sys.executable, "-B"] + args,
                       cwd=str(cwd or ROOT), env=env)
    ok = r.returncode == 0
    print("  -> {}".format("ok" if ok else "FAILED (exit {})".format(r.returncode)))
    return ok


def check_manifest():
    print("-" * 68)
    print("manifest: every shipped file matches its recorded SHA-256")
    print("-" * 68)
    mf = ROOT / "MANIFEST.json"
    if not mf.exists():
        print("  -> FAILED: MANIFEST.json is missing")
        return False
    man = json.loads(mf.read_text(encoding="utf-8"))
    bad, checked = [], 0
    rows = man.get("files")
    if not isinstance(rows, list) or not rows:
        print("  -> FAILED: manifest has no file rows")
        return False
    seen = set()
    for row in rows:
        rel = row.get("path") if isinstance(row, dict) else None
        if not isinstance(rel, str) or not rel or "\\" in rel:
            bad.append((repr(rel), "invalid relative path"))
            continue
        parts = Path(rel).parts
        if Path(rel).is_absolute() or any(x in ("", ".", "..") for x in parts):
            bad.append((rel, "path is not normalized and relative"))
            continue
        if rel in seen:
            bad.append((rel, "duplicate manifest row"))
            continue
        seen.add(rel)
        p = ROOT / rel
        if not p.exists():
            bad.append((rel, "missing"))
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        checked += 1
        if h.hexdigest() != row.get("sha256"):
            bad.append((rel, "hash differs"))
        if p.stat().st_size != row.get("bytes"):
            bad.append((rel, "byte count differs"))
    # An unlisted file is as much a manifest failure as a missing one: hashing only
    # what the manifest names cannot notice something that was added afterwards.
    listed = seen | {"MANIFEST.json"}
    for f in sorted(ROOT.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(ROOT).as_posix()
        if rel in listed:
            continue
        bad.append((rel, "present but not listed in the manifest"))
    # Scorer hashes and the content identifier are part of the manifest contract,
    # not decorative metadata. The exact key set is required: an absent or partial
    # dictionary must not turn this loop into a successful no-op.
    expected_scorers = {
        "gate8_crossdomain/src/s1f_crop_kappa.py",
        "gate7_rebuild/src/crop_alignment.py",
        "cropalign.py",
    }
    scorer_rows = man.get("scorer_sha256")
    if not isinstance(scorer_rows, dict) or set(scorer_rows) != expected_scorers:
        bad.append(("scorer_sha256", "must contain exactly the three scorer paths"))
        scorer_rows = scorer_rows if isinstance(scorer_rows, dict) else {}
    for rel, expected in scorer_rows.items():
        p = ROOT / rel
        if rel not in seen:
            bad.append((rel, "scorer is not also a manifest payload row"))
        if not p.exists():
            bad.append((rel, "scorer named by manifest is missing"))
        elif hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            bad.append((rel, "scorer_sha256 differs"))
    digest = hashlib.sha256()
    for row in sorted((r for r in rows if r.get("path") != "ARCHIVE_ID.txt"),
                      key=lambda r: r["path"]):
        digest.update((row["path"] + " " + row["sha256"] + "\n").encode("utf-8"))
    got_id = digest.hexdigest()
    if man.get("archive_content_id") != got_id:
        bad.append(("archive_content_id", "does not match payload rows"))
    aid = ROOT / "ARCHIVE_ID.txt"
    if not aid.exists() or got_id not in aid.read_text(encoding="utf-8"):
        bad.append(("ARCHIVE_ID.txt", "does not carry archive_content_id"))
    for path, why in bad[:10]:
        print("  {}: {}".format(path, why))
    print("  {} file(s) checked against {} listed, {} problem(s)".format(
        checked, len(rows), len(bad)))
    print("  -> {}".format("ok" if not bad else "FAILED"))
    return not bad


def check_cells():
    """The scored set must be the exact design, cell by cell, parsed not counted.

    The previous version counted filenames matching a prefix. It could not tell one
    acquisition from another, could not see a cell scored twice under two names, and
    said nothing at all about the human-label damage scores. So it passed on a set
    that happened to be right, which is not the same as checking that it is.

    This reads DESIGN.json, parses every file into its (unit, arm, seed) tuple, and
    requires exactly one record per designed cell: missing, unexpected and duplicate
    are three different failures and are reported as three.
    """
    print("-" * 68)
    print("completeness: every designed cell scored exactly once")
    print("-" * 68)
    dfile = ROOT / "DESIGN.json"
    if not dfile.exists():
        print("  -> FAILED: DESIGN.json is missing, so there is no design to check")
        return False
    design = json.loads(dfile.read_text(encoding="utf-8"))
    ok = True

    def close(label, got, want, tol=1e-12):
        nonlocal ok
        if len(got) != len(want) or any(
                not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
                for a, b in zip(got, want)):
            print("    SUMMARY MISMATCH {}".format(label))
            ok = False

    def report(name, expected, found):
        nonlocal ok
        seen = collections.Counter(found)
        missing = sorted(expected - set(seen))
        extra = sorted(set(seen) - expected)
        dupes = sorted(c for c, n in seen.items() if n > 1)
        print("  {}: {} designed cells, {} records".format(
            name, len(expected), len(found)))
        for what, rows in (("MISSING", missing), ("UNEXPECTED", extra),
                           ("DUPLICATE", dupes)):
            for r in rows[:4]:
                print("    {} {}".format(what, r))
            if len(rows) > 4:
                print("    ... and {} more {}".format(len(rows) - 4, what))
        if missing or extra or dupes:
            ok = False
        else:
            print("    none missing, none unexpected, none duplicated")

    # --- sea ice: loao_<arm>_<acq>_s<seed>.json
    sea = design["sea_ice"]
    expected = {(a, arm, sd) for a in sea["acquisitions"]
                for arm in sea["arms"] for sd in sea["seeds"]}
    found = []
    sea_rows = []
    d = ROOT / "gate7_rebuild" / "results" / "per_run_kappa"
    # read the tuple out of the record, not out of the name. The two arms are named
    # differently on disk (the control carries a _vsorig suffix), and a filename
    # pattern that fits one arm silently drops the other: an earlier version of this
    # very check reported 34 of 68 cells for exactly that reason.
    for f in sorted(d.glob("loao_*.json")) if d.exists() else []:
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("eval_labels", r.get("labels")) != "original":
            continue
        found.append((r.get("acq"), r.get("labels"), r.get("seed")))
        sea_rows.append(r)
    report("sea ice", expected, found)
    input_modes = {int(k): v for k, v in sea.get("input_mode_by_seed", {}).items()}
    if set(input_modes) != set(sea["seeds"]):
        print("    PROVENANCE MISMATCH DESIGN.json does not map every seed to an input mode")
        ok = False
    metadata_path = (ROOT / "gate7_rebuild" / "results" /
                     "seaice_training_metadata.json")
    metadata_by_cell = {}
    if not metadata_path.exists():
        print("    PROVENANCE MISMATCH authenticated training metadata is absent")
        ok = False
    else:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        for item in payload.get("records", []):
            try:
                raw = item["test_metrics_text"].encode("utf-8")
                if hashlib.sha256(raw).hexdigest() != item["source_sha256"]:
                    raise ValueError("source hash mismatch")
                meta = json.loads(item["test_metrics_text"])
                cell = (meta.get("holdout_acq"), meta.get("labels"),
                        meta.get("seed"))
                if cell in metadata_by_cell:
                    raise ValueError("duplicate cell {}".format(cell))
                metadata_by_cell[cell] = meta
            except (KeyError, TypeError, ValueError) as exc:
                print("    PROVENANCE MISMATCH {}: {}".format(
                    item.get("run", "unnamed record"), exc))
                ok = False
        if set(metadata_by_cell) != expected:
            print("    PROVENANCE MISMATCH training metadata is not the exact design")
            ok = False
    for r in sea_rows:
        seed = r.get("seed")
        cell = (r.get("acq"), r.get("labels"), seed)
        historical = metadata_by_cell.get(cell, {})
        want_photon = historical.get("photon")
        want_mode = ("optical-only" if want_photon == "none" else
                     "optical+ICESat-2 photon" if want_photon == "true" else None)
        if want_mode != input_modes.get(seed):
            print("    PROVENANCE MISMATCH metadata and DESIGN disagree for {}"
                  .format(r.get("run")))
            ok = False
        if (r.get("input_mode") != want_mode or
                r.get("photon_mode") != want_photon or
                r.get("optical_normalisation") != "fixed ImageNet RGB mean/std" or
                r.get("photon_normalisation") !=
                ("not used" if want_photon == "none" else
                 "fold-local non-test-acquisition mean/std")):
            print("    PROVENANCE MISMATCH {} seed {}".format(r.get("run"), seed))
            ok = False
    sea_summary = ROOT / "gate7_rebuild" / "results" / "kappa_summary.json"
    if sea_summary.exists() and len(found) == len(expected):
        s = json.loads(sea_summary.read_text(encoding="utf-8"))
        acqs = sorted(sea["acquisitions"])
        if s.get("acqs") != acqs:
            print("    SUMMARY MISMATCH acquisition order")
            ok = False
        primary_seed = int(sea.get("primary_seed", -1))
        if (s.get("primary_seed") != primary_seed or
                s.get("primary_input_mode") != input_modes.get(primary_seed) or
                s.get("input_mode_by_seed") !=
                {str(k): v for k, v in input_modes.items()}):
            print("    SUMMARY MISMATCH primary seed/input-mode metadata")
            ok = False
        def sea_values(label, key, seed):
            nonlocal ok
            vals = []
            for a in acqs:
                cell = [float(r[key]) for r in sea_rows
                        if r["acq"] == a and r["labels"] == label
                        and r["seed"] == seed]
                if len(cell) != 1:
                    print("    SUMMARY MISMATCH {} {} seed {}".format(a, label, seed))
                    ok = False
                    return []
                vals.append(cell[0])
            return vals
        close("sea kappa treated", sea_values("original", "kappa_pixel", primary_seed),
              s.get("kappa_noisy", []))
        close("sea kappa control", sea_values("scene", "kappa_pixel", primary_seed),
              s.get("kappa_clean", []))
        close("sea crop-read treated", sea_values("original", "kappa", primary_seed),
              s.get("kappa_cropread_noisy", []))
        close("sea crop-read control", sea_values("scene", "kappa", primary_seed),
              s.get("kappa_cropread_clean", []))
        close("sea omega treated", sea_values("original", "omega", primary_seed),
              s.get("omega_noisy", []))
        close("sea omega control", sea_values("scene", "omega", primary_seed),
              s.get("omega_clean", []))
        close("sea artefact prevalence", sea_values("original", "frac_artefact", primary_seed),
              s.get("frac_artefact", []))
        by_seed = s.get("by_seed", {})
        for seed in sea["seeds"]:
            row = by_seed.get(str(seed), {})
            if row.get("input_mode") != input_modes.get(seed):
                print("    SUMMARY MISMATCH input mode for seed {}".format(seed))
                ok = False
            for label, run_key, summary_key in (
                    ("original", "kappa_pixel", "kappa_noisy"),
                    ("scene", "kappa_pixel", "kappa_clean"),
                    ("original", "kappa", "kappa_cropread_noisy"),
                    ("scene", "kappa", "kappa_cropread_clean"),
                    ("original", "omega", "omega_noisy"),
                    ("scene", "omega", "omega_clean")):
                close("sea seed {} {}".format(seed, summary_key),
                      sea_values(label, run_key, seed), row.get(summary_key, []))
        if ok:
            print("    optical-only primary and photon-enabled sensitivity "
                  "reaggregate exactly from 68 runs")
    else:
        print("    sea-ice summary missing or design incomplete")
        ok = False

    # --- flood kappa: j1_<arm>_<event>_s<seed>.json
    fl = design["flood"]
    expected = {(e, arm, sd) for e in fl["events"]
                for arm in fl["arms"] for sd in fl["seeds"]}
    found = []
    flood_rows = []
    mismatched = []
    d = ROOT / "gate8_crossdomain" / "results" / "per_run_kappa"
    # Parsed from the payload, like the sea-ice arm above. This read the FILENAME
    # while the docstring promised the record, which is a gate weaker than its own
    # wording: a file renamed, or written under a name that disagrees with what is
    # inside it, would have been counted as whatever the name said. The name is now
    # checked against the record instead of trusted in its place.
    for f in sorted(d.glob("j1_*.json")) if d.exists() else []:
        if "_thr" in f.name or "_step" in f.name or "_foldnorm" in f.name:
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        cell = (r.get("event"), r.get("arm"), r.get("seed"))
        found.append(cell)
        flood_rows.append(r)
        parts = f.stem.split("_")
        if len(parts) >= 4 and parts[-1].startswith("s") and parts[-1][1:].isdigit():
            from_name = ("_".join(parts[2:-1]), parts[1], int(parts[-1][1:]))
            if from_name != cell:
                mismatched.append((f.name, from_name, cell))
    report("flood kappa", expected, found)
    for name, fn, rec in mismatched[:4]:
        print("    NAME DISAGREES WITH RECORD  {}: name says {}, file says {}".format(
            name, fn, rec))
    if mismatched:
        ok = False
    else:
        print("    every filename agrees with the record inside it")

    flood_summary = ROOT / "gate8_crossdomain" / "results" / "j1_summary.json"
    if flood_summary.exists() and len(found) == len(expected):
        s = json.loads(flood_summary.read_text(encoding="utf-8"))
        for summary_key, run_key in (("kappa_by_event", "kappa_pixel"),
                                     ("kappa_cropread_by_event", "kappa"),
                                     ("omega_by_event", "omega")):
            stored = s.get(summary_key, {})
            for arm in fl["arms"]:
                for event in fl["events"]:
                    vals = [float(r[run_key]) for r in flood_rows
                            if r["arm"] == arm and r["event"] == event]
                    if len(vals) != len(fl["seeds"]):
                        ok = False
                        continue
                    want = stored.get(arm, {}).get(event)
                    if want is None or not math.isclose(
                            sum(vals) / len(vals), float(want),
                            rel_tol=0.0, abs_tol=1e-12):
                        print("    SUMMARY MISMATCH {} {} {}".format(
                            summary_key, arm, event))
                        ok = False
        if ok:
            print("    primary flood summaries reaggregate exactly from 231 runs")
    else:
        print("    flood summary missing or design incomplete")
        ok = False

    # --- damage: re-read from a summary, so say that rather than count nothing
    summ = ROOT / "gate8_crossdomain" / "results" / "j1_summary.json"
    if not summ.exists():
        print("  damage: j1_summary.json absent")
        ok = False
    else:
        dmg = json.loads(summ.read_text(encoding="utf-8")).get("damage_by_event", {})
        # Against the design's exact arm and event names, not against a count. The
        # previous version asked only that every arm present carried eleven entries,
        # so a summary with the right shape and the wrong identities passed.
        want_arms, want_events = set(fl["arms"]), set(fl["events"])
        got_arms = set(dmg)
        print("  damage: {} arms over {} events, against the design".format(
            len(got_arms), len(want_events)))
        print("    NOT rebuilt from per-run scores: they were not retained here, so "
              "this is a re-read of the per-event summary")
        for what, missing, extra in (
                ("arm", sorted(want_arms - got_arms), sorted(got_arms - want_arms)),):
            for m in missing[:4]:
                print("    MISSING {} {}".format(what, m))
            for e in extra[:4]:
                print("    UNEXPECTED {} {}".format(what, e))
            if missing or extra:
                ok = False
        for a in sorted(want_arms & got_arms):
            got_ev = set(dmg[a])
            miss, extra = sorted(want_events - got_ev), sorted(got_ev - want_events)
            for m in miss[:2]:
                print("    ARM {}: MISSING event {}".format(a, m))
            for e in extra[:2]:
                print("    ARM {}: UNEXPECTED event {}".format(a, e))
            if miss or extra:
                ok = False
        if ok:
            print("    every designed arm and event present, none unexpected")

    print("  -> {}".format("ok" if ok else "FAILED"))
    return ok


def main():
    print("=" * 68)
    print("VERIFYING THIS ARCHIVE")
    print("=" * 68)
    results = [
        ("estimator self-test", run(
            "1. estimator controls", ["cropalign.py"])),
        ("lifecycle regressions", run(
            "2. lifecycle and hardening regressions",
            ["tests_cropalign_hardening.py"])),
        ("paper numbers", run(
            "3. stored-value claims: exact inputs, sign-aware tokens, no silent skips",
            ["gate7_rebuild/src/verify_draft_numbers.py"])),
        ("completeness", check_cells()),
        ("layout", run("5a. rendered layout", ["paper/check_layout.py"])),
        ("presentation", run(
            "5b. citations, commands, floats and build logs",
            ["paper/check_presentation.py"])),
        ("manifest", check_manifest()),
    ]
    print("=" * 68)
    bad = [n for n, ok in results if not ok]
    for n, ok in results:
        print("  {:24s} {}".format(n, "ok" if ok else "FAILED"))
    print("=" * 68)
    if bad:
        print("{} check(s) failed: {}".format(len(bad), ", ".join(bad)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
