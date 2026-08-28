r"""Build the anonymous supplementary archive, and a manifest that ties it together.

The repository is not the release. It includes third-party
copyrighted PDFs, archived notebooks carrying the author's home directory, a README
naming the authors and linking a personal GitHub account, and the conference template.
Uploading it as supplementary material would ship all of that. An audit was right to
say so, and right that the supplement's release paragraphs described something that
did not exist.

So the archive is built, not swept up: an explicit allow-list, a generated manifest,
and a scan of the result for anything identifying. Nothing here is copied unless a
rule below names it.

    python make_release.py            # writes release/ and release_anon.zip
    python make_release.py --check    # rebuild, scan and verify; do not write a zip

The manifest records every payload path, size and SHA-256. Per-run kappa outputs are
included; per-run human-label damage scores were not retained in this repository or
archive, so those quantities are explicitly classified as summary-only rather than
reproduced.
"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "release"
BUILD = REPO / ".release-build-incomplete"
ZIP = REPO / "release_anon.zip"

# Explicit allow-list. A deny-list would let the next file added to the repository
# into the archive by default, which is how the identity-bearing notebooks would have
# travelled.
FILES = [
    "cropalign.py",
    "tests_cropalign_hardening.py",
    "SUPPLEMENT_README.md",
    "REPRODUCE.md",
    "LICENSE",
    "verify.py",                # one CPU entry point that checks the whole archive
    "DESIGN.json",              # the exact cells the experiment is designed to have,
                                # so the completeness gate compares against a frozen
                                # list rather than against what it happens to find
    # The claim-bearing paper sources, so the number checker can reach the claims it
    # verifies.  This is deliberately not advertised as a compile-ready template
    # bundle; the rendered PDFs are the submitted documents.
    # Without them it printed "no source files to audit", which is a verifier that
    # cannot verify: worse than none, because it invites the reader to assume a
    # check happened. They are anonymous; the scan below reads them like any other.
    "paper/main.tex",
    "paper/supp.tex",
    "paper/supp_perfold.tex",
    "paper/main.pdf",
    "paper/supp.pdf",
    "paper/main.log",
    "paper/supp.log",
    "requirements.txt",
    "paper/check_layout.py",
    "paper/check_presentation.py",
]
DIRS = [
    ("gate7_rebuild/src", "*.py"),
    ("gate8_crossdomain/src", "*.py"),
    ("gate7_rebuild/src", "*.sh"),
    ("gate8_crossdomain/src", "*.sh"),
    ("gate7_rebuild/results", "*.json"),
    ("gate8_crossdomain/results", "*.json"),
    ("paper/sec", "*.tex"),
    # the per-run scored outputs, so seed completeness can be audited and the
    # summaries recomputed rather than taken from a summary file
    ("gate7_rebuild/results/per_run_kappa", "*.json"),
    ("gate8_crossdomain/results/per_run_kappa", "*.json"),
]

# Historical scripts remain in the research repository for provenance, but these
# implement superseded pre-J1, premium, budget or fold-inferential analyses and do not produce a
# claim in the submitted paper. Shipping them beside the current summaries would
# make the anonymous archive's run path ambiguous.
RELEASE_EXCLUDE = {
    "gate8_crossdomain/src/figures_v3.py",
    "gate8_crossdomain/src/kappa_summary.py",
    "gate8_crossdomain/src/s1fc_summary.py",
    "gate8_crossdomain/src/s1f_mechanism.py",
    "gate7_rebuild/src/premium_test.py",
    "gate7_rebuild/src/mosaic_premium.py",
    "gate7_rebuild/src/mosaic_scale.py",
    "gate7_rebuild/src/epoch_control.py",
    "gate7_rebuild/src/tost_nulls.py",
    "gate7_rebuild/src/run_robustness.sh",
    "gate7_rebuild/src/run_mosaic.sh",
    "gate7_rebuild/src/run_epochs.sh",
    "gate7_rebuild/src/run_ep60_rest.sh",
    "gate7_rebuild/src/run_ep120.sh",
    "gate7_rebuild/src/ep120_robust.py",
    "gate7_rebuild/src/premium_local.py",
    "gate7_rebuild/src/robust_premium.py",
    "gate7_rebuild/src/power.py",
    "gate7_rebuild/src/kappa_predicts.py",
    "gate7_rebuild/src/patch_disjoint.py",
    "gate7_rebuild/src/tile_overlap.py",
    "gate7_rebuild/src/diagnose_folds.py",
    "gate7_rebuild/results/epoch_control.json",
    "gate7_rebuild/results/mosaic_premium.json",
    "gate7_rebuild/results/power.json",
    "gate7_rebuild/results/fold_diagnosis.json",
}

# The names to scan for live in an untracked file beside this one, one per line,
# and are never written here.
#
# They used to be assembled from string fragments so that this module would not match
# its own detector. An audit was right that this is the wrong instinct: fragmentation
# hides an identifier from a regex, not from a person, and a reviewer reading the
# archive reconstructs it at a glance. The fix is for the shipped tree to contain no
# name in any form, which also means this file is no longer shipped.
IDFILE = REPO / ".release_identities"


def _names():
    if not IDFILE.exists():
        raise SystemExit(
            "{} is missing. It holds one identifier per line -- author names, "
            "usernames, institutions, emails, repository handles -- and is "
            "deliberately untracked and never shipped.".format(IDFILE.name))
    return [n.strip() for n in IDFILE.read_text(encoding="utf-8").splitlines()
            if n.strip() and not n.startswith("#")]


_NAMES = _names()


def _flex_literal(value):
    """Match a literal even when a build log wraps it across a newline."""
    return r"[\r\n]*".join(re.escape(ch) for ch in value)


IDENTITY = re.compile(
    "|".join(_flex_literal(n) for n in _NAMES)
    + r"|/net/" + r"home/[^/\s\"']+/[^/\s\"']+"
    + r"|/" + r"home/[a-z][a-z0-9_-]*/"
    + r"|C:[\\/]Users[\\/]"
    + r"|[A-Za-z0-9._-]{1,}\s*[\\/]\s*AppData\s*[\\/](?:Local|Roaming)"
    + r"|github\.com/[A-Za-z0-9_-]+",
    re.I)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect():
    """Every file the archive will contain, as repo-relative paths."""
    out, missing = [], []
    for f in FILES:
        p = REPO / f
        if p.exists():
            out.append(f)
        else:
            missing.append(f)
    for d, pat in DIRS:
        root = REPO / d
        matches = sorted(root.glob(pat)) if root.is_dir() else []
        if not matches:
            missing.append("{}/{}".format(d, pat))
        for p in matches:
            rel = str(p.relative_to(REPO)).replace(os.sep, "/")
            if rel not in RELEASE_EXCLUDE:
                out.append(rel)
    if missing:
        raise SystemExit("required release input(s) missing:\n  " + "\n  ".join(missing))
    # The PDFs and logs are generated evidence. A source edit after either target
    # means the archive would pair fresh LaTeX with a stale render or stale warning
    # log, so refuse the build instead of relying on the user to remember a rebuild.
    paper_dir = REPO / "paper"
    common_sources = [paper_dir / name for name in (
        "preamble.tex", "main.bib", "wacv.sty", "ieeenat_fullname.bst")]
    common_sources += sorted((paper_dir / "figures").glob("*.png"))
    main_sources = common_sources + [paper_dir / "main.tex"]
    main_sources += sorted((paper_dir / "sec").glob("*.tex"))
    supp_sources = common_sources + [paper_dir / "supp.tex", paper_dir / "supp_perfold.tex"]
    build_sources = main_sources + supp_sources
    missing_build_inputs = [p for p in build_sources if not p.exists()]
    if missing_build_inputs:
        raise SystemExit("paper build input(s) missing:\n  " + "\n  ".join(
            str(p.relative_to(REPO)) for p in missing_build_inputs))
    # A PDF rebuilt against a stale .bbl is still stale.  Require each generated
    # bibliography to postdate both the database and its style before using its
    # timestamp as part of the PDF/log freshness check.
    bibliography_inputs = [paper_dir / "main.bbl", paper_dir / "supp.bbl"]
    bib_floor = max((paper_dir / "main.bib").stat().st_mtime_ns,
                    (paper_dir / "ieeenat_fullname.bst").stat().st_mtime_ns)
    for bbl in bibliography_inputs:
        if not bbl.exists() or bbl.stat().st_mtime_ns < bib_floor:
            raise SystemExit("{} is missing or predates the bibliography inputs; "
                             "run BibTeX and rebuild both documents before packaging"
                             .format(bbl.relative_to(REPO)))
    freshness_groups = {
        "paper/main.pdf": main_sources + [paper_dir / "main.bbl"],
        "paper/main.log": main_sources + [paper_dir / "main.bbl"],
        "paper/supp.pdf": supp_sources + [paper_dir / "supp.bbl"],
        "paper/supp.log": supp_sources + [paper_dir / "supp.bbl"],
    }
    # Tectonic may write the PDF/log and generated bibliography within the same
    # build in an order that differs by milliseconds.  Treat sub-second differences
    # as one build while still catching genuinely stale renders.
    freshness_slack_ns = 2_000_000_000
    for rel, sources in freshness_groups.items():
        newest_source = max(p.stat().st_mtime_ns for p in sources)
        if (REPO / rel).stat().st_mtime_ns + freshness_slack_ns < newest_source:
            raise SystemExit("{} predates a paper source; rebuild that document "
                             "before packaging".format(rel))
    return sorted(set(out))


def manifest(paths, root=REPO):
    """One row per shipped file, plus what the numbers in it were computed under."""
    scorers = {}
    for s in ("gate8_crossdomain/src/s1f_crop_kappa.py",
              "gate7_rebuild/src/crop_alignment.py",
              "cropalign.py"):
        p = root / s
        if p.exists():
            scorers[s] = sha256(p)
    rows = []
    for rel in paths:
        p = root / rel
        row = {"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p)}
        if rel.endswith(".json") and "/results/" in rel:
            row["kind"] = "result"
            # what produced the numbers in it, where the file records that itself
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                d = {}
            if isinstance(d, dict):
                if "kappa_by_event" in d or "kappa_noisy" in d:
                    row["weighting"] = "source-pixel (kappa); crop-read carried alongside"
                if "kappa_by_event" in d:
                    row["normalisation"] = (
                        "flood optical inputs: fold-local mean/std from each training run")
                elif "kappa_noisy" in d:
                    row["normalisation"] = (
                        "sea ice: seed 42 primary is optical-only with fixed ImageNet "
                        "RGB mean/std; seed 7 is a photon-enabled sensitivity with "
                        "fold-local non-test-acquisition photon-feature mean/std")
                if "scored_seeds" in d:
                    row["scored_seeds"] = d["scored_seeds"]
                    row["primary_seed"] = d.get("primary_seed")
                    row["input_mode_by_seed"] = d.get("input_mode_by_seed")
        else:
            row["kind"] = "code"
        rows.append(row)
    return {
        "what_this_is": "anonymous supplementary archive for a WACV Datasets Track "
                        "submission on crop-label alignment",
        "statistic": "the primary kappa is weighted uniformly over source pixels; "
                     "the crop-read weighting is reported alongside",
        "weighting_key_crosswalk": {
            "released_cropalign_api": "kappa and kappa_pixel are source-pixel; "
                                      "kappa_crop_read is crop-read",
            "legacy_pipeline_per_run_json": "kappa_pixel is source-pixel; kappa is "
                                            "crop-read",
            "summary_files": "use explicitly named source-pixel and crop-read fields",
        },
        "normalisation": {
            "flood": "fold-local optical mean/std recorded by each training run",
            "sea_ice_primary_seed_42":
                "optical-only; fixed ImageNet RGB mean/std in training and scoring",
            "sea_ice_sensitivity_seed_7":
                "optical plus ICESat-2 photon features; fixed ImageNet RGB optical "
                "normalisation and fold-local non-test-acquisition photon-feature "
                "mean/std",
        },
        "scorer_sha256": scorers,
        # This block said per-run outputs and the paper source were absent while the
        # archive shipped 369 of the former and 14 files of the latter. An audit
        # found the manifest contradicting the bytes beside it, which is the worst
        # place for this project to be wrong. It is now built from what is actually
        # excluded, and the tiers below say what each shipped thing supports.
        "not_included": [
            "imagery; public sources and repository-assisted fetch steps are "
            "documented in REPRODUCE.md, but the archive alone does not fetch sea ice",
            "model checkpoints, which live on the training machine",
            "per-run human-label damage scores: they were generated on the training "
            "machine but not retained here, so the shipped per-event summaries can "
            "be re-read but not rebuilt",
            "the conference template",
        ],
        "verification_tiers": {
            "1_estimator_controls":
                "reproduced: cropalign.py self-test and the hardening regressions "
                "run here and pin the estimator's behaviour",
            "2_kappa_reaggregation":
                "reproduced: the primary sea-ice and flood kappa tables are "
                "recomputed from exact per-run design cells; survey, architecture "
                "and grid summaries are stored-value consistency checks",
            "3_accuracy":
                "re-read, NOT rebuilt: the expert-label endpoint difference and "
                "the 94% scramble figure come from the per-event summaries "
                "shipped here, because per-run damage scores were not retained",
            "4_inference":
                "not reproduced: scoring model predictions needs the checkpoints "
                "and the imagery, neither of which ships",
            "5_training":
                "not reproduced: REPRODUCE.md documents it and it needs a GPU",
        },
        "files": rows,
    }


def scan(root):
    """Anything identifying, in paths, text, binary bytes or PDF metadata/text."""
    bad = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if IDENTITY.search(rel):
            bad.append((rel, "path"))
        try:
            raw = p.read_bytes()
        except OSError:
            raw = b""
        # latin-1 is a lossless byte-to-character mapping, so configured ASCII
        # identities are visible even in an otherwise binary file.
        m = IDENTITY.search(raw.decode("latin-1", errors="ignore"))
        if m:
            bad.append((rel, "contains {!r}".format(m.group(0)[:40])))
        if p.suffix.lower() == ".pdf":
            try:
                import fitz
                doc = fitz.open(p)
                pdf_text = "\n".join(page.get_text() for page in doc)
                pdf_text += "\n" + json.dumps(doc.metadata or {}, sort_keys=True)
                m = IDENTITY.search(pdf_text)
                if m:
                    bad.append((rel, "PDF text/metadata contains {!r}".format(
                        m.group(0)[:40])))
                doc.close()
            except Exception as exc:
                bad.append((rel, "PDF could not be inspected: {}".format(exc)))
    return bad


def selftest():
    """The scan must fire on a planted identity and ignore clean text."""
    plants = ["/" + "home/someone/x", "C:" + chr(92) + "Users" + chr(92) + "someone",
              "git" + "hub.com/someone", "h/AppData/Local", "sh/AppData/Roaming"]
    if _NAMES:
        n = _NAMES[0]
        cut = max(1, len(n) // 2)
        plants.append(n[:cut] + "\n" + n[cut:])
    ok = all(IDENTITY.search(t) for t in plants)
    clean = IDENTITY.search("results/run.json and ./src/train.py are fine")
    print("  self-test: plants seen={}  clean text quiet={}".format(ok, clean is None))
    return ok and clean is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="build, scan and verify in a staging tree; do not write the zip")
    ap.add_argument("--final", action="store_true",
                    help="also refuse the paper-ID placeholder")
    args = ap.parse_args()

    if args.final:
        unresolved = [p for p in (REPO / "paper" / "main.tex",
                                  REPO / "paper" / "supp.tex")
                      if "*****" in p.read_text(encoding="utf-8")]
        if unresolved:
            print("FINAL PREFLIGHT FAILED: replace the WACV paper-ID placeholder in "
                  + ", ".join(str(p.relative_to(REPO)) for p in unresolved))
            return 1

    print("=" * 70)
    print("BUILDING THE ANONYMOUS RELEASE")
    print("=" * 70)
    if not selftest():
        print("  the identity scan cannot see what it is meant to; not proceeding")
        return 1

    paths = collect()
    print("  {} files on the allow-list".format(len(paths)))
    # Build in a deliberately non-publishable staging directory. A failed check
    # preserves the last verified ZIP and release tree.
    if BUILD.exists():
        shutil.rmtree(BUILD)
    for rel in paths:
        dst = BUILD / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = REPO / rel
        if rel.endswith(".log"):
            # Build logs are evidence for the presentation gate, but MiKTeX writes
            # the local user-home prefix into every package path. Preserve every
            # warning and line number while removing only that identifying prefix.
            body = src.read_text(encoding="utf-8", errors="replace")
            body = re.sub(r"C:[\\/]Users[\\/]\s*[^\\/\s]+", "<user-home>", body,
                          flags=re.I)
            for identity in _NAMES:
                body = re.sub(_flex_literal(identity), "<redacted>", body, flags=re.I)
            body = re.sub(
                r"[A-Za-z0-9._-]{1,}\s*[\\/]\s*AppData\s*[\\/](Local|Roaming)",
                lambda m: "<user-home>/AppData/" + m.group(1), body, flags=re.I)
            dst.write_text(body, encoding="utf-8")
        else:
            shutil.copy2(src, dst)

    man = manifest(paths, BUILD)

    # One digest that names this exact archive, so a reviewer can say which version
    # they read without any identifying string. It is computed over the sorted
    # (path, sha256) pairs of the payload, NOT over the zip: the zip contains this
    # file, so hashing the zip and then writing the hash into it cannot converge.
    # The paper points at the identifier rather than quoting it, for the same reason.
    digest = hashlib.sha256()
    for row in sorted(man["files"], key=lambda r: r["path"]):
        digest.update((row["path"] + " " + row["sha256"] + chr(10)).encode("utf-8"))
    archive_id = digest.hexdigest()
    (BUILD / "ARCHIVE_ID.txt").write_text(
        "archive content id (sha256 over the sorted path+hash list of the "
        + str(len(man["files"])) + " payload files)" + chr(10)
        + archive_id + chr(10), encoding="utf-8")
    man["archive_content_id"] = archive_id
    _aid = BUILD / "ARCHIVE_ID.txt"
    man["files"].append({"path": "ARCHIVE_ID.txt",
                         "bytes": _aid.stat().st_size,
                         "sha256": sha256(_aid), "kind": "code"})
    (BUILD / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("  manifest: {} rows, {} scorer hashes".format(
        len(man["files"]), len(man["scorer_sha256"])))
    print("  archive content id: {}".format(archive_id))

    bad = scan(BUILD)
    print("  identity scan over the built tree: {}".format(
        "clean" if not bad else "{} HIT(S)".format(len(bad))))
    for rel, why in bad[:12]:
        print("     {}  {}".format(rel, why))
    if bad:
        print("\n  refusing to package an archive that names anyone")
        return 1

    print("\n  running the verifier from the built tree")
    env = os.environ.copy()
    env.pop("SEAICE_ROOT", None)
    proc = subprocess.run([sys.executable, "-B", "verify.py"], cwd=str(BUILD), env=env)
    if proc.returncode:
        print("  release verification failed; no zip written")
        return 1

    if args.check:
        print("\n  --check: built, scanned and verified; zip not written")
        shutil.rmtree(BUILD)
        return 0

    zip_tmp = ZIP.with_suffix(".zip.tmp")
    if zip_tmp.exists():
        zip_tmp.unlink()
    with zipfile.ZipFile(zip_tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(BUILD.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(BUILD))
    mb = zip_tmp.stat().st_size / 1e6
    if mb > 200:
        print("  OVER the 200 MB supplementary limit")
        zip_tmp.unlink()
        return 1
    os.replace(zip_tmp, ZIP)
    if OUT.exists():
        shutil.rmtree(OUT)
    os.replace(BUILD, OUT)
    print("  wrote {} ({:.1f} MB)".format(ZIP.name, mb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
