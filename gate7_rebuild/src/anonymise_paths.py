r"""Strip the training cluster's absolute path out of the repository.

117 tracked files carried the training cluster's absolute path, which has two
components between the home root and the project directory and names a person.
That example is described rather than written out because writing it out is enough
for this file to match its own detector, which is how the closing check used to fail.
WACV tells authors to "take particular care to check submitted source code, which
often accidentally includes author identities", and a title search plus a username
undoes double-blind review.

TWO THINGS THIS SCRIPT LEARNED THE HARD WAY, both from its own first version.

The path has TWO components between /net/home and the project, not one. A regex
assuming one matched nothing, and the run still reported 96 files rewritten on the
strength of separate exact-string replacements, so a partial miss read as a pass.

And a generic "replace the path wherever it appears" pass is not safe in Python. It
turned `Path("<root>/src/train.py")` into an expression referring to a variable `W`
that those files never define. Twelve files came out parsing cleanly and broken at
run time, which is the failure mode this whole project is about. So the Python rules
are now explicit per pattern, nothing generic runs, and every touched file is both
parsed AND scanned for names it uses without defining before anything is written.

DELIBERATELY NOT RUN AGAINST THE TRAINING BOX while a job is executing there. Run it
again after the final pull, since scp brings the absolute paths back with the files.
"""
import ast
import builtins
import io
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
# Assembled from pieces rather than written out. Spelled literally, the pattern is
# itself a match for the broad detector below, so this file would be reported as
# carrying an identifying path on every run and the closing check could never pass.
_HOME = "/net/" + "home"
_SEG = r"[^/\s\"']+"

# What gets rewritten: the training project, whose two forms are known.
CLUSTER = _HOME + "/" + _SEG + "/" + _SEG + "/seaice_fusion"

# What gets detected: any user path under the cluster home, whatever project follows.
# The narrow pattern missed two files whose checkpoint lives under a different project
# directory entirely. Detection is deliberately wider than rewriting: anything found
# and not rewritten is reported and fails the run, rather than passing unseen.
DETECT = _HOME + "/" + _SEG + "/" + _SEG

# The Windows-side scripts carry the author's own home directory the same way, and
# were never covered here. Same convention as SEAICE_ROOT: an environment variable
# defaulting to the directory the script sits in, which is the repository root for
# every file that uses it.
WINDOWS = r'C:[\\/]Users[\\/][^\\/"\']+[\\/][Dd]esktop[\\/][^"\']*'

# A third shape, and the one that got past everything above: an ordinary Linux home
# directory, an ordinary Linux home path. The cluster pattern cannot match it, because that one
# requires TWO components between the home root and the project and this has one.
# Twenty-four tracked files carried it -- twenty-two notebooks and two result JSONs
# -- and the username is the author's, while this script reported the tree clean on
# every run. A detector that cannot express a shape reports its absence as success,
# which is the failure this whole project is about, here in the tool meant to prevent
# it. Assembled from pieces for the same reason as the others.
UNIXHOME = "/" + "home" + "/" + _SEG + "/"

# Two headers, because the two original forms differ in TYPE and that matters. A file
# written `W = Path(...)` goes on to do `W / "runs"`; a file written `W = "..."` goes
# on to do `W + "/runs"` or `os.path.join(W, ...)`, and a Path breaks the first of
# those. Both used to receive the Path header, which also assumed pathlib was imported
# when the string-form files never import it.
PY_HEADER = (
    'import os\n'
    'W = Path(os.environ.get("SEAICE_ROOT",\n'
    '                        Path(__file__).resolve().parents[1]))'
)
STR_HEADER = (
    'import os\n'
    'W = os.environ.get("SEAICE_ROOT",\n'
    '                   os.path.dirname(os.path.dirname(os.path.abspath(__file__))))'
)
WIN_HEADER = (
    'import os\n'
    'PROJECT = Path(os.environ.get("PROJECT_ROOT",\n'
    '                              Path(__file__).resolve().parent))'
)
WIN_SYSPATH = (
    'import os\n'
    'sys.path.insert(0, os.environ.get("PROJECT_ROOT",\n'
    '                                  os.path.dirname(os.path.abspath(__file__))))'
)
# The runners live in src/, so the project root is the PARENT of the script's
# directory. Resolving to the script's own directory instead put every rewritten
# runner one level too deep: each parsed fine, and each died at run time on
# `.venv/bin/activate: No such file or directory`. Eight scripts shipped that way.
SH_ROOT = 'cd "${SEAICE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"'


def defined_names(tree):
    """Every name the file binds anywhere: assignment, import, def, class, parameter,
    `except ... as`, `with ... as`, and comprehension target.

    Parameters and except-names were missing, which did not matter while the caller
    only compared before against after, since a missing name was missing on both
    sides. It matters now that the same function is useful as a check in its own
    right: without them it reports a parameter as an undefined global.
    """
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
    return out


def undefined_globals(src):
    """Names used at module level that nothing in the file defines or imports."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return ["SYNTAX: " + str(e)[:60]]
    # `dir(__builtins__)` is wrong inside an imported module, where __builtins__ is a
    # dict and dir() returns dict methods rather than the builtins. The real module.
    known = defined_names(tree) | set(dir(builtins)) | {
        "__file__", "__name__", "__doc__", "__builtins__", "self", "cls"}
    bad = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            # This used to require `n.id.isupper() and len(n.id) <= 2`, which only
            # ever caught `W`. The header it writes also uses `Path`, and four files
            # whose original line was `W = "<path>"` do not import pathlib at all, so
            # they took a header referring to a name they never define. All four
            # parsed, so the guard passed them, and they raised NameError on import.
            # One shipped that way through the first run of this script and stayed
            # broken. Any undefined module-level name now counts.
            if n.id not in known:
                bad.add(n.id)
    return sorted(bad)


def selftest():
    """Plant one path of each shape and confirm the detector sees all three.

    This exists because the detector silently could not express the an ordinary Linux home path
    shape, and reported the tree clean while twenty-four files carried the author's
    username. A detector is only trustworthy if it has been shown to fire.
    """
    pat = re.compile("(" + DETECT + ")|(" + WINDOWS + ")|(" + UNIXHOME + ")")
    plants = [
        ("cluster", "/net/" + "home/someone/Project/seaice_fusion/src/train.py"),
        ("windows", "C:" + chr(92) + "Users" + chr(92) + "someone" + chr(92) +
                    "Desktop" + chr(92) + "Project"),
        ("unix home", "/" + "home/someone/Research/run.py"),
    ]
    ok = True
    for name, text in plants:
        hit = bool(pat.search(text))
        if not hit:
            ok = False
        print("  self-test {:10s} -> {}".format(name, "seen" if hit else "MISSED"))
    clean = "results/run.json and ./src/train.py carry no identity"
    if pat.search(clean):
        ok = False
        print("  self-test clean text -> FALSE POSITIVE")
    return ok


def carrying(repo):
    """Tracked files that name a person, found by the shape of the path.

    This used to be `git grep -l <username>`, which matched THIS file, so the run's
    closing verification could never report clean: the script's own check was one that
    always failed, the mirror of the failure it exists to prevent. Matching on the
    shape of the path instead leaves nothing here for it to match.
    """
    pat = re.compile("(" + DETECT + ")|(" + WINDOWS + ")|(" + UNIXHOME + ")")
    out = []
    listed = subprocess.run(["git", "-C", str(repo), "ls-files"],
                            capture_output=True, text=True).stdout.split("\n")
    # Tracked files are not the same set as shipped files. make_supp_tables.py was
    # untracked, so this scan could not see the author's home directory hardcoded in
    # it, while the release builder copies it by directory glob and would have
    # shipped it. The release builder's own scan caught it; this one could not.
    # Source directories are now scanned whether or not git knows about them.
    for d in ("", "gate1/src", "gate2_prep/src", "gate3/src", "gate4_replication/src",
              "gate5_expansion/src", "gate6_scale/src", "gate7_rebuild/src",
              "gate8_crossdomain/src", "paper"):
        for glob_pat in ("*.py", "*.sh"):
            listed += [str(f.relative_to(repo)).replace(chr(92), "/")
                       for f in sorted((repo / d).glob(glob_pat))]
    for n in listed:
        if not n.strip():
            continue
        f = repo / n
        if not f.exists():
            continue
        try:
            body = io.open(f, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue                          # binary, and it carries no source path
        if pat.search(body):
            out.append(f)
    return out


def fix_python(s):
    """Only the forms that actually occur, and only at the top level."""
    m = re.search(r'^W = Path\("' + CLUSTER + r'"\)$', s, re.M)
    if m:
        s = s[:m.start()] + PY_HEADER + s[m.end():]
    m = re.search(r'^W = "' + CLUSTER + r'"$', s, re.M)
    if m:
        s = s[:m.start()] + STR_HEADER + s[m.end():]
    # sub-paths become relative to W, which the header now guarantees exists
    s = re.sub(r'Path\("' + CLUSTER + r'/([^"]+)"\)', r'(W / "\1")', s)
    s = re.sub(r'"' + CLUSTER + r'/([^"]+)"', r'str(W / "\1")', s)
    # the Windows forms, again only the two that occur
    s = re.sub(r'^PROJECT = Path\(r"' + WINDOWS + r'"\)$', WIN_HEADER, s, flags=re.M)
    s = re.sub(r'^sys\.path\.insert\(0, r"' + WINDOWS + r'"\)$', WIN_SYSPATH, s,
               flags=re.M)
    return s


def fix_shell(s):
    s = re.sub(r"^cd " + CLUSTER + r"$", SH_ROOT, s, flags=re.M)
    return re.sub(CLUSTER, '"$SEAICE_ROOT"', s)


def main():
    if not selftest():
        print("the detector cannot see a shape it is meant to catch; not proceeding")
        return 1
    files = carrying(REPO)
    print("files carrying an identifying path:", len(files))

    staged, refused = {}, []
    for f in files:
        if not f.exists():
            continue
        s = io.open(f, encoding="utf-8", errors="surrogateescape").read()
        if f.suffix == ".py":
            new = fix_python(s)
            if new != s:
                before, after = undefined_globals(s), undefined_globals(new)
                if set(after) - set(before):
                    refused.append((f, "would use undefined " + ", ".join(
                        set(after) - set(before))))
                    continue
        elif f.suffix == ".sh":
            new = fix_shell(s)
        else:
            new = re.sub(WINDOWS, "<repo>", re.sub(CLUSTER, "<repo>", s))
            # an ordinary Linux home pathx keeps its tail: the trailing slash is part of the
            # pattern, so the replacement carries one too.
            new = re.sub(UNIXHOME, "<repo>/", new)
            if f.suffix in (".ipynb", ".json"):
                # These are structured. A path sits inside a JSON string and the
                # replacement introduces no quote or backslash, so it stays valid --
                # but assert that rather than assume it, because a corrupted notebook
                # would still be written and would still look like a success.
                try:
                    json.loads(new)
                except ValueError as e:
                    refused.append((f, "rewrite would break JSON: " + str(e)[:50]))
                    continue
        if new != s:
            staged[f] = new

    for f, why in refused:
        print("  REFUSED {}: {}".format(f.relative_to(REPO), why))
    for f, new in staged.items():
        io.open(f, "w", encoding="utf-8", errors="surrogateescape",
                newline="\n").write(new)
    print("rewritten: {}   refused: {}".format(len(staged), len(refused)))

    left = carrying(REPO)
    if left:
        print("still carrying an identifying path: {} files".format(len(left)))
        for f in left[:10]:
            print("  ", f.relative_to(REPO))
        return 1
    print("no tracked file carries a cluster or home path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
