r"""Emit the per-fold LaTeX tables the supplement should carry instead of a changelog.

A supplement that is twenty paragraphs of debugging history reads as instability, and
WACV forbids one that is "an updated or corrected version of the submission". What a
reader actually needs is the per-fold numbers behind every aggregate in the paper, so
that any of them can be recomputed by hand.
"""
import io
import json
import pathlib

import numpy as np

import os
REPO = pathlib.Path(os.environ.get("PROJECT_ROOT",
                                   pathlib.Path(__file__).resolve().parents[2]))
d = json.loads((REPO / "gate7_rebuild" / "results" / "kappa_summary.json").read_text())

acqs = d["acqs"]
kn = np.array(d["kappa_noisy"])
kc = np.array(d["kappa_clean"])
on = np.array(d["omega_noisy"])
oc = np.array(d["omega_clean"])
fa = np.array(d["frac_artefact"])
diff = kn - kc

rows = []
for a, x, y, z, w, v, f in zip(acqs, kn, kc, diff, on, oc, fa):
    short = a.split("_")[0] if "_" in a else a
    rows.append("{} & {:+.4f} & {:+.4f} & {:+.4f} & {:.4f} & {:.4f} & {:.1f} \\\\"
                .format(short, x, y, z, w, v, 100 * f))

n = len(kn)
def line(v):
    return "{:+.4f} & descriptive sd {:.4f}".format(v.mean(), v.std(ddof=1))

body = "\n".join(rows)
tex = r"""\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{lcccccc}
\toprule
acquisition & $\kap$ treated & $\kap$ control & difference & $\Omega$ tr. & $\Omega$ ct. & $|\A|$ \% \\
\midrule
""" + body + r"""
\midrule
mean & """ + "{:+.4f} & {:+.4f} & {:+.4f} & {:.4f} & {:.4f} & {:.1f}".format(
    kn.mean(), kc.mean(), diff.mean(), on.mean(), oc.mean(), 100 * fa.mean()) + r""" \\
\bottomrule
\end{tabular}
\caption{Every sea-ice fold in the optical-only primary arm (seed 42). The treated
model trains on the per-patch labels, the control on labels regenerated once per
scene, both on identical pixels. The difference column is what the paper reports:
mean """ + "{:+.4f}".format(diff.mean()) + r""", descriptive sd """ + \
    "{:.4f}".format(diff.std(ddof=1)) + r""", positive in """ + \
    "{} of {}".format(int((diff > 0).sum()), n) + r""" dependent folds. $|\A|$ is
the artefact set as a percentage of covered pixels. The photon-enabled seed-7
sensitivity is reported separately rather than averaged into this table.}
\label{tab:perfold}
\end{table}
"""

out = REPO / "paper" / "supp_perfold.tex"
io.open(out, "w", encoding="utf-8", newline="\n").write(tex)
print("wrote", out.name)
print("  treated  ", line(kn))
print("  control  ", line(kc))
print("  difference", line(diff))
print("  folds:", n, " positive:", int((diff > 0).sum()))
