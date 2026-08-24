"""Recompute manuscript claims from artifacts actually committed to Git.

This deliberately distinguishes machine-readable evidence from text-only reports.
It exits nonzero when a verifiable number drifts, but a known text/JSON provenance
gap is emitted as a warning rather than silently converted into data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SECTION_FILES = sorted((ROOT / "paper" / "sections").glob("*.tex"))
PAPER = "\n".join(
    p.read_text(encoding="utf-8")
    for p in [ROOT / "paper" / "crop_alignment_wacv_2027.tex", *SECTION_FILES]
)
HEADLINE_PAPER = "\n".join(
    p.read_text(encoding="utf-8")
    for p in [
        ROOT / "paper" / "sections" / "abstract.tex",
        ROOT / "paper" / "sections" / "discussion.tex",
    ]
)


def require(label: str, value: str, failures: list[str]) -> None:
    if value not in PAPER:
        failures.append(f"{label}: expected {value!r} in paper")


def forbid(label: str, value: str, failures: list[str]) -> None:
    if value in HEADLINE_PAPER:
        failures.append(f"{label}: stale unsupported value {value!r} remains in a headline section")


def forbid_anywhere(label: str, value: str, failures: list[str]) -> None:
    if value in PAPER:
        failures.append(f"{label}: unsupported text {value!r} remains in the paper")


FLOOD_EVENTS = (
    "Bolivia",
    "Ghana",
    "India",
    "Mekong",
    "Nigeria",
    "Pakistan",
    "Paraguay",
    "Somalia",
    "Spain",
    "Sri-Lanka",
    "USA",
)


def parse_five_level_event_table(path: Path) -> np.ndarray:
    """Read the eleven rounded event rows from a committed flood summary log."""
    row_pattern = re.compile(
        r"^([A-Za-z-]+)\s+"
        + r"\s+".join([r"([+-]?\d+\.\d+)"] * 5)
        + r"$"
    )
    rows: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = row_pattern.fullmatch(line.strip())
        if match and match.group(1) in FLOOD_EVENTS:
            rows[match.group(1)] = [float(value) for value in match.groups()[1:]]
    missing = [event for event in FLOOD_EVENTS if event not in rows]
    if missing:
        raise ValueError(f"{path}: missing event rows {missing}")
    return np.asarray([rows[event] for event in FLOOD_EVENTS], dtype=float)


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    sea_path = ROOT / "experiments" / "sea_ice" / "results" / "kappa_summary.json"
    sea = json.loads(sea_path.read_text(encoding="utf-8"))
    keys = ("acqs", "kappa_noisy", "kappa_clean", "omega_noisy", "omega_clean", "frac_artefact")
    lengths = {key: len(sea[key]) for key in keys}
    if len(set(lengths.values())) != 1:
        failures.append(f"sea-ice summary arrays have different lengths: {lengths}")

    noisy = np.asarray(sea["kappa_noisy"], dtype=float)
    clean = np.asarray(sea["kappa_clean"], dtype=float)
    diff = noisy - clean
    t_stat = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
    computed = {
        "n": len(diff),
        "noisy_mean": noisy.mean(),
        "control_mean": clean.mean(),
        "difference": diff.mean(),
        "difference_sd": diff.std(ddof=1),
        "difference_t": t_stat,
        "positive": int((diff > 0).sum()),
        "artefact_percent": 100 * np.mean(sea["frac_artefact"]),
    }

    require("sea-ice difference", f"{computed['difference']:+.4f}", failures)
    require("sea-ice treatment mean", f"{computed['noisy_mean']:+.4f}", failures)
    require("sea-ice t", f"{computed['difference_t']:.2f}", failures)
    require("sea-ice folds", f"{computed['positive']} of {computed['n']} folds", failures)
    require("sea-ice control", f"{computed['control_mean']:+.4f}", failures)
    forbid("unverifiable 17-fold difference", "+0.1106", failures)
    forbid("unverifiable 17-fold t", "13.01", failures)
    forbid_anywhere("uncited fusion effect", "$+0.0306$", failures)
    forbid_anywhere("uncited fusion spread", "$0.0401$", failures)
    forbid_anywhere("uncited leak effect", "$0.0009$", failures)
    forbid_anywhere("unpublished comparison", "unpublished manuscript", failures)
    require("diagnostic scope", "cannot test a crop-invariant automatic labeller", failures)
    require("evidence boundary", "does not reproduce training", failures)

    mechanism_path = ROOT / "experiments" / "floods" / "results" / "mechanism.json"
    mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
    collapse = np.asarray(mechanism["collapse"], dtype=float)
    collapse_t = collapse.mean() / (collapse.std(ddof=1) / np.sqrt(len(collapse)))
    agreement_r = float(np.corrcoef(mechanism["agreement"], collapse)[0, 1])
    require("transfer collapse", f"{collapse.mean():+.4f}", failures)
    require("transfer collapse t", f"{collapse_t:.2f}", failures)
    require("agreement correlation", f"{agreement_r:+.3f}", failures)

    dose_path = ROOT / "experiments" / "floods" / "results" / "dose_response_analysis.txt"
    damage_path = ROOT / "experiments" / "floods" / "results" / "damage_analysis.txt"
    try:
        dose = parse_five_level_event_table(dose_path)
        damage = parse_five_level_event_table(damage_path)
    except ValueError as error:
        failures.append(str(error))
        dose = damage = np.empty((0, 5), dtype=float)

    flood_summary: dict[str, object] = {}
    if len(dose) == len(FLOOD_EVENTS) and len(damage) == len(FLOOD_EVENTS):
        dose_difference = dose[:, -1] - dose[:, 0]
        damage_difference = damage[:, -1] - damage[:, 0]
        dose_t = dose_difference.mean() / (
            dose_difference.std(ddof=1) / np.sqrt(len(dose_difference))
        )
        damage_t = damage_difference.mean() / (
            damage_difference.std(ddof=1) / np.sqrt(len(damage_difference))
        )
        within_event_r = float(
            np.corrcoef(
                (dose - dose.mean(axis=1, keepdims=True)).ravel(),
                (damage - damage.mean(axis=1, keepdims=True)).ravel(),
            )[0, 1]
        )
        perfectly_ordered = sum(bool(np.all(np.diff(row) > 0)) for row in dose)
        flood_summary = {
            "n_events": len(dose),
            "kappa_means_from_rounded_rows": dose.mean(axis=0).tolist(),
            "endpoint_difference": float(dose_difference.mean()),
            "endpoint_difference_sd": float(dose_difference.std(ddof=1)),
            "endpoint_t_from_rounded_rows": float(dose_t),
            "positive_endpoint_differences": int((dose_difference > 0).sum()),
            "perfectly_ordered_events": perfectly_ordered,
            "miou_means_from_rounded_rows": damage.mean(axis=0).tolist(),
            "miou_cost": float(damage_difference.mean()),
            "miou_cost_sd": float(damage_difference.std(ddof=1)),
            "miou_cost_t": float(damage_t),
            "events_worse": int((damage_difference < 0).sum()),
            "within_event_correlation": within_event_r,
        }
        require("flood kappa floor", f"{dose.mean(axis=0)[0]:+.4f}", failures)
        # The rounded event rows average to +0.2477; the log's summary, computed
        # before row rounding, is +0.2476 and is the value stated in the paper.
        require("flood kappa endpoint", "+0.2476", failures)
        require("flood endpoint difference", f"{dose_difference.mean():+.4f}", failures)
        require("flood endpoint sd", f"{dose_difference.std(ddof=1):.4f}", failures)
        require("flood ordered events", f"{perfectly_ordered} of {len(dose)} events", failures)
        require("flood mIoU cost", f"{damage_difference.mean():+.4f}", failures)
        require("flood mIoU cost sd", f"{damage_difference.std(ddof=1):.4f}", failures)
        require("flood mIoU cost t", f"{damage_t:.2f}", failures)
        require("flood within-event correlation", f"{within_event_r:+.3f}", failures)

    text_path = ROOT / "experiments" / "sea_ice" / "results" / "kappa_summary.txt"
    text = text_path.read_text(encoding="utf-8")
    if re.search(r"difference \(treated - control\) n=17\s+mean \+0\.1106", text):
        warnings.append(
            "kappa_summary.txt reports 17 folds (+0.1106), while JSON contains "
            f"{len(diff)}; the paper correctly uses the machine-readable result"
        )

    if dose_path.exists() and damage_path.exists():
        warnings.append(
            "flood calibration and cost aggregates recompute from rounded event rows "
            "in committed logs; raw predictions and checkpoints remain unavailable"
        )

    report = {
        "status": "fail" if failures else "pass_with_warnings" if warnings else "pass",
        "sea_ice_machine_readable": computed,
        "flood_committed_event_rows": flood_summary,
        "failures": failures,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
