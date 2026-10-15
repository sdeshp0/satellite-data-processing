"""
Plain-language "key findings" for a before/after comparison.

Turns a classification (core.change classifiers), its area summary and the
pair's comparability notes into a short list of sentences: the headline
change in hectares, where in the AOI it is concentrated, and the caveats
that most affect how far to trust it. Rules-based and deterministic, so the
wording is testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.change import ChangeAreaSummary, ComparabilityNote, format_area

# Below this share of the analysed area, change is reported as negligible.
NEGLIGIBLE_CHANGE_PCT = 1.0
# A 3x3 sector holding at least this share of changed pixels (vs ~11% if
# spread evenly) is reported as where the change is concentrated.
CONCENTRATION_SHARE = 0.4
# Too few changed pixels to say anything meaningful about their layout.
MIN_PIXELS_FOR_LOCATION = 50
# Below this analysed share of the AOI window, totals are a lower bound.
LOW_COVERAGE_PCT = 90.0

_SECTOR_NAMES = [
    ["north-west", "north", "north-east"],
    ["west", "centre", "east"],
    ["south-west", "south", "south-east"],
]


@dataclass
class Finding:
    kind: str  # "headline", "detail", or "caveat"
    text: str


def _area_of(codes: np.ndarray, class_codes: Sequence[int], pixel_area_ha: float) -> float:
    return int(np.isin(codes, list(class_codes)).sum()) * pixel_area_ha


def _pct(ha: float, area: ChangeAreaSummary) -> float:
    return 100 * ha / area.analysed_ha if area.analysed_ha else 0.0


def _share(ha: float, area: ChangeAreaSummary) -> str:
    return f"{format_area(ha)} ({_pct(ha, area):.1f}% of the analysed area)"


def _wildfire(codes, px_ha, area) -> List[Finding]:
    burned = _area_of(codes, [3, 4, 5, 6], px_ha)
    severe = _area_of(codes, [5, 6], px_ha)
    regrowth = _area_of(codes, [0, 1], px_ha)
    out = [Finding("headline", f"About {_share(burned, area)} burned at low severity or worse.")]
    if severe:
        out.append(Finding(
            "detail",
            f"{format_area(severe)} of that is moderate-high or high severity, "
            f"{100 * severe / burned:.0f}% of the burned area.",
        ))
    if _pct(regrowth, area) >= NEGLIGIBLE_CHANGE_PCT:
        out.append(Finding("detail", f"{_share(regrowth, area)} shows enhanced regrowth."))
    return out


def _flood(codes, px_ha, area) -> List[Finding]:
    new = _area_of(codes, [2], px_ha)
    lost = _area_of(codes, [3], px_ha)
    persistent = _area_of(codes, [1], px_ha)
    out = [Finding("headline", f"{_share(new, area)} became newly water-covered.")]
    if _pct(lost, area) >= NEGLIGIBLE_CHANGE_PCT:
        out.append(Finding("detail", f"{_share(lost, area)} lost surface water."))
    if persistent:
        out.append(Finding("detail", f"{format_area(persistent)} was water on both dates."))
    return out


def _logging(codes, px_ha, area) -> List[Finding]:
    loss = _area_of(codes, [2], px_ha)
    gain = _area_of(codes, [3], px_ha)
    out = [Finding("headline", f"{_share(loss, area)} lost vegetation cover.")]
    if gain:
        net = loss - gain
        direction = "net loss" if net >= 0 else "net gain"
        out.append(Finding(
            "detail",
            f"{format_area(gain)} gained vegetation, a {direction} of {format_area(abs(net))}.",
        ))
    return out


def _custom(codes, px_ha, area, index_name: str) -> List[Finding]:
    up = _area_of(codes, [1], px_ha)
    down = _area_of(codes, [2], px_ha)
    name = index_name.upper()
    return [Finding(
        "headline",
        f"{_share(up, area)} shows a significant increase in {name} and "
        f"{_share(down, area)} a significant decrease.",
    )]


_HEADLINES = {"wildfire": _wildfire, "flood": _flood, "logging": _logging}


def change_location(changed: np.ndarray) -> Optional[str]:
    """
    Where changed pixels sit in the AOI, assuming a north-up grid: the name
    of the 3x3 sector holding at least CONCENTRATION_SHARE of them, "spread"
    if none does, or None when there are too few pixels to say.
    """
    total = int(changed.sum())
    if total < MIN_PIXELS_FOR_LOCATION:
        return None
    rows = np.array_split(np.arange(changed.shape[0]), 3)
    cols = np.array_split(np.arange(changed.shape[1]), 3)
    best_share, best_name = 0.0, None
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            share = changed[np.ix_(r, c)].sum() / total
            if share > best_share:
                best_share, best_name = share, _SECTOR_NAMES[i][j]
    return best_name if best_share >= CONCENTRATION_SHARE else "spread"


def change_findings(
    preset_key: str,
    codes: np.ndarray,
    baseline_codes: Sequence[int],
    area: ChangeAreaSummary,
    pixel_area_ha: float,
    notes: Sequence[ComparabilityNote] = (),
    index_name: str = "",
) -> List[Finding]:
    """
    Key findings for one comparison, headline first and caveats last.
    `codes` is the preset's classification (-1 = masked); `area` is the
    matching core.change.change_area_summary result.
    """
    if not area.analysed_ha:
        return [Finding(
            "caveat",
            "No part of the AOI has usable data in both dates, so nothing could be compared.",
        )]

    findings: List[Finding] = []
    if area.changed_pct < NEGLIGIBLE_CHANGE_PCT:
        findings.append(Finding(
            "headline",
            f"Little change detected: {area.changed_pct:.1f}% of the analysed area "
            f"({format_area(area.changed_ha)}). Either nothing significant happened "
            "here between these dates, or it falls outside the AOI.",
        ))
    elif preset_key in _HEADLINES:
        findings.extend(_HEADLINES[preset_key](codes, pixel_area_ha, area))
    else:
        findings.extend(_custom(codes, pixel_area_ha, area, index_name))

    if area.changed_pct >= NEGLIGIBLE_CHANGE_PCT:
        changed = (codes >= 0) & ~np.isin(codes, list(baseline_codes))
        where = change_location(changed)
        if where == "spread":
            findings.append(Finding("detail", "The change is spread across the AOI rather than concentrated in one part."))
        elif where:
            findings.append(Finding("detail", f"The change is concentrated in the {where} of the AOI."))

    if area.analysed_pct < LOW_COVERAGE_PCT:
        findings.append(Finding(
            "caveat",
            f"Only {area.analysed_pct:.0f}% of the AOI could be analysed (cloud, shadow "
            "or missing data in at least one date), so these areas are a lower bound.",
        ))
    findings.extend(
        Finding("caveat", n.summary) for n in notes if n.severity == "warning" and n.summary
    )
    return findings


def findings_markdown(findings: Sequence[Finding]) -> Dict[str, str]:
    """Split findings into markdown bullet lists keyed "summary" and "caveats"."""
    summary = [f"- {f.text}" for f in findings if f.kind != "caveat"]
    caveats = [f"- {f.text}" for f in findings if f.kind == "caveat"]
    return {"summary": "\n".join(summary), "caveats": "\n".join(caveats)}
