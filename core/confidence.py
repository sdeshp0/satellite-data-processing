"""
How far to trust a change-detection result: the changed area re-measured at
nearby classification thresholds, and a High / Medium / Low rating that
combines that spread with data coverage and the pair's comparability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from core.change import ChangeAreaSummary, ComparabilityNote, change_area_summary

# Relative spread of the changed area ((high - low) / central) above which
# the estimate is treated as threshold-sensitive.
SPREAD_MEDIUM = 0.3
SPREAD_HIGH = 0.6
# Analysed share of the AOI window below which coverage limits confidence.
COVERAGE_MEDIUM = 90.0
COVERAGE_LOW = 60.0


@dataclass
class AreaRange:
    low_ha: float
    central_ha: float
    high_ha: float

    @property
    def spread(self) -> float:
        """(high - low) / central, or 0 when nothing changed."""
        if self.central_ha <= 0:
            return 0.0 if self.high_ha <= 0 else float("inf")
        return (self.high_ha - self.low_ha) / self.central_ha


@dataclass
class Confidence:
    rating: str                      # "High", "Medium" or "Low"
    reasons: List[str] = field(default_factory=list)


def change_area_range(
    preset: Dict,
    index_before: np.ndarray,
    index_after: np.ndarray,
    central: ChangeAreaSummary,
    pixel_area_ha: float,
) -> AreaRange:
    """
    Re-run the preset's classifier at each of its "sensitivity" settings and
    return the lowest, central and highest changed area.
    """
    areas = [central.changed_ha]
    keyword, values = preset.get("sensitivity", (None, []))
    for value in values:
        codes = preset["classify"](index_before, index_after, **{keyword: value})
        areas.append(change_area_summary(codes, preset["baseline_codes"], pixel_area_ha).changed_ha)
    return AreaRange(low_ha=min(areas), central_ha=central.changed_ha, high_ha=max(areas))


def assess_confidence(
    area: ChangeAreaSummary,
    area_range: AreaRange,
    notes: Sequence[ComparabilityNote] = (),
) -> Confidence:
    """
    Start at High and step down one level for each issue: partial coverage
    (two steps below COVERAGE_LOW), a threshold-sensitive area (two steps
    above SPREAD_HIGH), and each comparability warning (at most two).
    """
    penalty = 0
    reasons: List[str] = []

    if area.analysed_pct < COVERAGE_LOW:
        penalty += 2
        reasons.append(f"only {area.analysed_pct:.0f}% of the AOI could be analysed")
    elif area.analysed_pct < COVERAGE_MEDIUM:
        penalty += 1
        reasons.append(f"{area.analysed_pct:.0f}% of the AOI could be analysed")

    spread = area_range.spread
    if spread > SPREAD_HIGH:
        penalty += 2
        reasons.append("the changed area is very sensitive to the classification threshold")
    elif spread > SPREAD_MEDIUM:
        penalty += 1
        reasons.append("the changed area shifts noticeably with the classification threshold")

    warnings = [n for n in notes if n.severity == "warning"]
    if warnings:
        penalty += min(2, len(warnings))
        reasons.append(
            "the two scenes aren't fully comparable"
            + (f" ({len(warnings)} warnings)" if len(warnings) > 1 else "")
        )

    rating = "High" if penalty == 0 else "Medium" if penalty == 1 else "Low"
    if not reasons:
        reasons.append("full coverage, a stable estimate and a well-matched scene pair")
    return Confidence(rating=rating, reasons=reasons)

