"""
Change & recovery over time: the event's index, month by month, inside the
area Change Detection flagged and on nearby unchanged land.

Comparing the two answers two questions a single before/after pair can't:

- Was it the event or the season? If unchanged land moved the same way
  over the same dates, much of the "change" is seasonal or regional.
- Is it recovering? The gap between the affected area and unchanged land
  is measured before the event, at the event, and on the latest scene; the
  share of the event's gap that has since closed is the recovery. Using
  the gap rather than the raw index cancels most of the seasonal swing,
  since both areas see the same weather and sun.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.findings import Finding


@dataclass(frozen=True)
class RecoveryPreset:
    affected_codes: tuple  # change classes that make up the affected area
    control_code: int  # the no-change class used as the comparison
    area_name: str
    control_name: str
    event_noun: str
    months_before: int
    months_after: int
    step_months: int


# Burn recovery and regrowth take years and are seasonal, so they're
# sampled quarterly over a long window; floods recede in weeks to months.
RECOVERY_PRESETS: Dict[str, RecoveryPreset] = {
    "wildfire": RecoveryPreset((3, 4, 5, 6), 2, "the burned area", "nearby unburned land", "the fire", 12, 36, 3),
    "flood": RecoveryPreset((2,), 0, "the newly flooded area", "nearby dry land", "the flood", 3, 9, 1),
    "logging": RecoveryPreset((2,), 1, "the vegetation-loss area", "nearby stable vegetation", "the clearing", 12, 24, 3),
}

# A scene only counts if at least this share of the affected area is clear.
MIN_CLEAR_SHARE = 0.3
# Below this event gap (index units) the affected area didn't stand out
# from unchanged land, so there is nothing to measure recovery against.
MIN_EVENT_GAP = 0.05


def add_months(when: date, months: int) -> date:
    """`when` moved by whole months, clamped to the 28th so every month has the day."""
    total = when.year * 12 + when.month - 1 + months
    return date(total // 12, total % 12 + 1, min(when.day, 28))


def _months_between(start: date, when: date) -> int:
    return (when.year - start.year) * 12 + when.month - start.month


def pick_scenes(items: Sequence[Any], start: date, step_months: int, keep: Sequence[Any] = ()) -> List[Any]:
    """
    One scene per `step_months` period from `start`, the least cloudy one,
    plus every scene in `keep` (the before/after pair), which replace the
    pick for their periods. Sorted by date.
    """
    def period(item: Any) -> int:
        return _months_between(start, item.datetime.date()) // step_months

    def cloud(item: Any) -> float:
        return item.properties.get("eo:cloud_cover", 100)

    kept_periods = {period(item) for item in keep}
    chosen: Dict[int, Any] = {}
    for item in items:
        p = period(item)
        if p not in kept_periods and (p not in chosen or cloud(item) < cloud(chosen[p])):
            chosen[p] = item
    return sorted(list(chosen.values()) + list(keep), key=lambda item: item.datetime)


@dataclass
class SamplePoint:
    when: date
    affected: Optional[float]  # median index over the clear affected area; None if too cloudy
    control: Optional[float]
    clear_share: float  # share of the affected area that was clear

    @property
    def gap(self) -> Optional[float]:
        if self.affected is None or self.control is None:
            return None
        return self.affected - self.control


def sample_point(
    index: np.ndarray, valid: np.ndarray, affected: np.ndarray, control: np.ndarray, when: date,
    min_clear_share: float = MIN_CLEAR_SHARE,
) -> SamplePoint:
    """Median index over the clear part of each area on one date."""
    usable = valid & np.isfinite(index)
    n_affected = int(affected.sum())
    clear = affected & usable
    share = float(clear.sum()) / n_affected if n_affected else 0.0
    control_clear = control & usable
    if share < min_clear_share or not control_clear.any():
        return SamplePoint(when, None, None, share)
    return SamplePoint(when, float(np.median(index[clear])), float(np.median(index[control_clear])), share)


@dataclass
class RecoverySummary:
    before: SamplePoint
    after: SamplePoint
    latest: Optional[SamplePoint]  # last clear scene after the event's after date
    baseline_gap: float  # median gap over the clear scenes up to the before date
    recovery_pct: Optional[float]  # None when the event gap is too small or nothing came after

    @property
    def event_shift(self) -> float:
        return self.after.affected - self.before.affected

    @property
    def control_shift(self) -> float:
        return self.after.control - self.before.control


def recovery_summary(points: Sequence[SamplePoint], before_date: date, after_date: date) -> Optional[RecoverySummary]:
    """
    The event's change and the recovery since, from clear sample points.
    None if there is no clear point at or before `before_date` and at or
    after `after_date`.
    """
    clear = [p for p in points if p.gap is not None]
    pre = [p for p in clear if p.when <= before_date]
    post = [p for p in clear if p.when >= after_date]
    if not pre or not post:
        return None
    before, after = pre[-1], post[0]
    baseline_gap = float(np.median([p.gap for p in pre]))
    latest = post[-1] if len(post) > 1 else None
    event_drop = baseline_gap - after.gap
    recovery = None
    if latest is not None and abs(event_drop) >= MIN_EVENT_GAP:
        recovery = 100 * (latest.gap - after.gap) / event_drop
    return RecoverySummary(before, after, latest, baseline_gap, recovery)


def _elapsed(start: date, end: date) -> str:
    months = _months_between(start, end)
    if months >= 24:
        return f"{months / 12:.1f} years"
    return f"{months} month{'s' if months != 1 else ''}"


def recovery_findings(summary: RecoverySummary, preset: RecoveryPreset, index_name: str) -> List[Finding]:
    """Plain-language lines for the change and recovery."""
    name = index_name.upper()
    shift, control = summary.event_shift, summary.control_shift
    text = (
        f"Between {summary.before.when} and {summary.after.when}, the {name} of {preset.area_name} "
        f"{'rose' if shift > 0 else 'fell'} by {abs(shift):.2f} while {preset.control_name} moved {control:+.2f}"
    )
    share = control / shift if shift else 0.0
    if share >= 0.5:
        text += ", so much of this change is seasonal or regional, not the event."
    elif share >= 0.25:
        text += ", so part of this change may be seasonal."
    else:
        text += ", so the change comes from the event, not the season."
    out = [Finding("headline", text)]

    if summary.latest is None:
        out.append(Finding("caveat", "No clear scene after the event's after date yet, so recovery can't be measured."))
    elif summary.recovery_pct is None:
        out.append(Finding(
            "caveat",
            f"{preset.area_name[0].upper()}{preset.area_name[1:]} barely stood out from {preset.control_name} "
            "at the event, so there's no clear gap to measure recovery against.",
        ))
    else:
        pct = summary.recovery_pct
        if pct >= 85:
            state = "close to or past its pre-event state"
        elif pct >= 15:
            state = "a partial recovery"
        else:
            state = "little recovery so far"
        out.append(Finding(
            "headline",
            f"By {summary.latest.when}, {_elapsed(summary.after.when, summary.latest.when)} after "
            f"{preset.event_noun}, {preset.area_name} is about {max(pct, 0):.0f}% of the way back to its "
            f"pre-event {name}, measured against {preset.control_name} ({state}).",
        ))
    return out
