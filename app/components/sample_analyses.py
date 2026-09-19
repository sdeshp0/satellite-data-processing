"""
Curated "sample analyses" for the Change Detection page: real, well-known
before/after events with AOI + date ranges pre-configured, so a visitor can
see a dramatic result in one click without first understanding AOI
selection, date ranges, or event presets.

Scene selection within each sample's date range is automatic -- asking a
first-time visitor to also pick a scene manually would defeat the point of
a "one click" sample. The pair is chosen jointly (see
core.change.select_best_pair, shared with the manual Change Detection
page's automatic pre-selection) to favor a seasonally/illumination-matched
before/after pair, using combined cloud cover only as a tiebreaker --
picking each side's lowest-cloud scene independently (the original
approach) could pair a summer "before" with a winter "after" purely
because each happened to be the clearest scene in its own search window,
which is exactly the kind of mismatch core.change.comparability_checks
now warns about on the comparison page.

Coverage note: the wildfire/flood/deforestation samples in coastal,
deltaic, or storm-driven settings near the top of SAMPLE_ANALYSES are the
most likely to hit swath-edge nodata gaps or heavy cloud cover -- see the
"Added for more reliable coverage" section below for inland, dry-climate
(or otherwise less cloud-prone) alternatives, including several flood
samples added specifically for this reason after the original
Ganges-Brahmaputra Delta monsoon-flood sample was removed for its own
unreliable coverage.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple
import math

import streamlit as st
from shapely.geometry import box, Polygon

# Reusing the existing cached STAC search rather than duplicating a second
# cache for the same underlying query.
from app.components.scene_selector import _search_sentinel2_cached
from core.change import select_best_pair


# Progressive relaxation steps tried in order until a search returns
# results. Tried first at the original date window, then -- if still
# nothing -- retried at a widened window. Event-driven searches (a flood's
# "after" date, in particular) are disproportionately likely to hit cloudy
# weather right when you most want a clear scene, so failing after a
# single strict search wastes what's often a recoverable situation.
CLOUD_COVER_STEPS = (40, 60, 80, 100)
DATE_WINDOW_EXPANSIONS_DAYS = (0, 30)


def _search_with_fallback(
    aoi_wkt: str, start: date, end: date
) -> Tuple[List[Any], Optional[int], Optional[int]]:
    """
    Search with progressively relaxed cloud-cover thresholds; if still
    empty, widen the date window symmetrically and retry the same
    relaxation sequence.

    Returns
    -------
    tuple(list, int or None, int or None)
        (items, cloud_cover_used, expansion_days_used). The latter two are
        None only when items is empty (nothing found at any relaxation
        level tried).
    """
    for expand_days in DATE_WINDOW_EXPANSIONS_DAYS:
        window_start = start - timedelta(days=expand_days)
        window_end = end + timedelta(days=expand_days)

        for cloud_pct in CLOUD_COVER_STEPS:
            items = _search_sentinel2_cached(aoi_wkt, window_start, window_end, cloud_pct)
            if items:
                return items, cloud_pct, expand_days

    return [], None, None


SAMPLE_ANALYSES: Dict[str, Dict[str, Any]] = {
    "camp_fire_2018": {
        "label": "Camp Fire — Paradise, CA (Nov 2018)",
        "preset": "wildfire",
        "lat": 39.7596, "lon": -121.6219, "width_km": 25, "height_km": 25,
        "before_range": (date(2018, 9, 1), date(2018, 10, 31)),
        "after_range": (date(2018, 11, 25), date(2019, 1, 31)),
        "description": (
            "The Camp Fire ignited on November 8, 2018 and destroyed the "
            "town of Paradise, California within hours -- at the time, the "
            "deadliest and most destructive wildfire in California history. "
            "After-window starts Nov 25 (the fire's containment date) rather "
            "than Dec 1, widening the pool of candidate scenes -- an earlier "
            "coverage check found the automatically-picked Dec 1 scene had "
            "cloud concentrated specifically over this AOI despite a "
            "moderate scene-wide cloud percentage."
        ),
    },
    "kangaroo_island_2020": {
        "label": "Kangaroo Island Bushfires, Australia (Jan 2020)",
        "preset": "wildfire",
        "lat": -35.7751, "lon": 137.2131, "width_km": 60, "height_km": 60,
        "before_range": (date(2019, 11, 1), date(2019, 11, 30)),
        "after_range": (date(2020, 2, 1), date(2020, 2, 29)),
        "description": (
            "Bushfires during Australia's 2019-2020 \"Black Summer\" burned "
            "roughly half of Kangaroo Island, including large portions of "
            "Flinders Chase National Park."
        ),
    },
    "harvey_houston_2017": {
        "label": "Hurricane Harvey Flooding — Houston, TX (Aug 2017)",
        "preset": "flood",
        "lat": 29.7604, "lon": -95.3698, "width_km": 40, "height_km": 40,
        "before_range": (date(2017, 6, 1), date(2017, 7, 31)),
        "after_range": (date(2017, 8, 29), date(2017, 9, 30)),
        "description": (
            "Hurricane Harvey stalled over southeast Texas in late August "
            "2017, dropping historic rainfall and causing catastrophic "
            "flooding across the Houston metro area. Both windows widened "
            "from their original single-month span -- an earlier coverage "
            "check found real, substantial cloud on both sides (not a "
            "tiling issue), typical of the Gulf Coast humid subtropical "
            "climate even outside the storm itself; the wider windows give "
            "the search more candidate dates to find a clearer scene. This "
            "climate may simply not have a fully clear Sentinel-2 (optical) "
            "pair close to the event -- Sentinel-1 SAR, which sees through "
            "cloud, would be a more reliable fit for hurricane-flood "
            "comparisons specifically, if that's ever added."
        ),
    },
    "rondonia_deforestation": {
        "label": "Amazon Deforestation — Rondônia, Brazil (2019-2023)",
        "preset": "logging",
        "lat": -10.83, "lon": -62.90, "width_km": 40, "height_km": 40,
        "before_range": (date(2019, 6, 1), date(2019, 7, 31)),
        "after_range": (date(2023, 6, 1), date(2023, 7, 31)),
        "description": (
            "Rondônia sits on Brazil's \"arc of deforestation\" -- one of "
            "the most actively cleared regions of the Amazon, driven "
            "largely by cattle ranching and agricultural expansion. "
            "Comparing dry-season imagery a few years apart makes gradual "
            "clearing visible in a way a single before/after pair usually "
            "can't."
        ),
    },

    # --- Added for more reliable coverage ---
    # The four samples above are all coastal, deltaic, or storm-driven --
    # exactly the geography/weather combinations most likely to hit swath-
    # edge nodata gaps or heavy cloud cover. The seven below are
    # deliberately inland and dry-climate (or otherwise less cloud-prone)
    # instead -- Mediterranean, semi-arid, desert, or continental --
    # where fully-covered, cloud-free Sentinel-2 scenes are much easier to
    # come by. This includes three flood samples added specifically to
    # replace the original Ganges-Brahmaputra Delta monsoon-flood sample,
    # which was removed after its actual coverage (checked pixel-by-pixel
    # via core.change.coverage_overlap once loaded) turned out to be far
    # too low for a usable comparison -- exactly the coastal/deltaic/
    # monsoon risk this section exists to avoid.
    "dixie_fire_2021": {
        "label": "Dixie Fire — Plumas County, CA (Jul-Oct 2021)",
        "preset": "wildfire",
        "lat": 40.1401, "lon": -120.9438, "width_km": 45, "height_km": 45,
        "before_range": (date(2021, 5, 1), date(2021, 6, 30)),
        "after_range": (date(2021, 10, 1), date(2021, 11, 15)),
        "description": (
            "The Dixie Fire burned nearly 1,000,000 acres across Northern "
            "California's interior Sierra Nevada foothills between July and "
            "October 2021, destroying the town of Greenville -- at the "
            "time, the second-largest single wildfire in California "
            "history. Inland forested mountains with a dry Mediterranean "
            "summer climate, so both dates should have clear, well-covered "
            "scenes."
        ),
    },
    "mati_greece_2018": {
        "label": "Mati Wildfire, Greece (Jul 2018)",
        "preset": "wildfire",
        "lat": 38.0000, "lon": 24.0000, "width_km": 20, "height_km": 20,
        "before_range": (date(2018, 5, 1), date(2018, 6, 30)),
        "after_range": (date(2018, 8, 1), date(2018, 9, 15)),
        "description": (
            "A fast-moving wildfire struck the coastal town of Mati, near "
            "Athens, on July 23, 2018 -- one of the deadliest wildfires in "
            "modern Greek history. Mediterranean dry-summer climate makes "
            "clear scenes easy to find on both sides of the event."
        ),
    },
    "lake_mead_drought": {
        "label": "Lake Mead Water Level Decline, USA (2018-2022)",
        "preset": "flood",
        "lat": 36.0000, "lon": -114.7000, "width_km": 45, "height_km": 45,
        "before_range": (date(2018, 6, 1), date(2018, 7, 31)),
        "after_range": (date(2022, 6, 1), date(2022, 7, 31)),
        "description": (
            "A prolonged Southwest US drought pushed Lake Mead -- the "
            "country's largest reservoir by volume -- to its lowest water "
            "levels since it was first filled, exposing a dramatic "
            "\"bathtub ring\" around the shoreline. Uses the flood preset's "
            "\"Water Loss (Recession)\" class rather than new water. Desert "
            "climate: this is about as reliably cloud-free as Sentinel-2 "
            "imagery gets."
        ),
    },
    "gran_chaco_paraguay": {
        "label": "Gran Chaco Deforestation, Paraguay (2018-2022)",
        "preset": "logging",
        "lat": -22.3000, "lon": -60.3000, "width_km": 45, "height_km": 45,
        "before_range": (date(2018, 6, 1), date(2018, 8, 31)),
        "after_range": (date(2022, 6, 1), date(2022, 8, 31)),
        "description": (
            "Paraguay's Gran Chaco is one of the fastest-deforesting regions "
            "in the world, cleared largely for cattle ranching. Unlike "
            "humid Amazon rainforest, the Chaco is semi-arid tropical dry "
            "forest, with markedly less persistent cloud cover -- a better "
            "bet for consistently clean scenes than the Rondônia sample "
            "above."
        ),
    },
    "missouri_river_flood_2019": {
        "label": "Missouri River Flooding — Nebraska/Iowa, USA (Mar 2019)",
        "preset": "flood",
        "lat": 40.6708, "lon": -95.8608, "width_km": 45, "height_km": 45,
        "before_range": (date(2018, 9, 1), date(2018, 10, 31)),
        "after_range": (date(2019, 3, 15), date(2019, 4, 15)),
        "description": (
            "A mid-March 2019 \"bomb cyclone\" dropped heavy rain onto "
            "already-frozen, snow-covered ground across Nebraska and Iowa, "
            "triggering catastrophic, fast-moving flooding along the "
            "Missouri River -- among the costliest U.S. flood events on "
            "record at the time. Before-window moved to fall 2018 (post-"
            "harvest, pre-snow) rather than Jan-Feb 2019 -- an earlier "
            "coverage check found the original before-window was itself "
            "snow-covered, which Sentinel-2's own classification correctly "
            "flags as unusable (same as cloud), not a bug, but it meant "
            "almost the entire AOI was excluded on the before side. The "
            "wider seasonal gap to the after-window is a deliberate "
            "tradeoff to get a real, snow-free baseline."
        ),
    },
    "murray_darling_flood_2022": {
        "label": "Murray-Darling Basin Flooding — Central West NSW, Australia (Nov 2022)",
        "preset": "flood",
        "lat": -33.3833, "lon": 148.0000, "width_km": 45, "height_km": 45,
        "before_range": (date(2022, 3, 1), date(2022, 5, 31)),
        "after_range": (date(2022, 11, 1), date(2022, 11, 30)),
        "description": (
            "Repeated La Ni\u00f1a-driven rainfall through 2022 pushed the "
            "Lachlan and wider Murray-Darling river system to record "
            "levels, flooding towns across central west New South Wales "
            "in November 2022. Before-window moved to autumn (Mar-May) "
            "and widened to three months -- an earlier coverage check "
            "found real cloud (not tiling) concentrated over this AOI in "
            "the original June-July window even after the search's own "
            "cloud-relaxation fallback; NSW's winter frontal systems make "
            "that window a harder ask than the drier autumn shoulder "
            "season."
        ),
    },
    "pantanal_flood_pulse": {
        "label": "Pantanal Seasonal Flood Pulse, Brazil (Dry \u2192 Wet Season)",
        "preset": "flood",
        "lat": -17.7000, "lon": -57.6000, "width_km": 45, "height_km": 45,
        "before_range": (date(2021, 8, 1), date(2021, 9, 30)),
        "after_range": (date(2022, 1, 15), date(2022, 4, 15)),
        "description": (
            "The Pantanal, the world's largest tropical wetland, floods "
            "predictably every year as its rivers overflow during the wet "
            "season -- a natural, recurring \"flood pulse\" rather than a "
            "single disaster event, similar in spirit to the Bangladesh "
            "sample that was removed but in a tropical-savanna climate "
            "rather than a full monsoon delta, which keeps cloud cover "
            "considerably more manageable. After-window widened to three "
            "months (was six weeks) -- an earlier coverage check found "
            "real wet-season cloud over roughly a third of the AOI in the "
            "narrower window; wet season is inherently the harder half of "
            "this comparison to keep clear, so the wider window trades a "
            "looser seasonal match for more candidate dates to find one."
        ),
    },
}



def _build_aoi(lat: float, lon: float, width_km: float, height_km: float) -> Polygon:
    dlat = (height_km / 2) / 111.0
    dlon = (width_km / 2) / (111.0 * abs(math.cos(math.radians(lat))))
    return box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _run_sample_analysis(key: str) -> None:
    """
    Populate session_state for a sample analysis: AOI, event preset,
    before/after date ranges, and an automatically-selected scene pair
    (see core.change.select_best_pair) for each date range.

    Caller is responsible for calling st.rerun() immediately afterward, so
    that the date_input / selectbox widgets on the page -- which read their
    initial value from session_state[key] if already present -- pick up
    these new values on the next run instead of their hardcoded defaults.
    """
    sample = SAMPLE_ANALYSES[key]

    aoi = _build_aoi(sample["lat"], sample["lon"], sample["width_km"], sample["height_km"])
    st.session_state["aoi"] = aoi
    st.session_state["aoi_label"] = sample["label"]
    st.session_state["change_preset"] = sample["preset"]

    before_start, before_end = sample["before_range"]
    after_start, after_end = sample["after_range"]
    st.session_state["before_start_date"] = before_start
    st.session_state["before_end_date"] = before_end
    st.session_state["after_start_date"] = after_start
    st.session_state["after_end_date"] = after_end

    with st.spinner("Finding before/after scenes for this sample..."):
        before_items, before_cloud, before_expand = _search_with_fallback(
            aoi.wkt, before_start, before_end
        )
        after_items, after_cloud, after_expand = _search_with_fallback(
            aoi.wkt, after_start, after_end
        )

    st.session_state["before_stac_items"] = before_items
    st.session_state["after_stac_items"] = after_items

    if not before_items or not after_items:
        missing = []
        if not before_items:
            missing.append("before")
        if not after_items:
            missing.append("after")
        st.session_state["sample_analysis_error"] = (
            f"Couldn't find any scenes for the {', '.join(missing)} window(s), "
            "even after relaxing the cloud-cover filter and widening the date "
            "range. Try picking scenes manually below with a looser cloud "
            "cover slider, or run the sample again later."
        )
        st.session_state.pop("before_stac_item", None)
        st.session_state.pop("after_stac_item", None)
        return

    best_before, best_after = select_best_pair(before_items, after_items)
    st.session_state["before_stac_item"] = best_before
    st.session_state["after_stac_item"] = best_after
    st.session_state["sample_analysis_error"] = None

    # Let the user know if either side needed relaxed cloud cover or a
    # widened window -- worth knowing, since a heavily relaxed cloud filter
    # can mean a noticeably cloudier scene than the default 40% would give.
    # Note this reflects what the SEARCH needed to return any results at
    # all, not the cloud cover of the specific pair ultimately chosen --
    # any comparability concerns about the chosen pair itself (seasonal
    # distance, sun elevation, UTM/tile/platform) surface separately via
    # core.change.comparability_checks on the comparison page below.
    notes = []
    if before_cloud > 40 or before_expand > 0:
        notes.append(
            f"Before: used cloud cover < {before_cloud}%"
            + (f", widened by {before_expand} days" if before_expand else "")
        )
    if after_cloud > 40 or after_expand > 0:
        notes.append(
            f"After: used cloud cover < {after_cloud}%"
            + (f", widened by {after_expand} days" if after_expand else "")
        )
    st.session_state["sample_analysis_relaxation_note"] = "; ".join(notes) if notes else None


def sample_analysis_picker() -> None:
    """
    Render the "Try a Sample Analysis" section: a picker plus a button that
    runs _run_sample_analysis and reruns the page so every downstream
    widget (event type, date ranges, scene pickers, comparison) reflects
    the choice immediately.
    """
    st.subheader("Try a Sample Analysis")
    st.caption(
        "Real, well-documented before/after events, pre-configured end to "
        "end -- AOI, dates, and event type all set automatically. The "
        "before/after scene pair is chosen to be seasonally and "
        "illumination-matched, not just individually low-cloud."
    )

    options = ["-- Select a sample --"] + list(SAMPLE_ANALYSES.keys())
    choice = st.selectbox(
        "Sample event",
        options=options,
        format_func=lambda k: SAMPLE_ANALYSES[k]["label"] if k in SAMPLE_ANALYSES else k,
        key="sample_analysis_choice",
    )

    if choice in SAMPLE_ANALYSES:
        st.caption(SAMPLE_ANALYSES[choice]["description"])
        if st.button("Run this analysis", key="run_sample_analysis"):
            _run_sample_analysis(choice)
            st.rerun()

    error = st.session_state.get("sample_analysis_error")
    if error:
        st.warning(error)

    note = st.session_state.get("sample_analysis_relaxation_note")
    if note:
        st.info(f"Note: the strict search was empty, so this used a fallback. {note}")
