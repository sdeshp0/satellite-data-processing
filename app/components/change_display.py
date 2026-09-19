"""
Before/after change detection display: loads two scenes for the same AOI,
aligns them to a common grid, computes the event-appropriate index and
classification, and renders the comparison.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Tuple

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
from rasterio.io import MemoryFile
from shapely.geometry import Polygon

from app.components.scene_loader import load_scene_cached
from app.components.index_display import CATEGORY_STYLE, INDEX_INFO
from core.utils import resample_bands, scale_bands
from core.indices import compute_indices
from core.viz import to_rgb
from core.change import (
    EVENT_PRESETS,
    align_bands,
    compute_delta,
    class_breakdown,
    comparability_checks,
    coverage_overlap,
    COVERAGE_BOTH_CODE,
    COVERAGE_LABELS,
    COVERAGE_COLORS,
)


ALL_INDEX_NAMES = ["ndvi", "evi", "savi", "nbr", "ndmi", "ndwi", "ndbi", "bsi"]

# Assumed rendered width (CSS px) used to pick a fixed iframe height for
# the swipe-compare component -- see _render_before_after_slider. Chosen
# to roughly match this app's typical main-content width under
# layout="wide"; on a much narrower viewport (e.g. mobile) the image is
# still fully visible and responsive in *width* (object-fit: cover fills
# the box either way), just not perfectly aspect-correct.
_SLIDER_ASSUMED_WIDTH_PX = 800


def _array_to_geotiff_bytes(arr: np.ndarray, reference) -> bytes:
    """
    Package a 2D array as a single-band, georeferenced GeoTIFF using a
    reference band's CRS/transform (matches the pattern in index_display.py).
    """
    transform = reference.rio.transform()
    crs = reference.rio.crs
    height, width = arr.shape
    data = arr.astype("float32")

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=height, width=width, count=1,
            dtype="float32", crs=crs, transform=transform, nodata=np.nan,
        ) as dst:
            dst.write(data, 1)
        return memfile.read()


def _plot_classification(codes: np.ndarray, labels, colors, title: str):
    """
    Render a discrete classification map with a matching legend, sized and
    laid out the same way as the delta map next to it (see the
    fig.colorbar(..., shrink=0.7) call in change_display) -- a discrete
    colorbar with one tick+label per class, rather than a separate legend
    block below the axes. The old below-plot legend's height grew with the
    number of classes (3 for flood/logging, 7 for dNBR), which made the
    two side-by-side map images end up different sizes/aspect ratios
    depending on the event preset; a colorbar occupies a fixed-shape strip
    beside the axes regardless of class count, matching the delta map's
    layout exactly. codes < 0 (masked/invalid pixels) are shown as
    transparent.
    """
    ordered = sorted(labels.keys())
    cmap = ListedColormap([colors[c] for c in ordered])
    bounds = [c - 0.5 for c in ordered] + [ordered[-1] + 0.5]
    norm = BoundaryNorm(bounds, cmap.N)
    display = np.ma.masked_where(codes < 0, codes)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(display, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, ticks=ordered)
    cbar.ax.set_yticklabels([labels[c] for c in ordered], fontsize=7)
    return fig


def _index_style_for(name: str) -> Tuple[str, float, float]:
    """
    Resolve (cmap, vmin, vmax) for an index, mirroring
    app.components.index_display's per-index styling so the swipe
    comparison uses the same color convention as the static index charts
    elsewhere in the app (e.g. NBR: green = healthy, red = more burn
    signal). Falls back to the vegetation style if the index has no
    category mapped.
    """
    category = INDEX_INFO.get(name, {}).get("category", "vegetation")
    style = CATEGORY_STYLE.get(category, CATEGORY_STYLE["vegetation"])
    return style["cmap"], style["vmin"], style["vmax"]


def _rgb_array_to_png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 RGB array as PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _index_array_to_png_bytes(arr: np.ndarray, cmap_name: str, vmin: float, vmax: float) -> bytes:
    """
    Colorize a 2D index array into an RGBA PNG using the given cmap/vmin/vmax
    (see _index_style_for). NaN pixels -- outside the "Both Dates" coverage
    class per core.change.coverage_overlap -- are rendered fully
    transparent rather than a solid color, so they read as "no data" in the
    swipe comparison rather than a spurious extreme value.
    """
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(cmap_name)
    nan_mask = np.isnan(arr)
    filled = np.where(nan_mask, vmin, arr)
    rgba = (cmap(norm(filled)) * 255).astype("uint8")
    rgba[nan_mask, 3] = 0
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _to_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _render_before_after_slider(
    before_png: bytes,
    after_png: bytes,
    array_shape: Tuple[int, int],
    key: str,
    before_label: str,
    after_label: str,
) -> None:
    """
    Render an interactive swipe/wipe before-after image comparison: a
    self-contained HTML/CSS/JS snippet via st.components.v1.html, with no
    external JS dependency -- unlike, e.g., Nominatim geocoding elsewhere
    in this app, this needs no outbound network access, so it behaves
    identically on Streamlit Community Cloud and locally.

    Dragging the handle reveals more/less of the "before" image layered
    over the "after" image. This complements, rather than replaces, the
    static side-by-side RGB and the delta/classification maps elsewhere in
    this component: the slider is best for an at-a-glance "does this look
    different", the delta/classification maps give the quantified answer.

    Parameters
    ----------
    before_png, after_png : bytes
        PNG-encoded images with the same pixel dimensions (both scenes are
        aligned to a common grid earlier in change_display, so this holds
        for both the RGB and index layers).
    array_shape : tuple(int, int)
        (height, width) of the source array, used only to pick a
        reasonable fixed iframe height (see _SLIDER_ASSUMED_WIDTH_PX) --
        st.components.v1.html doesn't auto-size to its content.
    key : str
        Unique DOM id for this instance, so more than one slider could
        exist on a page without their JS colliding.
    before_label, after_label : str
        Small text labels overlaid in the top corners.
    """
    height, width = array_shape
    height_px = max(200, round(_SLIDER_ASSUMED_WIDTH_PX * height / max(width, 1)))

    before_uri = _to_data_uri(before_png)
    after_uri = _to_data_uri(after_png)

    html = f"""
<div id="{key}" style="position:relative;width:100%;height:{height_px}px;
     overflow:hidden;border-radius:6px;background:#111;user-select:none;
     font-family:sans-serif;">
  <img src="{after_uri}" style="position:absolute;top:0;left:0;
       width:100%;height:100%;object-fit:cover;display:block;">
  <div class="ba-before-wrap" style="position:absolute;top:0;left:0;
       width:50%;height:100%;overflow:hidden;">
    <img class="ba-before-img" src="{before_uri}" style="position:absolute;
         top:0;left:0;height:100%;object-fit:cover;display:block;">
  </div>
  <div class="ba-handle" style="position:absolute;top:50%;left:50%;
       transform:translate(-50%,-50%);width:34px;height:34px;
       border-radius:50%;background:rgba(255,255,255,0.95);
       box-shadow:0 1px 4px rgba(0,0,0,0.5);display:flex;
       align-items:center;justify-content:center;font-size:13px;
       color:#333;pointer-events:none;">&#8596;</div>
  <div style="position:absolute;top:8px;left:8px;
       background:rgba(0,0,0,0.6);color:#fff;padding:2px 8px;
       border-radius:4px;font-size:12px;">{before_label}</div>
  <div style="position:absolute;top:8px;right:8px;
       background:rgba(0,0,0,0.6);color:#fff;padding:2px 8px;
       border-radius:4px;font-size:12px;">{after_label}</div>
  <input type="range" min="0" max="100" value="50" style="position:absolute;
       top:0;left:0;width:100%;height:100%;margin:0;opacity:0;
       cursor:ew-resize;">
</div>
<script>
(function() {{
  var container = document.getElementById("{key}");
  var range = container.querySelector("input[type=range]");
  var wrap = container.querySelector(".ba-before-wrap");
  var beforeImg = container.querySelector(".ba-before-img");
  var handle = container.querySelector(".ba-handle");

  function sizeBeforeImg() {{
    // The before-image must render at the FULL container width (not the
    // wrap's clipped width) so the overlap lines up pixel-for-pixel with
    // the after-image underneath as the wrap is resized.
    beforeImg.style.width = container.clientWidth + "px";
  }}
  function update(val) {{
    wrap.style.width = val + "%";
    handle.style.left = val + "%";
  }}

  range.addEventListener("input", function(e) {{ update(e.target.value); }});
  window.addEventListener("resize", sizeBeforeImg);
  sizeBeforeImg();
  update(range.value);
}})();
</script>
"""
    components.html(html, height=height_px + 4, scrolling=False)


def change_display(
    before_item: Any,
    after_item: Any,
    aoi: Polygon,
    preset_key: str,
    key_prefix: str = "change",
    preview_max_dim: int = 1024,
) -> None:
    """
    Load, align, and compare two Sentinel-2 scenes over the same AOI.

    Parameters
    ----------
    before_item, after_item : pystac.Item
        The two scenes to compare.
    aoi : shapely.geometry.Polygon
        Area of interest.
    preset_key : str
        Key into core.change.EVENT_PRESETS selecting the index, delta
        direction, and classification methodology to use.
    key_prefix : str
        Prefix for widget/session_state keys on this page.
    preview_max_dim : int
        Passed through to the scene loader; caps read resolution.

    Returns
    -------
    None
        Results are displayed directly in Streamlit.
    """
    preset = EVENT_PRESETS[preset_key]

    st.subheader(preset["label"])
    st.caption(preset["methodology"])

    # --- Index selection (custom preset only) ---
    if preset["index"] is None:
        index_name = st.selectbox(
            "Index to compare",
            options=ALL_INDEX_NAMES,
            index=0,
            key=f"{key_prefix}_custom_index",
        )
    else:
        index_name = preset["index"]

    # --- Multi-tile mosaic toggle (TEMPORARY -- for measuring cost) ---
    # See core/load.py's module docstring for why this exists: an AOI near
    # a Sentinel-2 tile boundary can have real coverage gaps that no choice
    # of date fixes, which load_scene now closes by fetching companion
    # tiles from the same acquisition date. That's real added network cost
    # only when it's actually needed, but "only when needed" should be
    # measured, not assumed -- this checkbox plus the "Load cost" expander
    # below make that cost visible and A/B-able directly in the app,
    # without having to go through check_sample_coverage.py's --no-mosaic
    # flag. Safe to remove this checkbox (and always pass enable_mosaic=
    # True) once the cost is well understood and no longer needs watching.
    mosaic_enabled = st.checkbox(
        "Enable multi-tile mosaicking (temporary -- for measuring cost)",
        value=True,
        key=f"{key_prefix}_mosaic_enabled",
        help=(
            "When an AOI straddles a Sentinel-2 tile boundary, fills the "
            "gap using other tiles from the same acquisition date. Turn "
            "this off to see the old single-tile-only coverage/cost for "
            "comparison -- see the 'Load cost' section below once scenes "
            "are loaded."
        ),
    )

    # --- Load both scenes ---
    with st.spinner("Loading before/after scenes..."):
        bands_before, before_coverage, before_load_info = load_scene_cached(
            before_item, aoi.wkt, before_item.id, preview_max_dim,
            enable_mosaic=mosaic_enabled,
        )
        bands_after, after_coverage, after_load_info = load_scene_cached(
            after_item, aoi.wkt, after_item.id, preview_max_dim,
            enable_mosaic=mosaic_enabled,
        )

        bands_before = scale_bands(resample_bands(bands_before))
        bands_after = scale_bands(resample_bands(bands_after))

        # Align "after" onto "before"'s grid HERE, before anything (RGB or
        # indices) is derived from it. Two independently loaded scenes over
        # the "same" AOI can come from different UTM zones and therefore
        # different pixel dimensions -- aligning only a derived index (as an
        # earlier version did) left the RGB preview unaligned, which is why
        # one image could look stretched relative to the other.
        bands_after = align_bands(bands_after, bands_before["nir"])

    # --- Load cost (TEMPORARY diagnostic -- see the checkbox above) ---
    # Surfaces core.load.load_scene's load_info for both scenes: how many
    # tiles were actually fetched, how much that improved coverage, and how
    # much time the mosaic machinery itself took, specifically so the added
    # cost of this feature can be measured rather than assumed. Safe to
    # remove this whole expander once that's well understood.
    with st.expander("\u23f1\ufe0f Load cost (temporary diagnostic)"):
        cost_col1, cost_col2 = st.columns(2)
        for label, info in (("Before", before_load_info), ("After", after_load_info)):
            col = cost_col1 if label == "Before" else cost_col2
            with col:
                st.markdown(f"**{label}**")
                st.write(f"Tiles used: {info['tiles_used']}")
                st.write(
                    f"Coverage: {info['primary_geom_coverage'] * 100:.1f}% "
                    f"\u2192 {info['final_coverage'] * 100:.1f}%"
                    if info["mosaic_attempted"] else
                    f"Coverage: {info['primary_geom_coverage'] * 100:.1f}% (single tile, no mosaic needed)"
                )
                if info["mosaic_attempted"]:
                    st.write(f"Companions found / used: {info['companions_found']} / {info['companions_used']}")
                    st.write(f"Companion search time: {info['companion_search_time_s']:.2f}s")
                    st.write(f"Companion read time: {info['companion_read_time_s']:.2f}s")
                st.write(f"Total load time: {info['total_load_time_s']:.2f}s")
        st.caption(
            "Temporary instrumentation for evaluating the multi-tile "
            "mosaic feature's cost -- toggle the checkbox above to compare "
            "against single-tile-only loading."
        )

    # --- Data coverage check (per scene) ---
    # Sentinel-2 granules aren't always fully covered by real data at swath
    # edges. If the AOI mostly misses one scene's actual footprint, that
    # scene renders as mostly black/empty AND -- before the boundless-read
    # fix in core/load.py -- could produce genuinely corrupted-looking
    # output. Coverage alone no longer causes corruption, but a low value
    # still means real data is missing for a meaningful chunk of the AOI,
    # so it's still worth flagging as a reason to pick a different scene.
    #
    # This is a single geometric ratio per scene (see core.load.load_scene),
    # computed before reading -- cheap, and a good early signal, but it
    # can't say *where* a scene's gaps are or whether they land in the same
    # place as the other date's gaps. That's what the joint coverage-
    # overlap check right below this one is for.
    if before_coverage < 0.9 or after_coverage < 0.9:
        low = []
        if before_coverage < 0.9:
            low.append(f"before (~{before_coverage * 100:.0f}% coverage)")
        if after_coverage < 0.9:
            low.append(f"after (~{after_coverage * 100:.0f}% coverage)")
        st.warning(
            f"The AOI only partially falls within the real data footprint "
            f"of the {' and '.join(low)} scene. Consider picking a "
            "different scene for a cleaner comparison."
        )

    # --- Data coverage overlap (joint, pixel-level) ---
    # Two scenes can each individually look fine above (e.g. both ~95%
    # covered) while still covering DIFFERENT parts of the AOI -- before
    # missing the NW corner, after missing the SE corner. What actually
    # feeds the comparison below is the pixel-wise overlap of usable
    # (cloud-free, real-data) pixels in BOTH dates -- see
    # core.change.coverage_overlap, which also fixes a real gap in the
    # previous version of this check: it now excludes SCL's "No Data"
    # class, which is also what boundless-read fill uses for AOI pixels
    # outside a scene's real footprint. Previously those fill pixels were
    # silently treated as valid (near-zero-reflectance) data.
    coverage_codes, coverage_pct = coverage_overlap(bands_before["scl"], bands_after["scl"])

    mismatch_pct = coverage_pct["before_only"] + coverage_pct["after_only"]
    if mismatch_pct >= 5.0:
        st.warning(
            f"{mismatch_pct:.0f}% of the AOI has usable data in only ONE of "
            f"the two dates ({coverage_pct['before_only']:.0f}% before-only, "
            f"{coverage_pct['after_only']:.0f}% after-only). These pixels are "
            "excluded from the comparison below -- only "
            f"{coverage_pct['both']:.0f}% of the AOI has usable data in "
            "BOTH dates and actually feeds the delta/classification/swipe "
            "views. See the coverage map below for where the gaps fall."
        )
    elif coverage_pct["neither"] >= 10.0:
        st.caption(
            f"{coverage_pct['neither']:.0f}% of the AOI has no usable data "
            "in EITHER date (cloud, shadow, or outside both scenes' real "
            "footprint) and is excluded from the comparison below."
        )

    with st.expander("Coverage overlap map"):
        fig_coverage = _plot_classification(
            coverage_codes, COVERAGE_LABELS, COVERAGE_COLORS, "Data Coverage Overlap"
        )
        st.pyplot(fig_coverage)
        st.caption(
            "Where each date has usable (cloud-free, real-data) pixels. "
            "Only the green \u201cBoth Dates\u201d area feeds the delta, "
            "classification, and swipe-compare views below."
        )

    # --- Comparability checks: are these two scenes really apples-to-apples? ---
    # Covers UTM zone, MGRS tile, platform, seasonal (day-of-year) distance,
    # and sun elevation -- see core.change.comparability_checks. Consolidated
    # there (rather than inline here) so the same logic can also score
    # candidate pairs before one is picked; see scene_selector.scene_picker's
    # compare_item option and sample_analyses.py's automatic pair selection.
    for note in comparability_checks(before_item, after_item):
        if note.severity == "warning":
            st.warning(note.message)
        else:
            st.caption(note.message)

    # --- Index per date ---
    # Computed here (ahead of the RGB/swipe sections below) since the swipe
    # comparison can show either the RGB or the colorized index layer, and
    # both need to be ready before that section renders.
    index_before = compute_indices(bands_before)[index_name]
    index_after = compute_indices(bands_after)[index_name]

    # --- Apply the joint coverage mask: a pixel only counts if usable in
    # BOTH dates. Reuses coverage_codes computed above (rather than a
    # separate combined_valid_mask call) so the mask applied here is
    # exactly the "Both Dates" class shown in the coverage map. ---
    valid = coverage_codes == COVERAGE_BOTH_CODE
    index_before = np.where(valid, index_before, np.nan)
    index_after = np.where(valid, index_after, np.nan)

    # --- RGB previews, side by side ---
    rgb_before = to_rgb(bands_before)
    rgb_after = to_rgb(bands_after)

    rgb_col1, rgb_col2 = st.columns(2)
    with rgb_col1:
        st.image(
            rgb_before,
            caption=f"Before — {before_item.datetime.date()}",
            width="stretch",
        )
    with rgb_col2:
        st.image(
            rgb_after,
            caption=f"After — {after_item.datetime.date()}",
            width="stretch",
        )

    # --- Before/After swipe comparison ---
    st.subheader("Before / After (Swipe Compare)")

    swipe_layer = st.radio(
        "Layer to compare",
        options=["RGB", index_name.upper()],
        horizontal=True,
        key=f"{key_prefix}_swipe_layer",
    )

    if swipe_layer == "RGB":
        before_png = _rgb_array_to_png_bytes(rgb_before)
        after_png = _rgb_array_to_png_bytes(rgb_after)
    else:
        cmap_name, vmin, vmax = _index_style_for(index_name)
        before_png = _index_array_to_png_bytes(index_before, cmap_name, vmin, vmax)
        after_png = _index_array_to_png_bytes(index_after, cmap_name, vmin, vmax)

    _render_before_after_slider(
        before_png,
        after_png,
        array_shape=index_before.shape,
        key=f"{key_prefix}_baslider",
        before_label=f"Before — {before_item.datetime.date()}",
        after_label=f"After — {after_item.datetime.date()}",
    )
    st.caption("Drag the handle to reveal more of the before/after image.")

    # --- Delta + classification ---
    delta = compute_delta(index_before, index_after, preset["direction"])
    codes = preset["classify"](index_before, index_after)

    # --- Summary ---
    st.subheader("Change Summary")

    breakdown_df = class_breakdown(codes, preset["labels"])
    baseline = set(preset["baseline_codes"])
    valid_codes = codes[codes >= 0]
    total_valid = valid_codes.size
    changed_count = int(np.sum(~np.isin(valid_codes, list(baseline)))) if total_valid else 0
    changed_pct = round(100 * changed_count / total_valid, 1) if total_valid else 0.0
    mean_delta = float(np.nanmean(delta)) if total_valid else float("nan")

    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.metric("Area with detected change", f"{changed_pct}%")
    with stat_col2:
        st.metric(f"Mean \u0394{index_name.upper()} over AOI", f"{mean_delta:.3f}")

    if not breakdown_df.empty:
        st.dataframe(breakdown_df, hide_index=True, width="stretch")
        st.bar_chart(breakdown_df.set_index("Class")["Percent"])
    else:
        st.caption("No valid (cloud-free in both dates) pixels to summarize.")

    # --- Delta + classification maps ---
    map_col1, map_col2 = st.columns(2)

    finite_delta = delta[~np.isnan(delta)]
    vlim = float(np.nanpercentile(np.abs(finite_delta), 98)) if finite_delta.size else 1.0
    vlim = max(vlim, 1e-3)

    with map_col1:
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(delta, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.set_title(f"\u0394{index_name.upper()}", fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)
        st.pyplot(fig)
        st.caption(f"Red = increase in {index_name.upper()}, blue = decrease.")

        st.download_button(
            f"Download \u0394{index_name.upper()} (GeoTIFF)",
            data=_array_to_geotiff_bytes(delta, bands_before["nir"]),
            file_name=f"{before_item.id}_{after_item.id}_delta_{index_name}.tif",
            mime="image/tiff",
            key=f"{key_prefix}_download_delta",
        )

    with map_col2:
        fig2 = _plot_classification(codes, preset["labels"], preset["colors"], "Classification")
        st.pyplot(fig2)

        codes_export = codes.astype("float32")
        codes_export[codes < 0] = np.nan

        st.download_button(
            "Download Classification (GeoTIFF)",
            data=_array_to_geotiff_bytes(codes_export, bands_before["nir"]),
            file_name=f"{before_item.id}_{after_item.id}_classification.tif",
            mime="image/tiff",
            key=f"{key_prefix}_download_class",
        )
