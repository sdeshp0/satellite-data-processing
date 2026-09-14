# satellite-data-processing

An interactive toolkit for exploring Sentinel-2 satellite imagery using the
Microsoft Planetary Computer. The project provides a modular pipeline for:

- selecting an Area of Interest (AOI) -- from a curated sample list or by
  text search
- searching Sentinel-2 scenes via STAC, with a thumbnail-based picker
- loading and clipping scene bands, with concurrent, decimated COG reads
- computing eight spectral indices (NDVI, EVI, SAVI, NBR, NDMI, NDWI, NDBI,
  BSI) and per-scene summary statistics
- running before/after change detection for wildfire, flood, and logging
  events, using established remote-sensing methodologies, with built-in
  checks that flagged comparisons are actually apples-to-apples
- highlighting water, urban, vegetation, and bare-soil extent directly on a
  scene via spectral thresholding and contour extraction
- a set of one-click "sample analyses" against real, documented events
- visualizing RGB, spectral indices, and change maps through a
  Streamlit-based UI

The repository is structured to support experimentation and extension into
additional workflows or pages.

---

## Features

### Home page -- AOI selection
- 🌍 **Sample locations**: a curated list of pre-configured AOIs (including
  several tied to real wildfire/flood/deforestation events), with no
  external dependency -- works everywhere, including hosts where geocoding
  is unreliable.
- 🔎 **Search by name**: free-text geocoding, via LocationIQ if configured
  (see [Secrets](#2-secrets), recommended for cloud deployments) or free
  OSM Nominatim otherwise.

### Single Scene Explorer
- 🛰️ **Sentinel-2 STAC search** with cloud-cover filtering, cached across
  reruns.
- 🖼️ **Thumbnail-based scene picker**: a sortable table (date, cloud cover,
  platform, and a quicklook preview image) instead of a plain dropdown.
- 🎨 **RGB rendering** with a 2nd-98th percentile contrast stretch.
- 📈 **Eight spectral indices** (NDVI, EVI, SAVI, NBR, NDMI, NDWI, NDBI,
  BSI), each colored by what it represents (green = vegetation, blue =
  water/moisture, red = burn, orange = built-up, brown = bare soil), with a
  multiselect so you're not shown all eight by default.
- 📊 **Scene statistics**: an SCL-based land-cover breakdown (% cloud,
  vegetation, water, bare soil, etc.), band and index summary statistics
  (mean/std/min/max/p10/p90), and AOI coverage info (resolution, pixel
  dimensions, approximate km²).
- ☁️ **Toggleable cloud/shadow masking** via the Scene Classification Layer.
- 💾 **Downloads**: RGB as PNG, any index as a georeferenced GeoTIFF.

### Change Detection
- 🔥🌊🪓 **Event-type presets** (Wildfire, Flood, Logging/Deforestation,
  or Custom) that auto-configure the index and classification methodology:
  - **Wildfire**: dNBR classified per USGS / Key & Benson (2006)
    burn-severity thresholds.
  - **Flood**: NDWI > 0 water threshold (McFeeters, 1996) per date, then
    differenced into new/lost/persistent water.
  - **Logging / Deforestation**: NDVI > 0.3 vegetation threshold per date,
    then differenced.
  - **Custom**: statistical (k-sigma) thresholding on any of the 8 indices.
- 🗺️ **Automatic grid alignment** between independently-loaded before/after
  scenes, and a combined cloud mask requiring a pixel to be clear in
  *both* dates.
- 📷 **Side-by-side RGB**, a continuous delta map, and a discrete
  classification map with a legend, plus a change summary (% of AOI
  flagged as changed, mean delta, per-class breakdown).
- 💾 **Downloads**: delta and classification layers as georeferenced
  GeoTIFFs.
- ⚡ **Sample analyses**: one-click, pre-configured comparisons against real
  events. The original five (Camp Fire, Kangaroo Island bushfires,
  Hurricane Harvey flooding, Amazon deforestation in Rondônia, and
  Bangladesh monsoon flooding) are coastal/deltaic/storm-driven and most
  likely to hit cloud or coverage issues; four more (Dixie Fire, the Mati
  wildfire in Greece, Lake Mead's drought-driven water-level decline, and
  Gran Chaco deforestation in Paraguay) were added specifically for more
  reliable, inland/dry-climate coverage. All use automatic
  (lowest-cloud-cover) scene selection with progressive fallback --
  relaxed cloud-cover thresholds, then a widened date window -- so a
  single cloudy scene doesn't fail the whole sample outright.

### Feature Identification (experimental)
- 🖍️ **Spectral feature outlines**: thresholds a chosen index (NDWI, NDBI,
  NDVI, or BSI) into a cleaned mask and traces its boundary directly on
  the RGB image, for water/rivers, urban/built-up areas, vegetation, and
  bare soil -- all several pixels wide at Sentinel-2's 10m resolution, so
  well suited to this approach.
- 🎚️ **Adjustable thresholds and minimum region size** per feature type,
  with noisy sub-threshold blips filtered out via morphological cleanup
  (`scikit-image`) before contours are extracted.
- 📐 **Region stats** (count and area in km²) per feature type.
- 💾 **Downloads**: outlines as georeferenced GeoJSON (opens directly in
  QGIS or similar), or a single selected feature's mask as GeoTIFF.
- 🚫 **Deliberately not attempted**: roads and individual buildings --
  sub-pixel at 10m resolution, not reliably extractable via spectral
  thresholding regardless of technique. A more realistic path to showing
  road locations is overlaying real OpenStreetMap vector data as a
  reference layer rather than deriving roads from pixels; not built yet.
- Standalone page for now, so the threshold/contour approach can be
  validated across different scenes before folding a version of it into
  Single Scene (index overlay) or Change Detection (a "damage area"
  outline derived from the classification mask).

### Comparison & Data Quality Checks
Added after finding that some before/after pairs were being compared even
though they weren't really apples-to-apples:
- **Radiometric offset correction**: ESA's Processing Baseline 04.00
  (2022-01-25+) added a constant -1000 DN offset to every reflectance
  band. Scenes processed under that baseline are corrected automatically
  on load, so a before/after pair spanning that date isn't thrown off by a
  large, spurious, uniform shift.
- **Data coverage check**: Sentinel-2 granules aren't always fully covered
  by real data at swath edges. Reads use `boundless=True` so a partially
  out-of-bounds AOI window can no longer produce corrupted/tiled-looking
  output (a real bug this fixed), and a computed coverage fraction is
  surfaced as a warning whenever the AOI significantly misses a granule's
  actual footprint.
- **Comparability warnings**: flags when before/after scenes differ in UTM
  zone (triggers reprojection, worth knowing about), MGRS tile (different
  source granule), or Sentinel-2 platform (2A vs 2B carry a small
  documented ~1.1% VNIR cross-calibration difference, not independently
  corrected for).
- **Scene picker visibility**: the thumbnail table shows NoData %, MGRS
  tile, and UTM zone columns, so a bad pairing is visible before a scene
  is even loaded, not just after.
- **Whole-scene grid alignment**: the "after" scene's bands are aligned
  onto the "before" scene's grid immediately after loading -- before RGB
  rendering or index computation -- rather than aligning only a derived
  index value, which previously left RGB previews visibly stretched
  relative to each other when the two scenes came from different UTM
  zones.

### Performance
- Concurrent, GDAL-tuned COG reads (merged byte ranges, VSI caching,
  `ThreadPoolExecutor` across bands) -- scene loads went from minutes to
  seconds.
- Optional decimated reads for preview-resolution imagery.
- `st.cache_data` applied at both the STAC search and scene-loading layers,
  shared across pages via a single cached loader.
- Asset hrefs are re-signed immediately before each raster read (rather
  than trusting a cached item's baked-in signed URL), so a stale cache
  entry can't cause an expired-token failure mid-session.

---

## Project Structure
```
satellite-data-processing/
│
├── core/
│   ├── __init__.py
│   ├── stac.py          # STAC search (Sentinel-2 only)
│   ├── load.py           # Concurrent, decimated COG reads clipped to the AOI
│   ├── indices.py        # NDVI, EVI, SAVI, NBR, NDMI, NDWI, NDBI, BSI
│   ├── stats.py           # Band/index summary stats, SCL land-cover breakdown
│   ├── change.py          # Grid alignment, delta, and event classification
│   ├── features.py        # Spectral feature masks, cleanup, contour extraction
│   ├── viz.py             # RGB percentile stretch, SCL visualization
│   └── utils.py            # AOI helpers, band scaling/resampling, cloud masking
│
├── app/
│   ├── Home.py
│   ├── pages/
│   │   ├── 01_Single_Scene.py
│   │   ├── 02_Change_Detection.py
│   │   └── 03_Feature_Identification.py
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── aoi_selector.py       # Sample-location + search-by-name AOI input
│   │   ├── scene_selector.py     # STAC search + thumbnail scene picker
│   │   ├── scene_loader.py       # Shared cached scene loader
│   │   ├── index_display.py      # RGB + spectral index charts + stats
│   │   ├── change_display.py     # Before/after comparison rendering
│   │   ├── sample_analyses.py    # Curated one-click event comparisons
│   │   └── feature_display.py    # Spectral feature identification rendering
│   │
│   └── requirements.txt
│
├── .streamlit/
│   └── secrets.toml      # Local-only; never committed (see Secrets below)
│
├── notebooks/
│   ├── 01_explore_planetary_computer.ipynb
│   └── 02_time_series_analysis.ipynb
│
├── .gitignore
├── environment.yaml
└── README.md
```

---
## Running the app

### 1. Install dependencies

```bash
conda env create -f environment.yaml
conda activate satellite-data-processing
```

If the environment.yaml file is updated:
```bash
conda env update -f environment.yaml --prune
```

Package versions in both `environment.yaml` and `app/requirements.txt` are
specified as minimum floors (`>=`), not exact pins, chosen to guarantee
features the app actually relies on (e.g. Streamlit's dataframe
row-selection and `ImageColumn`, used by the scene picker). Run
`pip freeze` after a local install if you want fully locked versions for a
production deploy.

### 2. Secrets

Only needed for reliable geocoding in "Search by name" mode -- the app is
fully usable without this via "Sample locations."

Free OSM Nominatim geocoding can be unreliable or blocked on shared cloud
hosts (e.g. Streamlit Community Cloud), since many unrelated apps share the
same outbound IP ranges and Nominatim rate-limits at the IP level.
[LocationIQ](https://locationiq.com) is a hosted, Nominatim-compatible
service built for this kind of traffic, with a free tier generous enough
for a demo app.

To enable it, create `.streamlit/secrets.toml`  with:

```toml
LOCATIONIQ_API_KEY = "your_locationiq_token_here"
```

For a deployed app on Streamlit Community Cloud, add the same key/value
pair under your app's **Settings → Secrets** instead -- don't upload the
local file there.

### 3. Launch Streamlit App

```bash
streamlit run app/Home.py
```

This opens an interactive UI where you can:
- pick an AOI from the sample list, or search for a place by name
- explore a single Sentinel-2 scene: RGB, spectral indices, and scene
  statistics
- run before/after change detection for a wildfire, flood, or logging
  event -- either manually, or via a one-click sample analysis
- highlight water, urban, vegetation, or bare-soil outlines on a scene via
  spectral thresholding

---

## Notes for deployment

A few things that matter specifically when deploying (e.g. to Streamlit
Community Cloud or Hugging Face Spaces) rather than running locally:

- `rioxarray` must be in `app/requirements.txt`, not just
  `environment.yaml` -- pip-based deploys don't read the conda manifest.
- Every page under `app/pages/` needs its own `sys.path` bootstrap (see the
  top of `01_Single_Scene.py` / `02_Change_Detection.py` /
  `03_Feature_Identification.py`), since a visitor
  can land directly on a page's URL without `Home.py` having run
  first in that process.
- Planetary Computer's signed asset URLs are time-limited; `core/load.py`
  re-signs each href immediately before reading it, rather than relying on
  the href attached to a `pystac.Item` that may have sat in Streamlit's
  shared, server-side cache for up to an hour.
- `.streamlit/secrets.toml` is gitignored and must never be committed --
  use the platform's own secrets UI for deployed apps.