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
  events, using established remote-sensing methodologies
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
  events (Camp Fire, Kangaroo Island bushfires, Hurricane Harvey flooding,
  Amazon deforestation in Rondônia, and Bangladesh monsoon flooding), with
  automatic (lowest-cloud-cover) scene selection -- no AOI or date-range
  setup required to see a result.

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
│   ├── viz.py             # RGB percentile stretch, SCL visualization
│   └── utils.py            # AOI helpers, band scaling/resampling, cloud masking
│
├── app/
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── 01_Single_Scene.py
│   │   └── 02_Change_Detection.py
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── aoi_selector.py       # Sample-location + search-by-name AOI input
│   │   ├── scene_selector.py     # STAC search + thumbnail scene picker
│   │   ├── scene_loader.py       # Shared cached scene loader
│   │   ├── index_display.py      # RGB + spectral index charts + stats
│   │   ├── change_display.py     # Before/after comparison rendering
│   │   └── sample_analyses.py    # Curated one-click event comparisons
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

To enable it, create `.streamlit/secrets.toml` (already gitignored --
**never commit this file**) with:

```toml
LOCATIONIQ_API_KEY = "your_locationiq_token_here"
```

For a deployed app on Streamlit Community Cloud, add the same key/value
pair under your app's **Settings → Secrets** instead -- don't upload the
local file there.

### 3. Launch Streamlit App

```bash
streamlit run app/streamlit_app.py
```

This opens an interactive UI where you can:
- pick an AOI from the sample list, or search for a place by name
- explore a single Sentinel-2 scene: RGB, spectral indices, and scene
  statistics
- run before/after change detection for a wildfire, flood, or logging
  event -- either manually, or via a one-click sample analysis

---

## Notes for deployment

A few things that matter specifically when deploying (e.g. to Streamlit
Community Cloud or Hugging Face Spaces) rather than running locally:

- `rioxarray` must be in `app/requirements.txt`, not just
  `environment.yaml` -- pip-based deploys don't read the conda manifest.
- Every page under `app/pages/` needs its own `sys.path` bootstrap (see the
  top of `01_Single_Scene.py` / `02_Change_Detection.py`), since a visitor
  can land directly on a page's URL without `streamlit_app.py` having run
  first in that process.
- Planetary Computer's signed asset URLs are time-limited; `core/load.py`
  re-signs each href immediately before reading it, rather than relying on
  the href attached to a `pystac.Item` that may have sat in Streamlit's
  shared, server-side cache for up to an hour.
- `.streamlit/secrets.toml` is gitignored and must never be committed --
  use the platform's own secrets UI for deployed apps.