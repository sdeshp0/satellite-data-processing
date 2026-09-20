# satellite-data-processing

An interactive toolkit for exploring Sentinel-2 satellite imagery using the
Microsoft Planetary Computer. The project provides a modular pipeline for:

- selecting an Area of Interest (AOI) -- from a curated sample list or by
  text search
- searching Sentinel-2 scenes via STAC, with a thumbnail-based picker and
  automatic best-pair selection for before/after comparisons
- loading and clipping scene bands, with concurrent, decimated COG reads
  and automatic multi-tile mosaicking when an AOI straddles a tile boundary
- computing eight spectral indices (NDVI, EVI, SAVI, NBR, NDMI, NDWI, NDBI,
  BSI) and per-scene summary statistics
- running before/after change detection for wildfire, flood, and logging
  events, using established remote-sensing methodologies, with extensive
  built-in checks that a comparison is actually apples-to-apples -- grid
  alignment, radiometric offset correction, seasonal/illumination
  matching, and pixel-level coverage-overlap verification
- highlighting water, urban, vegetation, and bare-soil extent directly on a
  scene via spectral thresholding and contour extraction
- a set of one-click "sample analyses" against real, documented events,
  each verified against live Planetary Computer data via a standalone
  coverage-health-check script
- visualizing RGB, spectral indices, and change maps -- including an
  interactive before/after swipe comparison -- through a Streamlit-based UI

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
  OSM Nominatim otherwise. Defaults to a 20km × 20km AOI -- large enough
  for useful context, small enough to somewhat reduce (not eliminate) the
  odds of straddling a Sentinel-2 UTM zone or tile boundary; still fully
  adjustable.

### Single Scene Explorer
- 🛰️ **Sentinel-2 STAC search** with cloud-cover filtering, cached across
  reruns.
- 🖼️ **Thumbnail-based scene picker**: a sortable table (date, cloud cover,
  platform, granule NoData %, MGRS tile, UTM zone, and a quicklook preview
  image) with a **minimum-coverage filter** to hide low-coverage candidates
  before picking, instead of a plain dropdown.
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
- 🤖 **Automatic best-pair selection**: as soon as before/after searches
  both return results, the pair with the best combined score is
  pre-selected and the comparison runs immediately -- no manual picking
  required. Selection uses a tiered preference (cloud cover primary,
  seasonal/day-of-year match secondary, matching Sentinel-2 tile
  tertiary) rather than a hard rule on any one factor, since real testing
  showed any single absolute veto could force a badly-clouded pick over a
  much clearer alternative. Manual override is always available via
  either scene-picker table below it, which also shows the delta in
  day-of-year and sun elevation against whatever's selected on the other
  side.
- 🧩 **Multi-tile mosaicking**: Sentinel-2 tiles sit on a fixed ~110km
  grid, independent of any AOI drawn on top of it -- an AOI near a tile
  boundary can have a real chunk of it fall outside whichever single tile
  a given scene belongs to, and no choice of date fixes that. When this
  happens, the app automatically fetches other tiles from the same
  acquisition date and fills the gap, entirely automatically. This is
  additive-only (a tile's own real data is never overwritten) and
  effectively free when it isn't needed -- an AOI that doesn't straddle a
  boundary triggers no extra fetches at all.
- 🗺️ **Coverage overlap map**: beyond each scene's own coverage, a
  pixel-level map showing exactly where the AOI has usable data in both
  dates, only one date, or neither -- since two scenes can each
  individually look fine while still covering different parts of the
  AOI. This is what the delta/classification results are actually
  computed from.
- 🧭 **Comparability checks**: warns when before/after scenes differ in
  UTM zone, MGRS tile, Sentinel-2 platform (2A/2B cross-calibration), or
  are seasonally/illumination mismatched (day-of-year distance, sun
  elevation difference) -- any of which can introduce a signal that looks
  like "change" but isn't.
- 🔀 **Before/After swipe compare**: an interactive slider revealing more
  or less of the before/after RGB (or the active index, colorized to
  match its static chart) as you drag -- a fast, at-a-glance
  "does this look different" check alongside the quantified delta and
  classification maps.
- 📷 **Side-by-side RGB**, a continuous delta map, and a discrete
  classification map (sized consistently with the delta map regardless
  of how many classes a preset uses) with a legend, plus a change summary
  (% of AOI flagged as changed, mean delta, per-class breakdown).
- 💾 **Downloads**: delta and classification layers as georeferenced
  GeoTIFFs.
- ⚡ **Sample analyses**: 11 one-click, pre-configured comparisons against
  real events -- Camp Fire (CA), Kangaroo Island bushfires (Australia),
  Hurricane Harvey flooding (Houston), Amazon deforestation (Rondônia),
  Dixie Fire (CA), the Mati wildfire (Greece), Lake Mead's drought-driven
  water-level decline, Gran Chaco deforestation (Paraguay), Missouri
  River flooding (Nebraska/Iowa), Murray-Darling Basin flooding
  (Australia), and the Pantanal's seasonal flood pulse (Brazil). Each
  sample's date windows were tuned using real coverage data (see
  `check_sample_coverage.py` below), not guessed -- several were adjusted
  or replaced entirely after live testing showed poor real-world
  coverage, most notably Murray-Darling, which is left in as a documented
  example of a genuine data-availability limit at that specific AOI
  position rather than something scene selection alone can fix.

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

### Comparison & Data Quality
A recurring theme in this project: a before/after pair that *looks*
reasonable by any single metric can still be badly mismatched, and the
failure modes compound in non-obvious ways. What's checked, and where:

- **Radiometric offset correction**: ESA's Processing Baseline 04.00
  (2022-01-25+) added a constant -1000 DN offset to every reflectance
  band. Scenes processed under that baseline are corrected automatically
  on load, per-tile (including mosaic companion tiles, which can have a
  different baseline than the primary), so a before/after pair spanning
  that date isn't thrown off by a large, spurious, uniform shift.
- **Data coverage**: two layers of checking. A cheap, geometry-only
  per-scene check (`core.load.load_scene`'s coverage_fraction) flags
  early when an AOI mostly misses a granule's real footprint; a deeper,
  pixel-level joint check (`core.change.coverage_overlap`) determines
  exactly where the AOI has usable data in both dates, one date, or
  neither -- and is what the multi-tile mosaicking above exists to
  improve. Reads use `boundless=True` so a partially out-of-bounds AOI
  window can't produce corrupted/tiled-looking output, and the
  coverage-overlap check correctly excludes Sentinel-2's own "No Data"
  SCL classification (which boundless-read padding also reports),
  something an earlier version of this check missed.
- **Comparability warnings**: UTM zone, MGRS tile, Sentinel-2 platform
  (2A vs 2B carry a small documented ~1.1% VNIR cross-calibration
  difference), seasonal (day-of-year) distance, and sun elevation
  difference are all checked and surfaced for whichever pair is
  currently loaded.
- **Automatic selection quality**: the same checks above inform automatic
  best-pair selection (see Change Detection), so problems are avoided
  where possible rather than only flagged after the fact.
- **Scene picker visibility**: the thumbnail table shows NoData %, MGRS
  tile, UTM zone, and (when comparing) the day-of-year and sun-elevation
  delta against whatever's selected on the other side -- so a bad pairing
  is visible before a scene is even loaded, not just after.
- **Whole-scene grid alignment**: the "after" scene's bands are aligned
  onto the "before" scene's grid immediately after loading (and after any
  mosaicking) -- before RGB rendering or index computation -- rather than
  aligning only a derived index value, which previously left RGB previews
  visibly stretched relative to each other when the two scenes came from
  different UTM zones.

### Performance
- Concurrent, GDAL-tuned COG reads (merged byte ranges, VSI caching,
  `ThreadPoolExecutor` across bands) -- scene loads went from minutes to
  seconds.
- Optional decimated reads for preview-resolution imagery.
- Multi-tile mosaicking (see above) is additive-only and cost-scoped: an
  AOI that doesn't need it triggers no extra network activity at all;
  when it is needed, real measured cost is on the order of ~11 seconds
  per scene on average, capped by a hard limit on how many companion
  tiles will ever be fetched for one load.
- `st.cache_data` applied at both the STAC search and scene-loading layers,
  shared across pages via a single cached loader.
- Asset hrefs are re-signed immediately before each raster read (rather
  than trusting a cached item's baked-in signed URL), so a stale cache
  entry can't cause an expired-token failure mid-session.

---

## Development & Testing

### `check_sample_coverage.py`
A standalone script (run outside Streamlit, from the repo root) that
exercises the real search → pair-selection → load → coverage-overlap
pipeline against live Planetary Computer data for every curated sample in
`app/components/sample_analyses.py`, and writes a full text report:

```bash
python check_sample_coverage.py                              # check every sample
python check_sample_coverage.py --list                       # list sample keys, no network calls
python check_sample_coverage.py --sample lake_mead_drought    # check just one
python check_sample_coverage.py --both-threshold 70           # stricter pass/fail bar
python check_sample_coverage.py --max-dim 256                 # faster, coarser pass
python check_sample_coverage.py --no-mosaic                   # A/B: disable mosaicking to see its cost/benefit
python check_sample_coverage.py --sample harvey_houston_2017 \
    --before-start 2017-01-01 --before-end 2017-02-28 \
    --after-start 2017-10-01 --after-end 2017-10-31           # test a specific date window without editing source
```

It flags any sample below the usable-both-dates threshold, and reports an
aggregate "Mosaic cost summary" (tiles fetched, time spent) so the
multi-tile mosaicking feature's cost can be measured rather than assumed.
This is how the current 11-sample set was actually tuned -- several
samples' date windows were revised, and one (the original
Ganges-Brahmaputra Delta monsoon-flood sample) was removed entirely,
based on what this script showed against real data rather than guesswork.
The `--before-start`/`--before-end`/`--after-start`/`--after-end`
override flags exist specifically to let a narrower or shifted window be
tested for one sample without touching `sample_analyses.py` first --
useful for hunting down a genuinely clear date before committing to it.

---

## Project Structure
```
satellite-data-processing/
│
├── core/
│   ├── __init__.py
│   ├── stac.py          # STAC search (Sentinel-2), companion-tile search for mosaicking
│   ├── load.py           # Concurrent, decimated COG reads; multi-tile mosaicking
│   ├── indices.py        # NDVI, EVI, SAVI, NBR, NDMI, NDWI, NDBI, BSI
│   ├── stats.py           # Band/index summary stats, SCL land-cover breakdown
│   ├── change.py          # Grid alignment, delta, classification, comparability checks,
│   │                       #   coverage overlap, automatic best-pair selection
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
│   │   ├── scene_selector.py     # STAC search, thumbnail scene picker, coverage filter
│   │   ├── scene_loader.py       # Shared cached scene loader (mosaic-aware)
│   │   ├── index_display.py      # RGB + spectral index charts + stats
│   │   ├── change_display.py     # Before/after comparison rendering, swipe compare
│   │   ├── sample_analyses.py    # Curated one-click event comparisons
│   │   └── feature_display.py    # Spectral feature identification rendering
│   │
│   └── requirements.txt
│
├── check_sample_coverage.py   # Standalone dev tool -- see Development & Testing above
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
  event -- either manually, or via a one-click sample analysis, with the
  before/after pair auto-selected and multi-tile mosaicking applied
  automatically wherever the AOI needs it
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
- Multi-tile mosaicking means a single scene load can now issue more than
  one tile's worth of band requests when an AOI straddles a Sentinel-2
  tile boundary (bounded by `core/load.py`'s `max_companion_tiles`, default
  3). This is measured to add roughly ~11 seconds per affected scene load
  in practice (see `check_sample_coverage.py --no-mosaic` for an A/B
  comparison) -- worth keeping in mind for perceived responsiveness on a
  constrained deployment, though it only affects AOIs that actually need
  it.
- `.streamlit/secrets.toml` is gitignored and must never be committed --
  use the platform's own secrets UI for deployed apps.