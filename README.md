# satellite-data-processing
A modular, multi‑project repository for learning, experimenting, and storytelling with satellite data using the 
Microsoft Planetary Computer.

The repo provides:
- a shared geospatial core (core/)
- a unified interactive app (app/)
- a config‑driven, reproducible pipeline

This structure allows for the addition of modules as new pages to the app

---
## Current Scope (Phase 1)

- Select an AOI
- Select a date range
- Search Sentinel-2 scenes
- Pick one scene
- Load RGB + compute NDVI/NBR/NDWI
- Display results

---
## Project Structure
```
satellite-data-processing/
│
├── core/
│   ├── __init__.py
│   ├── stac.py          # STAC search (Sentinel-2 only)
│   ├── load.py          # Load RGB + required bands for indices
│   ├── indices.py       # NDVI, NBR, NDWI (minimal set)
│   ├── viz.py           # RGB visualization helpers
│   └── utils.py         # AOI helpers, small utilities
│
├── app/
│   ├── streamlit_app.py
│   ├── pages/
│   │   └── 1_Single_Scene.py
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── aoi_selector.py      # AOI input + map preview
│   │   ├── scene_selector.py    # List STAC results + pick one
│   │   └── index_display.py     # Show NDVI/NBR/NDWI maps
│   │
│   └── requirements.txt
│
├── notebooks/
│   ├── 01_explore_planetary_computer.ipynb  
│   └── 02_time_series_analysis.ipynb
│
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
```commandline
conda env update -f environment.yaml --prune
```

### 2. Launch Streamlit App

```
streamlit run app/streamlit_app.py
```

This opens an interactive UI where you can:
- search for a place
- define AOI size
- choose indices (NBR, NDVI, NDWI, etc.)
- run the before/after pipeline
- explore RGB, delta maps, swipe comparison
- read human‑interpretable index summaries

---

## Future Growth

Once Page 1 is stable, we will add:

- Time Series Explorer (Page 2)
- Before/After Comparison (Page 3)
- Caching, config system, interpretation engine, etc.

