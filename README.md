# satellite-data-processing
A repository for learning and experimenting with satellite data using the Microsoft Planetary Computer.

The repo provides:
- shared geospatial core (core/)
- interactive app (app/)
- notebooks for experiments and testing

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
│   ├── load.py          # Load band data for scene, clipped to selected area of interest
│   ├── indices.py       # Computation of indices including NDVI, NBR, NDWI
│   ├── viz.py           # Visualization helpers
│   └── utils.py         # AOI helpers, raster data processing functions
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
- search for a place using Nominatim as a geolocator
- define AOI size
- search for Sentinel2 scenes in a configurable date range
- select a single scene and display charts for that scene

---
