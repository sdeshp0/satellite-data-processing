# satellite-data-processing

An interactive toolkit for exploring Sentinel‑2 satellite imagery using the
Microsoft Planetary Computer. The project provides a clean, modular pipeline for:

- selecting an Area of Interest (AOI)
- searching Sentinel‑2 scenes via STAC
- loading and clipping scene bands
- computing spectral indices (NDVI, NBR, NDWI)
- visualizing RGB and index maps
- running everything through a Streamlit-based UI

The repository is structured to support experimentation and extension into additional workflows or pages.

---

## Features

- 🌍 **AOI selection** using geocoded search + adjustable width/height  
- 🛰️ **Sentinel‑2 STAC search** with cloud-cover filtering  
- 🎨 **RGB rendering** with min–max stretch  
- 📈 **Spectral indices**: NDVI, NBR, NDWI  
- 🗺️ **Interactive map preview** using Leafmap  
- 🧩 **Modular architecture** (core processing + UI components)  
- 🧼 **Fully cleaned and modernized codebase** with type hints and consistent structure  

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
