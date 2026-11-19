# CLAUDE.md — aqi-hcho-india

Python + React project for ISRO's hackathon.

## Goals

1. **Daily surface AQI map of India** from satellite data — INSAT-3D AOD (MOSDAC), Sentinel-5P NO2/SO2/CO/O3, ERA5 meteorology — using ML (LightGBM) trained on CPCB ground-station data, then the CPCB National AQI formula.
2. **HCHO hotspot maps over India** from Sentinel-5P TROPOMI for biomass-burning seasons (Oct–Nov, Mar–May), linked to VIIRS fire counts and ERA5 wind transport.

## Fixed parameters (source of truth: `config/settings.yaml`)

- Grid: 0.1°, EPSG:4326, India bbox lon 68–98°E, lat 6–38°N
- Years: 2019–2025
- Never hard-code bbox/grid/dates/GEE project id in code — read from `config/settings.yaml`.

## Stack

- **Python 3.11**: earthengine-api, geemap, xarray, rioxarray, geopandas, lightgbm, shap, pysal
- **Backend**: FastAPI + titiler (serves COGs from `src/export/to_cog.py`)
- **Frontend**: React + Vite + TypeScript + MapLibre GL + Recharts + Tailwind (`web/`)

## Layout

```
config/settings.yaml        # bbox, grid, dates, GEE project id, AQI breakpoints
data/ (raw/, interim/, processed/)   # git-ignored
src/
  ingest/  gee_s5p.py, gee_era5.py, gee_firms.py, mosdac_insat.py, cpcb.py
  grid/    regrid.py, masks.py
  aqi/     features.py, train.py, predict.py, aqi.py, validate.py
  hcho/    composite.py, anomaly.py, hotspots.py, fire_link.py, trajectories.py, regions.py
  export/  to_cog.py, to_geojson.py
api/      main.py, routers/(aqi.py, hcho.py, stations.py)
web/      React app (pages: AQI Map, HCHO Hotspots, Fire & Transport, Validation, About)
notebooks/
tests/
docker-compose.yml
```

Pipeline: `ingest → grid (regrid to common 0.1° grid) → aqi | hcho → export (COG/GeoJSON) → api → web`.

## CPCB National AQI

Units: µg/m³, except CO in mg/m³. Overall AQI = max of sub-indices (need ≥3 pollutants, one of which must be PM2.5 or PM10).
Sub-index by linear interpolation within the breakpoint range.

| Category     | AQI     | Colour  | PM2.5 24h | PM10 24h | NO2 24h | O3 8h   | CO 8h    | SO2 24h  | NH3 24h   |
|--------------|---------|---------|-----------|----------|---------|---------|----------|----------|-----------|
| Good         | 0–50    | #00B050 | 0–30      | 0–50     | 0–40    | 0–50    | 0–1.0    | 0–40     | 0–200     |
| Satisfactory | 51–100  | #92D050 | 31–60     | 51–100   | 41–80   | 51–100  | 1.1–2.0  | 41–80    | 201–400   |
| Moderate     | 101–200 | #FFFF00 | 61–90     | 101–250  | 81–180  | 101–168 | 2.1–10   | 81–380   | 401–800   |
| Poor         | 201–300 | #FF9900 | 91–120    | 251–350  | 181–280 | 169–208 | 10.1–17  | 381–800  | 801–1200  |
| Very Poor    | 301–400 | #FF0000 | 121–250   | 351–430  | 281–400 | 209–748 | 17.1–34  | 801–1600 | 1201–1800 |
| Severe       | 401–500 | #C00000 | >250      | >430     | >400    | >748    | >34      | >1600    | >1800     |

## Conventions

- `data/` is never committed. Raw downloads → `data/raw/`, regridded → `data/interim/`, final products → `data/processed/`.
- Gridded data as xarray Datasets with `lat`/`lon` dims on the common grid; export rasters as COGs.
- Validate ML models with spatial cross-validation (hold out stations), not random splits.
- Current status: repo skeleton only — modules are stubs; no analysis code yet.
