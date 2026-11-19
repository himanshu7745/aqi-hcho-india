# aqi-hcho-india

Satellite-based air quality and formaldehyde (HCHO) mapping over India — ISRO hackathon project.

## What it does

1. **Daily surface AQI map** — combines INSAT-3D AOD, Sentinel-5P (NO₂, SO₂, CO, O₃) and ERA5 meteorology, trains LightGBM models on CPCB ground-station data to estimate surface pollutant concentrations, and applies the CPCB National AQI formula.
2. **HCHO hotspots** — Sentinel-5P TROPOMI HCHO composites and anomalies for biomass-burning seasons (Oct–Nov, Mar–May), hotspot statistics, and links to VIIRS fire counts and ERA5 wind transport.

Grid: 0.1° over India (68–98°E, 6–38°N), 2019–2025. All parameters live in [`config/settings.yaml`](config/settings.yaml).

## Repository layout

```
config/      settings.yaml (bbox, grid, dates, GEE project, AQI breakpoints)
data/        raw/, interim/, processed/  (git-ignored)
src/
  ingest/    GEE (S5P, ERA5, FIRMS), MOSDAC INSAT-3D, CPCB
  grid/      regridding and masks
  aqi/       features, training, prediction, AQI formula, validation
  hcho/      composites, anomalies, hotspots, fire link, transport, regions
  export/    COG and GeoJSON export
api/         FastAPI + titiler backend
web/         React + Vite + TypeScript + MapLibre GL + Recharts + Tailwind
notebooks/   exploration
tests/
```

## Setup

### Python (3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
earthengine authenticate
mkdir -p data/raw data/interim data/processed
```

Set your Earth Engine project id in `config/settings.yaml` (`project.gee_project_id`).

### API

```bash
uvicorn api.main:app --reload    # http://localhost:8000/health
```

### Web

```bash
cd web
npm install
npm run dev                      # http://localhost:5173
```

### Docker

```bash
docker compose up
```

## Data sources

| Source | Variables | Access |
|---|---|---|
| Sentinel-5P TROPOMI | NO₂, SO₂, CO, O₃, HCHO | Google Earth Engine |
| INSAT-3D/3DR | AOD | MOSDAC |
| ERA5 / ERA5-Land | winds, temperature, humidity, BLH, precip | Google Earth Engine |
| VIIRS (FIRMS) | active fires | Google Earth Engine |
| CPCB CAAQMS | ground-station PM2.5, PM10, NO₂, SO₂, CO, O₃, NH₃ | CPCB portal |

## Status

Skeleton only — modules are stubs.
