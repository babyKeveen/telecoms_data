# Telco Signal POC

LTE drive-test signal analysis — cell-to-cell handover routes and coverage gap detection.

## Getting Started

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) and Git. No WSL2 or other Linux setup required.

### 1. Clone the repo
```bash
git clone <repo-url>
cd endeavourDemo
```

### 2. Configure your environment
```bash
cp .env.example .env
```
Edit `.env` and set:
- `ANTHROPIC_API_KEY` — your Anthropic API key (needed for NL trip search)
- `DATA_PATH` — absolute path to your local data folder (e.g. `C:\S3Data` on Windows, `/Users/you/S3Data` on Mac)

### 3. Start the environment
```bash
docker compose up                                               # CPU (default)
docker compose -f docker-compose.yaml -f docker-compose.gpu.yml up  # NVIDIA GPU
```

| Service | URL |
|---|---|
| Jupyter Lab | http://localhost:8888 |
| Streamlit app | http://localhost:8501 |

### 4. Convert data to Parquet (run once)
Open Jupyter at http://localhost:8888 and run:
```python
import sys; sys.path.insert(0, '/home/jovyan')
from pipeline.ingest import load_raw, clean, to_parquet
to_parquet(clean(load_raw()))
```

### 5. Launch the demo app
The Streamlit app starts automatically. Open http://localhost:8501

## App Pages

| Page | Description |
|---|---|
| 1 — Route Map | Vehicle routes rendered as folium PolyLines; cell IDs resolved to lat/lon |
| 2 — Gap Analysis | Coverage gap hotspots (ping interval > 15 min) as CircleMarkers sized by event count |
| 3 — Corridor Analysis | City-pair corridor filter: clips trips to a perpendicular corridor around the A→B line |
| 4 — Trip Search | FAISS vector similarity search by duration, cell count, handovers, events, time of day |
| 5 — Route Search | Find trips matching an origin→destination city pair using start/end lat/lon vectors |
| 6 — Temporal Patterns | Fleet activity and handover rate by hour of day, day of week, month, and heatmap |

## Project Structure
```
telco-poc/
├── data/               # Dataset lives here (gitignored)
├── pipeline/           # Core logic — ingest, handovers, gaps, vectors
├── notebooks/          # Exploration notebooks + CLI rebuild scripts
├── app/                # Streamlit demo
│   ├── Home.py
│   └── pages/          # Pages 1–6 (see App Pages above)
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

## Use Cases
| # | Use Case | Key Output |
|---|---|---|
| 1 | Most used routes | Cell-to-cell handover graph, weighted by frequency |
| 2 | Coverage gaps | Map of poor RSRP / negative SINR zones per route |
| 3 | Corridor analysis | Trip density and gap hotspots along a city-pair corridor |
| 4 | Trip similarity | Top-K trips matching a feature profile via FAISS vector search |
| 5 | Route search | Trips whose start/end cells best match a city-to-city corridor |
| 6 | Temporal patterns | Fleet utilisation and network stress by time of day / week / month |

## Signal Quality Thresholds
| Metric | Poor threshold | Meaning |
|---|---|---|
| RSRP | < -90 dBm | Weak signal strength |
| SINR | < 0 dB | Noise dominates signal |
| Signal bars | ≤ 2 / 5 | Low usable signal |
