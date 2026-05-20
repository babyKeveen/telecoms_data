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

### 4. Build the data pipeline (run once, inside the container)

**Stage handover events** from raw daily files (~1–2 hours, resumable):
```bash
rm -rf /home/jovyan/data/stage/handover_events/
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=21600 \
    --output notebooks/build_stage_duckdb_out.ipynb \
    notebooks/build_stage_duckdb.ipynb
```
Monitor progress: `cat /tmp/stage_progress.json`

**Build trips** from staged events (~30 min):
```bash
python notebooks/rebuild_trips_stage.py
```

**Build vector DB** (seconds):
```bash
python notebooks/build_vector_db_np.py
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

## Data Pipeline

Raw data is 364 daily `.gz` files (headerless CSV, ~100 columns) sourced from the telco SIM platform. The pipeline extracts a subset of columns and stages them as hive-partitioned Parquet.

```
DATA_PATH/sim_test_decoded/2025/*.gz
        ↓  build_stage_duckdb.ipynb
stage/handover_events/event_date=*/   (one row per ping event)
        ↓  build_trips_duckDB.ipynb
stage/trips/event_date=*/             (one row per trip + KPI aggregates)
        ↓  build_vector_db_np.py
vector_db_np/                         (numpy similarity index)
```

**KPI columns added to trips** (aggregated from per-event values):

| Column | Description |
|---|---|
| `avg_neighbor_rsrp` | Average RSRP of strongest neighbour cell across trip |
| `min_neighbor_rsrp` | Minimum RSRP of strongest neighbour cell across trip |
| `avg_neighbor_rsrq` | Average RSRQ of strongest neighbour cell across trip |
| `avg_ping_ms` | Average of the four ping measurements per event, averaged across trip |

Raw per-event values (`pci_1_rsrp`, `pci_1_rsrq`, `ping1`–`ping4`) are retained in `handover_events` for event-level queries.

## Project Structure
```
telco-poc/
├── data/               # Column definitions reference (gitignored raw data)
├── pipeline/           # Core logic — ingest, handovers, gaps, vectors
├── notebooks/          # Build notebooks + CLI rebuild scripts
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
