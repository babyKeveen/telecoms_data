# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Everything runs inside a Docker container managed by **Docker Desktop** — no WSL2 setup needed.

**Prerequisites:** Docker Desktop (Windows / Mac / Linux), Git

**First-time setup:**
```bash
cp .env.example .env   # then edit .env with your ANTHROPIC_API_KEY and DATA_PATH
docker compose up                                                    # CPU
docker compose -f docker-compose.yaml -f docker-compose.gpu.yml up  # NVIDIA GPU
```
- Jupyter Lab: http://localhost:8888
- Streamlit app: http://localhost:8501

The repo is mounted at `/home/jovyan/telco-poc` inside the container. Raw data lives at `/home/jovyan/data/` (mounted from the path you set in `DATA_PATH` in your `.env` — not in the repo).

## Running the App

```bash
streamlit run /home/jovyan/app/Home.py
```

## Data Pipeline

**One-time ingestion** (CSV → Parquet):
```python
from pipeline.ingest import load_raw, clean, to_parquet
to_parquet(clean(load_raw()))
```
Or `python -m pipeline.ingest` from the container.

**Rebuild trip/output artifacts** (CLI scripts in `notebooks/`):
```bash
python notebooks/rebuild_trips.py [--max-hours 4.0] [--start-date 2025-07-01] [--end-date 2025-07-31]
python notebooks/rebuild_outputs.py
python notebooks/build_vector_db.py   # re-run after rebuild_trips.py
```

## Architecture

### Data Flow
```
SRFG-v1.csv  →  pipeline/ingest.py  →  SRFG-v1.parquet
                                              ↓
                                  pipeline/handovers.py  →  handover events / NetworkX graph
                                  pipeline/gaps.py       →  gap-flagged records
```

The staged Parquet data lives outside the repo under `/home/jovyan/data/stage/` in hive-partitioned layouts:
- `stage/trips/event_date=*/*.parquet` — one row per trip with `cell_sequence` (e.g. `"123->456->789"`)
- `stage/handover_events/**/*.parquet` — one row per handover event with `vehicle_id`, `event_ts`, `cell_id`

### Querying
Both Streamlit pages and the CLI scripts use **DuckDB with no persistent DB file** — they call `duckdb.connect()` and query directly against Parquet via `read_parquet(..., hive_partitioning=true)`. There is no ORM or database schema to maintain.

### Cell Coordinate Lookup
`/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv` maps `global_cell_id` (format: `MCC-MNC-cell_id`, e.g. `310-410-12345`) to lat/lon. Both app pages filter to MCC=310, MNC=410 (US AT&T) and build a `dict[int, (lat, lon)]` keyed by cell_id integer. This lookup is `@st.cache_data`-decorated in the app.

### Streamlit App (`app/`)
- `Home.py` — landing page only
- `pages/1_Route_Map.py` — queries the trips Parquet, resolves cell IDs to coordinates via the lookup, renders folium PolyLines
- `pages/2_Gap_Analysis.py` — detects coverage gaps as ping intervals > expected cadence (~485s), maps hotspot cells as CircleMarkers sized by event count
- `pages/3_Corridor_Analysis.py` — city-pair corridor view: filters trips whose first/last cells fall within a configurable bounding box around two chosen cities, then clips each trip's `cell_sequence` to a perpendicular corridor around the A→B line (rejects trips where <60% of cells are on-corridor). City cell sets are resolved in Python from `coord_lookup`; the DuckDB query pushes `first_cell`/`last_cell` filtering into SQL via registered DataFrames. Silence-gap hotspots from `handover_events` are optionally overlaid as red CircleMarkers.
- `pages/4_Trip_Search.py` — FAISS vector similarity search: sidebar feature toggles (duration, cell count, handovers, events, hour of day, day of week) build a partial-feature query against the numpy vector DB; matched trips are rendered as colour-coded folium PolyLines with a results table below.
- `pages/5_Route_Search.py` — route similarity search: user picks an origin and destination from a hardcoded city list (29 US cities); searches the vector DB on `first_lat/lon` + `last_lat/lon` only; optional reverse-direction toggle; city markers pinned on the folium map alongside matched trip PolyLines.
- `pages/6_Temporal_Patterns.py` — fleet activity and network quality broken down by hour of day, day of week, and month. Four Plotly charts per time dimension (trip volume, avg duration, active vehicles, handover rate). Includes an hour × day-of-week activity heatmap. All data from the trips Parquet; no handover events scan needed.

### Vector Database (`/home/jovyan/data/vector_db/`)
FAISS index over all trips with 8 numeric features: `duration_minutes`, `n_cells`, `n_handovers`, `n_events`, `first_lat/lon`, `last_lat/lon`. Built by `pipeline/vectors.py`. Supports explicit feature-dict search and NL queries (NL requires `ANTHROPIC_API_KEY`).

### Pipeline Package (`pipeline/`)
- `ingest.py` — loads/cleans raw CSV; paths controlled by env vars `RAW_DATA_PATH` / `PARQUET_PATH`
- `handovers.py` — derives handover events (cell_id changes within a trip) and builds a weighted `nx.DiGraph`
- `gaps.py` — flags poor signal (RSRP < -90 dBm, SINR < 0 dB, signal bars ≤ 2) and detects handover stress zones (N+ handovers within a rolling time window)

### Signal Quality Thresholds
| Metric | Poor | Meaning |
|---|---|---|
| RSRP | < -90 dBm | Weak signal strength |
| SINR | < 0 dB | Noise dominates signal |
| Signal bars | ≤ 2/5 | Low usable signal |
| Ping gap | > 15 min (default) | Likely signal loss |

### Notebooks (`notebooks/`)
Numbered notebooks (`01_`–`08_`) are exploratory. Build notebooks (`build_*.ipynb`) were used to construct the staged Parquet data. The two `.py` scripts are CLI-runnable equivalents of the most common rebuild tasks.
