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

**Full pipeline rebuild** (run in order inside the container):
```bash
# 1. Re-stage handover events from raw daily files (clears existing stage first)
rm -rf /home/jovyan/data/stage/handover_events/
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=21600 \
    --output notebooks/build_stage_duckdb_out.ipynb notebooks/build_stage_duckdb.ipynb

# 2. Rebuild trips from handover events (~20-40 min)
python notebooks/rebuild_trips_stage.py

# 3. Rebuild numpy vector DB
python notebooks/build_vector_db_np.py
```

Monitor handover events rebuild progress (updates after every file):
```bash
cat /tmp/stage_progress.json
```

**Partial rebuild** (trip outputs/map only):
```bash
python notebooks/rebuild_trips.py [--max-hours 4.0] [--start-date 2025-07-01] [--end-date 2025-07-31]
python notebooks/rebuild_outputs.py
```

## Architecture

### Data Flow
```
C:\S3Data\sim_test_decoded\2025\*.gz   (364 daily files, ~100 cols, headerless CSV)
          ↓
  build_stage_duckdb.ipynb   →   stage/handover_events/event_date=*/*.parquet
          ↓
  build_trips_duckDB.ipynb   →   stage/trips/event_date=*/*.parquet
          ↓
  build_vector_db_np.py      →   vector_db_np/  (numpy similarity index)
```

The staged Parquet data lives outside the repo under `/home/jovyan/data/stage/` in hive-partitioned layouts:
- `stage/handover_events/event_date=*/*.parquet` — one row per ping event with `vehicle_id`, `event_ts`, `cell_id`, `rat`, `pci_1_rsrp`, `pci_1_rsrq`, `ping1`–`ping4`
- `stage/trips/event_date=*/*.parquet` — one row per trip with `cell_sequence` and KPI aggregates (see below)

**Raw source columns extracted** (by position, zero-based):

| Position | Column | Stage field |
|---|---|---|
| 03 | collection_time | event_ts |
| 05 | sim_id | vehicle_id |
| 13 | imsi | imsi |
| 18 | cell_id | cell_id |
| 23 | technology | rat |
| 34 | pci_1_rsrp | pci_1_rsrp |
| 35 | pci_1_rsrq | pci_1_rsrq |
| 84–87 | ping1–ping4 | ping1–ping4 |

**Trip-level KPI columns** (aggregated in `build_trips_duckDB.ipynb`):

| Column | Derivation |
|---|---|
| `avg_neighbor_rsrp` | AVG(pci_1_rsrp) across trip events |
| `min_neighbor_rsrp` | MIN(pci_1_rsrp) across trip events |
| `avg_neighbor_rsrq` | AVG(pci_1_rsrq) across trip events |
| `avg_ping_ms` | AVG((ping1+ping2+ping3+ping4)/4) across trip events |

Raw per-event KPI values are retained in `handover_events` for event-level queries — join on `(vehicle_id, event_ts BETWEEN trip_start AND trip_end)`.

### Querying
Both Streamlit pages and the CLI scripts use **DuckDB with no persistent DB file** — they call `duckdb.connect()` and query directly against Parquet via `read_parquet(..., hive_partitioning=true)`. There is no ORM or database schema to maintain.

### Cell Coordinate Lookup
`/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv` maps `global_cell_id` (format: `MCC-MNC-cell_id`, e.g. `310-410-12345`) to lat/lon. Both app pages filter to MCC=310, MNC=410 (US AT&T) and build a `dict[int, (lat, lon)]` keyed by cell_id integer. This lookup is `@st.cache_data`-decorated in the app.

### Streamlit App (`app/`)
- `Home.py` — landing page only
- `pages/1_Route_Map.py` — queries the trips Parquet, resolves cell IDs to coordinates via the lookup, renders folium PolyLines
- `pages/2_Gap_Analysis.py` — three tabs: (1) Silence Gaps — ping intervals > expected cadence (~485s) mapped as CircleMarkers sized by event count; (2) Signal Quality — RSRP/SINR/signal bar threshold analysis from raw events; (3) Neighbour Signal — `avg_neighbor_rsrp`/`rsrq` per trip origin cell from trips Parquet, mapped with red→green colour scale.
- `pages/3_Corridor_Analysis.py` — city-pair corridor view: filters trips whose first/last cells fall within a configurable bounding box around two chosen cities, then clips each trip's `cell_sequence` to a perpendicular corridor around the A→B line (rejects trips where <60% of cells are on-corridor). City cell sets are resolved in Python from `coord_lookup`; the DuckDB query pushes `first_cell`/`last_cell` filtering into SQL via registered DataFrames. Silence-gap hotspots from `handover_events` are optionally overlaid as red CircleMarkers.
- `pages/4_Trip_Search.py` — numpy vector similarity search: sidebar feature toggles (duration, cell count, handovers, events, hour of day, day of week) build a partial-feature query; matched trips rendered as folium PolyLines; results table includes `avg_neighbor_rsrp`, `avg_neighbor_rsrq`, `avg_ping_ms` fetched from trips Parquet.
- `pages/5_Route_Search.py` — route similarity search: user picks an origin and destination from a hardcoded city list (29 US cities); searches the vector DB on `first_lat/lon` + `last_lat/lon` only; optional reverse-direction toggle; city markers pinned on the folium map alongside matched trip PolyLines.
- `pages/6_Temporal_Patterns.py` — fleet activity and network quality broken down by hour of day, day of week, and month. Charts include trip volume, avg duration, active vehicles, handover rate, avg ping latency, and avg neighbour RSRP. Includes an hour × day-of-week activity heatmap. All data from the trips Parquet.

### Vector Database (`/home/jovyan/data/vector_db_np/`)
Numpy-based similarity index over all trips with 10 features: `duration_minutes`, `n_cells`, `n_handovers`, `n_events`, `hour_of_day`, `day_of_week`, `first_lat/lon`, `last_lat/lon`. Built by `pipeline/vectors_np.py` via `notebooks/build_vector_db_np.py`. Used by pages 4 and 5.

NL query support is in `pipeline/vectors.py` (FAISS variant, requires `ANTHROPIC_API_KEY`). Uses `claude-haiku-4-5` with prompt caching on the system prompt to minimise token cost.

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
Numbered notebooks (`01_`–`08_`) are exploratory. Key build notebooks and scripts:

| File | Purpose |
|---|---|
| `build_stage_duckdb.ipynb` | Stage raw daily files → `handover_events` Parquet; resumable per-file; writes `/tmp/stage_progress.json` for progress monitoring |
| `build_trips_duckDB.ipynb` | Detect trips day-by-day from `handover_events` → `trips` Parquet |
| `rebuild_trips_stage.py` | CLI wrapper: clears `stage/trips/` and re-executes `build_trips_duckDB.ipynb` |
| `build_vector_db_np.py` | Rebuild numpy vector DB from trips Parquet |
| `rebuild_trips.py` | Regenerate `top20_trips_mapped.json` and HTML map from existing trips Parquet |
| `rebuild_outputs.py` | Rebuild other JSON/HTML output artefacts |
