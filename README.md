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

## Project Structure
```
telco-poc/
├── data/               # Dataset lives here (gitignored)
├── pipeline/           # Core logic — ingest, handovers, gaps
├── notebooks/          # Exploration & analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_handover_analysis.ipynb
│   └── 03_gap_detection.ipynb
├── app/                # Streamlit demo
│   ├── Home.py
│   └── pages/
│       ├── 1_Route_Map.py
│       └── 2_Gap_Analysis.py
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

## Use Cases
| # | Use Case | Key Output |
|---|---|---|
| 1 | Most used routes | Cell-to-cell handover graph, weighted by frequency |
| 2 | Coverage gaps | Map of poor RSRP / negative SINR zones per route |

## Signal Quality Thresholds
| Metric | Poor threshold | Meaning |
|---|---|---|
| RSRP | < -90 dBm | Weak signal strength |
| SINR | < 0 dB | Noise dominates signal |
| Signal bars | ≤ 2 / 5 | Low usable signal |
