# Telco Signal POC

LTE drive-test signal analysis — cell-to-cell handover routes and coverage gap detection.

## Setup

### 1. Add your data
```bash
cp /path/to/SRFG-v1.csv ./data/
```

### 2. Start the environment
```bash
docker compose up
```

### 3. Convert data to Parquet (run once)
Open Jupyter at http://localhost:8888 and run:
```python
import sys; sys.path.insert(0, '/home/jovyan')
from pipeline.ingest import load_raw, clean, to_parquet
to_parquet(clean(load_raw()))
```

### 4. Launch the demo app
Inside the container terminal:
```bash
streamlit run /home/jovyan/app/Home.py
```
Then open http://localhost:8501

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

# telecoms_data
