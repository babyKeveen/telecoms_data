"""
Build coverage gap prediction model from historical handover events.

Trains a HistGradientBoostingClassifier on (lat, lon, hour, dow, month) features
to predict P(poor signal): RSRP < -90 dBm OR avg ping > 300 ms.

Train: Jan-Sep 2025  |  Test: Oct-Dec 2025 (temporal split)

Outputs:
  /home/jovyan/data/models/coverage_gap_model.pkl  — trained model
"""
import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

HANDOVER_DIR = "/home/jovyan/data/stage/handover_events"
COORD_CSV    = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
MODEL_DIR    = Path("/home/jovyan/data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["lat", "lon", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "mon_sin", "mon_cos"]

RSRP_THRESHOLD = -90.0
PING_THRESHOLD = 300.0
SAMPLE_PCT     = 10         # percent of non-null-RSRP events to use


def cyclic_encode(arr, max_val):
    a = np.asarray(arr, dtype=float)
    return np.sin(2 * np.pi * a / max_val), np.cos(2 * np.pi * a / max_val)


# ---------------------------------------------------------------------------
# 1. Load coord lookup into DuckDB
# ---------------------------------------------------------------------------
print("Loading coordinate lookup...")
con = duckdb.connect()
con.execute(f"""
CREATE TABLE coords AS
SELECT
    TRY_CAST(SPLIT_PART(global_cell_id, '-', 3) AS INTEGER) AS cell_id,
    latitude  AS lat,
    longitude AS lon
FROM read_csv_auto('{COORD_CSV}')
WHERE SPLIT_PART(global_cell_id, '-', 1) = '310'
  AND SPLIT_PART(global_cell_id, '-', 2) = '410'
  AND TRY_CAST(SPLIT_PART(global_cell_id, '-', 3) AS INTEGER) IS NOT NULL
""")
n_cells = con.execute("SELECT COUNT(*) FROM coords").fetchone()[0]
print(f"  {n_cells:,} cells in lookup")

# ---------------------------------------------------------------------------
# 2. Extract sampled events with coordinates
# ---------------------------------------------------------------------------
print(f"Extracting {SAMPLE_PCT}% sample of non-null-RSRP events...")
df = con.execute(f"""
WITH sampled AS (
    SELECT
        TRY_CAST(cell_id AS INTEGER)         AS cell_id,
        pci_1_rsrp,
        (ping1 + ping2 + ping3 + ping4) / 4.0 AS avg_ping,
        CAST(EXTRACT(hour  FROM event_ts) AS INTEGER) AS hour_of_day,
        CAST(EXTRACT(dow   FROM event_ts) AS INTEGER) AS day_of_week,
        CAST(EXTRACT(month FROM event_ts) AS INTEGER) AS month
    FROM read_parquet('{HANDOVER_DIR}/event_date=*/*.parquet', hive_partitioning=true)
    WHERE pci_1_rsrp IS NOT NULL
    USING SAMPLE {SAMPLE_PCT} PERCENT
)
SELECT s.cell_id, s.pci_1_rsrp, s.avg_ping,
       s.hour_of_day, s.day_of_week, s.month,
       c.lat, c.lon
FROM sampled s
JOIN coords c ON s.cell_id = c.cell_id
""").df()

print(f"  {len(df):,} events loaded")

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
df["is_gap"] = (df["pci_1_rsrp"] < RSRP_THRESHOLD) | (df["avg_ping"] > PING_THRESHOLD)
df["hour_sin"], df["hour_cos"] = cyclic_encode(df["hour_of_day"], 24)
df["dow_sin"],  df["dow_cos"]  = cyclic_encode(df["day_of_week"],  7)
df["mon_sin"],  df["mon_cos"]  = cyclic_encode(df["month"],        12)
df = df.dropna(subset=FEATURES)

gap_rate = df["is_gap"].mean()
print(f"  Gap rate: {gap_rate:.1%}  ({df['is_gap'].sum():,} / {len(df):,})")

# ---------------------------------------------------------------------------
# 4. Train / test split (temporal)
# ---------------------------------------------------------------------------
train = df[df["month"].between(1, 9)]
test  = df[df["month"].between(10, 12)]
X_train, y_train = train[FEATURES].values, train["is_gap"].values.astype(int)
X_test,  y_test  = test[FEATURES].values,  test["is_gap"].values.astype(int)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ---------------------------------------------------------------------------
# 5. Train model
# ---------------------------------------------------------------------------
print("Training HistGradientBoostingClassifier...")
model = HistGradientBoostingClassifier(
    max_iter=300,
    max_depth=6,
    learning_rate=0.05,
    min_samples_leaf=50,
    random_state=42,
    verbose=1,
)
model.fit(X_train, y_train)

print("\n--- Test set classification report ---")
print(classification_report(y_test, model.predict(X_test), target_names=["ok", "gap"]))

# ---------------------------------------------------------------------------
# 6. Save model
# ---------------------------------------------------------------------------
out_path = MODEL_DIR / "coverage_gap_model.pkl"
with open(out_path, "wb") as f:
    pickle.dump({"model": model, "features": FEATURES,
                 "rsrp_threshold": RSRP_THRESHOLD,
                 "ping_threshold": PING_THRESHOLD}, f)
print(f"Model saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
print("\nDone. Run the Streamlit app to use page 10 (Coverage Forecast).")
