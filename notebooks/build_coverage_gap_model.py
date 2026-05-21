"""
Build coverage gap prediction model from historical handover events.

Aggregates to one row per cell (overall gap rate across full year).
Trains HistGradientBoostingClassifier on (lat, lon) + time context
to predict P(gap): RSRP < -90 dBm OR avg_ping > 300 ms.

The time features (hour, dow) are supplied at inference time by the app;
the model learns which geographic locations are structurally prone to gaps,
modulated by time-of-day context from the training distribution.

Outputs:
  /home/jovyan/data/models/coverage_gap_model.pkl
"""
import pickle
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

def log(msg):
    print(msg, flush=True)

HANDOVER_DIR    = "/home/jovyan/data/stage/handover_events"
COORD_CSV       = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
MODEL_DIR       = Path("/home/jovyan/data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES        = ["lat", "lon", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
RSRP_THRESHOLD  = -90.0
PING_THRESHOLD  = 300.0


def cyclic_encode(arr, max_val):
    a = np.asarray(arr, dtype=float)
    return np.sin(2 * np.pi * a / max_val), np.cos(2 * np.pi * a / max_val)


# ---------------------------------------------------------------------------
# 1. Build coord table in DuckDB
# ---------------------------------------------------------------------------
log("Loading coordinate lookup...")
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
log(f"  {n_cells:,} cells in lookup")

# ---------------------------------------------------------------------------
# 2. Aggregate to one row per (cell, hour_of_day, day_of_week) — 168 buckets
#    but only across cells that have >= 5 observations (active cells)
#    Limit to Q1-Q3 for training, Q4 for test
# ---------------------------------------------------------------------------
log("Aggregating events by cell × hour × dow (train: Jan-Sep, test: Oct-Dec)...")
log("  Scanning 364 parquet files — may take 5-10 min...")

df = con.execute(f"""
SELECT
    c.cell_id,
    c.lat,
    c.lon,
    CAST(EXTRACT(hour FROM e.event_ts) AS INTEGER) AS hour_of_day,
    CAST(EXTRACT(dow  FROM e.event_ts) AS INTEGER) AS day_of_week,
    -- month omitted: not a model feature, dropping it reduces rows ~12x
    COUNT(*) AS n_events,
    AVG((e.ping1 + e.ping2 + e.ping3 + e.ping4) / 4.0) AS avg_ping_ms
FROM read_parquet('{HANDOVER_DIR}/event_date=*/*.parquet', hive_partitioning=true) e
JOIN coords c ON TRY_CAST(e.cell_id AS INTEGER) = c.cell_id
WHERE e.ping1 IS NOT NULL
GROUP BY c.cell_id, c.lat, c.lon,
         EXTRACT(hour FROM e.event_ts),
         EXTRACT(dow  FROM e.event_ts)
HAVING COUNT(*) >= 5
""").df()

log(f"  {len(df):,} active (cell × hour × dow) buckets")
log(f"  {df['cell_id'].nunique():,} unique cells")

# ---------------------------------------------------------------------------
# 3. Feature engineering — one training row per bucket
# ---------------------------------------------------------------------------
log("Engineering features...")
df = df.dropna(subset=["avg_ping_ms"])

# Label: above-median latency → "high latency cell/time"
# ~50/50 split; P(high latency) > 0.5 = worse than fleet median
median_ping  = float(df["avg_ping_ms"].median())
log(f"  Fleet median avg_ping_ms: {median_ping:.1f} ms")
df["is_gap"] = df["avg_ping_ms"] > median_ping
df["hour_sin"], df["hour_cos"] = cyclic_encode(df["hour_of_day"], 24)
df["dow_sin"],  df["dow_cos"]  = cyclic_encode(df["day_of_week"],  7)
df = df.dropna(subset=FEATURES)

log(f"  Overall gap rate: {df['is_gap'].mean():.1%}  ({df['is_gap'].sum():,} / {len(df):,} buckets)")

# ---------------------------------------------------------------------------
# 4. Train / test split (random 80/20 — no month dimension in data)
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
X = df[FEATURES].values
y = df["is_gap"].values.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
log("Training HistGradientBoostingClassifier...")
model = HistGradientBoostingClassifier(
    max_iter=200, max_depth=6, learning_rate=0.05,
    min_samples_leaf=20, random_state=42, verbose=1,
)
model.fit(X_train, y_train)

log("\n--- Test set classification report ---")
log(classification_report(y_test, model.predict(X_test), target_names=["ok", "gap"]))

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
out_path = MODEL_DIR / "coverage_gap_model.pkl"
with open(out_path, "wb") as f:
    pickle.dump({"model": model, "features": FEATURES,
                 "rsrp_threshold": RSRP_THRESHOLD,
                 "ping_threshold": PING_THRESHOLD}, f)
log(f"Model saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
log("Done.")
