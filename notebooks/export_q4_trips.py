"""Export all Q4 2025 trips with full KPI + ping buckets to a single JSON file."""
from pathlib import Path

import duckdb
import pandas as pd

TRIPS_DIR  = "/home/jovyan/data/stage/trips"
COORD_CSV  = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
OUT_FILE   = Path("/home/jovyan/telco-poc/exports/trips_q4_2025.json")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

START_DATE = "2025-10-01"
END_DATE   = "2025-12-31"

PING_COL_MAP = [
    ("ping_le_100",  "ping ≤100"),
    ("ping_101_150", "ping 101-150"),
    ("ping_151_200", "ping 151-200"),
    ("ping_201_250", "ping 201-250"),
    ("ping_251_300", "ping 251-300"),
    ("ping_301_350", "ping 301-350"),
    ("ping_351_400", "ping 351-400"),
    ("ping_401_450", "ping 401-450"),
    ("ping_451_500", "ping 451-500"),
    ("ping_gt_500",  "ping >500"),
]

# Build coord lookup
print("Loading coordinate lookup...")
coord_df = pd.read_csv(COORD_CSV, usecols=["global_cell_id", "latitude", "longitude"])
coord_lookup = {}
for row in coord_df.itertuples(index=False):
    parts = str(row.global_cell_id).split("-")
    if len(parts) >= 3:
        try:
            if int(parts[0]) == 310 and int(parts[1]) == 410:
                coord_lookup[int(parts[2])] = (float(row.latitude), float(row.longitude))
        except ValueError:
            pass
print(f"  {len(coord_lookup):,} cells loaded")

# Query trips
print(f"Querying trips {START_DATE} → {END_DATE}...")
con = duckdb.connect()
df = con.execute(f"""
    SELECT trip_id, vehicle_id, trip_start, trip_end,
           duration_minutes, n_cells, n_handovers, n_events,
           first_cell, last_cell, dominant_rat,
           avg_neighbor_rsrp, min_neighbor_rsrp,
           avg_neighbor_rsrq, avg_ping_ms,
           ping_le_100, ping_101_150, ping_151_200, ping_201_250, ping_251_300,
           ping_301_350, ping_351_400, ping_401_450, ping_451_500, ping_gt_500
    FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
    WHERE event_date BETWEEN '{START_DATE}' AND '{END_DATE}'
    ORDER BY trip_start
""").df()
print(f"  {len(df):,} trips loaded")

# Resolve coordinates
print("Resolving coordinates...")
def get_lat(cell):
    try:
        return round(coord_lookup[int(cell)][0], 1) if pd.notna(cell) else None
    except (KeyError, ValueError):
        return None

def get_lon(cell):
    try:
        return round(coord_lookup[int(cell)][1], 1) if pd.notna(cell) else None
    except (KeyError, ValueError):
        return None

df["start_lat"] = df["first_cell"].map(get_lat)
df["start_lon"] = df["first_cell"].map(get_lon)
df["end_lat"]   = df["last_cell"].map(get_lat)
df["end_lon"]   = df["last_cell"].map(get_lon)

# Build export table
print("Building export table...")
table = pd.DataFrame({
    "trip_id":        df["trip_id"],
    "vehicle_id":     df["vehicle_id"],
    "trip_start":     df["trip_start"].astype(str).str[:19],
    "trip_end":       df["trip_end"].astype(str).str[:19],
    "duration (h)":   (df["duration_minutes"] / 60).round(2),
    "n_cells":        df["n_cells"].astype("Int64"),
    "n_handovers":    df["n_handovers"].astype("Int64"),
    "n_events":       df["n_events"].astype("Int64"),
    "dominant_rat":   df["dominant_rat"],
    "avg_rsrp (dBm)": df["avg_neighbor_rsrp"].round(2),
    "min_rsrp (dBm)": df["min_neighbor_rsrp"].round(2),
    "avg_rsrq (dB)":  df["avg_neighbor_rsrq"].round(2),
    "avg_ping (ms)":  df["avg_ping_ms"].round(1),
    "start_lat":      df["start_lat"],
    "start_lon":      df["start_lon"],
    "end_lat":        df["end_lat"],
    "end_lon":        df["end_lon"],
})
for col, label in PING_COL_MAP:
    table[label] = df[col].fillna(0).astype(int)

print(f"Writing to {OUT_FILE} ...")
OUT_FILE.write_text(table.to_json(orient="records", indent=2))
size_mb = OUT_FILE.stat().st_size / 1e6
print(f"Done. {len(table):,} trips — {size_mb:.1f} MB")
