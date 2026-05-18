"""
vectors_np.py
-------------
Numpy-based vector database for trip similarity search.

Features per trip (10-dimensional):
  duration_minutes, n_cells, n_handovers, n_events,
  hour_of_day, day_of_week,
  first_lat, first_lon, last_lat, last_lon

Build:  python notebooks/build_vector_db_np.py
Load:   vectors, meta, stats = vectors_np.load()
Search: results = vectors_np.search(vectors, meta, stats, {"duration_minutes": 60})
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

VECTOR_DIR = Path("/home/jovyan/data/vector_db_np")
COORD_CSV  = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
TRIPS_DIR  = "/home/jovyan/data/stage/trips"

FEATURES = [
    "duration_minutes", "n_cells", "n_handovers", "n_events",
    "hour_of_day", "day_of_week",
    "first_lat", "first_lon", "last_lat", "last_lon",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_coord_lookup() -> dict:
    df = pd.read_csv(COORD_CSV, usecols=["global_cell_id", "latitude", "longitude"])
    lookup = {}
    for row in df.itertuples(index=False):
        parts = str(row.global_cell_id).split("-")
        if len(parts) < 3:
            continue
        try:
            if int(parts[0]) == 310 and int(parts[1]) == 410:
                lookup[int(parts[2])] = (float(row.latitude), float(row.longitude))
        except ValueError:
            continue
    return lookup


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(output_dir: Path = VECTOR_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trips from hive parquet...")
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT vehicle_id, trip_id, trip_start, trip_end,
               duration_minutes, n_handovers, n_cells, n_events,
               first_cell, last_cell
        FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
        WHERE duration_minutes > 0
    """).df()
    print(f"  {len(df):,} trips across {df['vehicle_id'].nunique():,} vehicles")

    print("Building cell coordinate lookup...")
    coord_lookup = _build_coord_lookup()
    print(f"  {len(coord_lookup):,} AT&T cells indexed")

    # Temporal features
    df["hour_of_day"] = pd.to_datetime(df["trip_start"]).dt.hour.astype(float)
    df["day_of_week"] = pd.to_datetime(df["trip_start"]).dt.dayofweek.astype(float)

    # Geographic features via direct first_cell / last_cell integers
    df["first_lat"] = df["first_cell"].map(lambda c: coord_lookup.get(c, (None, None))[0])
    df["first_lon"] = df["first_cell"].map(lambda c: coord_lookup.get(c, (None, None))[1])
    df["last_lat"]  = df["last_cell"].map( lambda c: coord_lookup.get(c, (None, None))[0])
    df["last_lon"]  = df["last_cell"].map( lambda c: coord_lookup.get(c, (None, None))[1])

    for col in ["first_lat", "first_lon", "last_lat", "last_lon"]:
        n_missing = df[col].isna().sum()
        if n_missing:
            print(f"  {col}: {n_missing:,} missing → median fill")
        df[col] = df[col].fillna(df[col].median())

    # Build feature matrix and normalise to [0, 1]
    feat = df[FEATURES].astype(np.float32)
    stats: dict = {}
    for col in FEATURES:
        mn = float(feat[col].min())
        mx = float(feat[col].max())
        stats[col] = {"min": mn, "max": mx}
        feat[col] = (feat[col] - mn) / (mx - mn + 1e-9)

    vectors = feat.values
    print(f"  Feature matrix: {vectors.shape}")

    np.save(output_dir / "vectors.npy", vectors)

    meta_cols = [
        "trip_id", "vehicle_id", "trip_start", "trip_end",
        "duration_minutes", "n_cells", "n_handovers", "n_events",
        "first_lat", "first_lon", "last_lat", "last_lon",
    ]
    df[meta_cols].reset_index(drop=True).to_parquet(output_dir / "metadata.parquet", index=True)

    with open(output_dir / "norm_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Vector DB saved to {output_dir}")


# ---------------------------------------------------------------------------
# Load & search
# ---------------------------------------------------------------------------

def load(output_dir: Path = VECTOR_DIR) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Return (vectors, metadata_df, norm_stats)."""
    vectors  = np.load(output_dir / "vectors.npy")
    metadata = pd.read_parquet(output_dir / "metadata.parquet")
    with open(output_dir / "norm_stats.json") as f:
        norm_stats = json.load(f)
    return vectors, metadata, norm_stats


def _to_normalised_vec(query: dict, norm_stats: dict) -> np.ndarray:
    arr = []
    for col in FEATURES:
        mn = norm_stats[col]["min"]
        mx = norm_stats[col]["max"]
        if col in query:
            v = (float(query[col]) - mn) / (mx - mn + 1e-9)
        else:
            v = 0.5  # unknown → mid-range
        arr.append(v)
    return np.clip(np.array(arr, dtype=np.float32), 0.0, 1.0)


def search(
    vectors: np.ndarray,
    metadata: pd.DataFrame,
    norm_stats: dict,
    query: dict,
    k: int = 10,
    mask_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Find the k most similar trips to a feature dict query.

    query         – dict with any subset of FEATURES as keys
    mask_features – if given, only these features contribute to distance
    """
    q = _to_normalised_vec(query, norm_stats)

    if mask_features:
        idx = [FEATURES.index(f) for f in mask_features if f in FEATURES]
        vecs = vectors[:, idx]
        q    = q[idx]
    else:
        vecs = vectors

    dists = np.linalg.norm(vecs - q, axis=1)
    top_k = np.argsort(dists)[:k]

    results = metadata.iloc[top_k].copy()
    results["distance"] = dists[top_k]
    return results.reset_index(drop=True)
