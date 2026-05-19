"""
vectors.py
----------
FAISS-backed vector database for trip similarity search and NL queries.

Features per trip (8-dimensional):
  duration_minutes, n_cells, n_handovers, n_events,
  first_lat, first_lon, last_lat, last_lon

Build:  python notebooks/build_vector_db.py
Load:   index, meta, stats = vectors.load()
Query:  vectors.nl_query(index, meta, stats, "long trips starting near Detroit")
"""
import json
from pathlib import Path

import duckdb
import faiss
import numpy as np
import pandas as pd
from loguru import logger

VECTOR_DIR = Path("/home/jovyan/data/vector_db")
COORD_CSV  = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
TRIPS_DIR  = "/home/jovyan/data/stage/trips"

FEATURES = [
    "duration_minutes", "n_cells", "n_handovers", "n_events",
    "first_lat", "first_lon", "last_lat", "last_lon",
]

# City centres used to help Claude resolve geographic references in NL queries.
CITIES = {
    "Ann Arbor":    (42.281, -83.748), "Atlanta":      (33.749, -84.388),
    "Baltimore":    (39.290, -76.612), "Boston":       (42.360, -71.059),
    "Charlotte":    (35.227, -80.843), "Chicago":      (41.878, -87.630),
    "Cincinnati":   (39.103, -84.512), "Cleveland":    (41.499, -81.695),
    "Columbus":     (39.961, -82.999), "Dallas":       (32.776, -96.797),
    "Denver":       (39.739,-104.984), "Detroit":      (42.331, -83.046),
    "Houston":      (29.760, -95.370), "Indianapolis": (39.768, -86.158),
    "Kansas City":  (39.099, -94.578), "Los Angeles":  (34.052,-118.244),
    "Louisville":   (38.252, -85.758), "Memphis":      (35.149, -90.048),
    "Milwaukee":    (43.038, -87.906), "Minneapolis":  (44.977, -93.265),
    "Nashville":    (36.162, -86.781), "New York":     (40.713, -74.006),
    "Philadelphia": (39.952, -75.165), "Pittsburgh":   (40.440, -79.996),
    "Portland":     (45.523,-122.676), "San Francisco":(37.774,-122.419),
    "Seattle":      (47.606,-122.332), "St. Louis":    (38.627, -90.197),
    "Washington":   (38.907, -77.037),
}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _load_coord_lookup() -> pd.DataFrame:
    df = pd.read_csv(COORD_CSV, usecols=["global_cell_id", "latitude", "longitude"])
    rows = []
    for row in df.itertuples(index=False):
        parts = str(row.global_cell_id).split("-")
        if len(parts) < 3:
            continue
        try:
            if int(parts[0]) == 310 and int(parts[1]) == 410:
                rows.append((int(parts[2]), float(row.latitude), float(row.longitude)))
        except ValueError:
            continue
    return pd.DataFrame(rows, columns=["cell_id", "lat", "lon"])


def build(output_dir: Path = VECTOR_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading trips from Parquet...")
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT trip_id, vehicle_id, trip_start, trip_end, dominant_rat,
               duration_minutes, n_handovers, n_cells, n_events,
               first_cell, last_cell
        FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
        WHERE duration_minutes > 0
    """).df()
    logger.info(f"Loaded {len(df):,} trips")

    logger.info("Joining cell coordinates...")
    coords = _load_coord_lookup()
    df = (
        df
        .merge(coords.rename(columns={"cell_id": "first_cell", "lat": "first_lat", "lon": "first_lon"}),
               on="first_cell", how="left")
        .merge(coords.rename(columns={"cell_id": "last_cell", "lat": "last_lat", "lon": "last_lon"}),
               on="last_cell", how="left")
    )
    for col in ["first_lat", "first_lon", "last_lat", "last_lon"]:
        df[col] = df[col].fillna(df[col].median())

    # Build & normalise feature matrix
    feat = df[FEATURES].astype(np.float32)
    stats: dict[str, dict] = {}
    for col in FEATURES:
        mn, mx = float(feat[col].min()), float(feat[col].max())
        stats[col] = {"min": mn, "max": mx}
        feat[col] = (feat[col] - mn) / (mx - mn + 1e-9)

    vectors = feat.values  # shape (N, 8)

    logger.info(f"Building FAISS IndexFlatL2 ({vectors.shape})...")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    logger.info(f"Index ready: {index.ntotal:,} vectors")

    faiss.write_index(index, str(output_dir / "trips.index"))

    meta_cols = [
        "trip_id", "vehicle_id", "trip_start", "trip_end", "dominant_rat",
        "duration_minutes", "n_cells", "n_handovers", "n_events",
        "first_cell", "last_cell", "first_lat", "first_lon", "last_lat", "last_lon",
    ]
    df[meta_cols].reset_index(drop=True).to_parquet(output_dir / "metadata.parquet", index=True)

    with open(output_dir / "norm_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Vector DB saved to {output_dir}")


# ---------------------------------------------------------------------------
# Load & search
# ---------------------------------------------------------------------------

def load(output_dir: Path = VECTOR_DIR):
    """Return (index, metadata_df, norm_stats)."""
    index = faiss.read_index(str(output_dir / "trips.index"))
    metadata = pd.read_parquet(output_dir / "metadata.parquet")
    with open(output_dir / "norm_stats.json") as f:
        norm_stats = json.load(f)
    return index, metadata, norm_stats


def _to_normalised_vec(raw: dict, norm_stats: dict) -> np.ndarray:
    arr = []
    for col in FEATURES:
        v = raw.get(col)
        mn, mx = norm_stats[col]["min"], norm_stats[col]["max"]
        if v is None:
            v = (mn + mx) / 2  # unknown → mid-range
        arr.append((float(v) - mn) / (mx - mn + 1e-9))
    return np.clip(np.array(arr, dtype=np.float32), 0.0, 1.0).reshape(1, -1)


def search(index, metadata: pd.DataFrame, norm_stats: dict,
           query_vec: dict, k: int = 10) -> pd.DataFrame:
    """Search by explicit feature dict, e.g. {"duration_minutes": 120, "first_lat": 42.3}."""
    vec = _to_normalised_vec(query_vec, norm_stats)
    distances, indices = index.search(vec, k)
    results = metadata.iloc[indices[0]].copy()
    results["score"] = distances[0]
    return results.reset_index(drop=True)


def nl_query(index, metadata: pd.DataFrame, norm_stats: dict,
             question: str, k: int = 10) -> pd.DataFrame:
    """Translate a natural-language question into a feature vector and search."""
    import anthropic
    client = anthropic.Anthropic()

    ranges = {col: {"min": round(v["min"], 3), "max": round(v["max"], 3)}
              for col, v in norm_stats.items()}
    city_hint = "\n".join(f"  {name}: lat={lat}, lon={lon}"
                          for name, (lat, lon) in CITIES.items())

    system_text = f"""You parse natural-language trip queries into JSON feature vectors for a FAISS nearest-neighbour search.

Feature schema and real-world ranges:
{json.dumps(ranges, indent=2)}

City reference coordinates (lat, lon):
{city_hint}

Rules:
- Return ONLY a JSON object with keys from the schema. No explanation.
- Omit keys that the query gives no signal about (they default to mid-range).
- For geographic references ("near Detroit", "starting in Chicago") set first_lat/first_lon or last_lat/last_lon accordingly.
- For vague magnitude ("long trip", "many handovers") choose a value in the top 25% of the range.
- For "short" / "few" choose a value in the bottom 25%.
"""

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": question}],
    )
    raw = json.loads(resp.content[0].text)
    logger.info(f"NL→vector: {raw}")
    return search(index, metadata, norm_stats, raw, k=k)
