#!/usr/bin/env python3
"""Rebuild top20_trips_mapped.json and top20_trips_map.html.

Usage:
    python rebuild_trips.py [--max-hours HOURS]

    --max-hours   Maximum trip duration to include (default: 4.0)
"""

import argparse
import json

import duckdb
import folium
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--max-hours", type=float, default=4.0,
                    help="Maximum trip duration in hours (default: 4.0)")
parser.add_argument("--min-cells-per-hour", type=float, default=3.0,
                    help="Minimum cell transitions per hour (default: 3.0)")
parser.add_argument("--start-date", type=str, default=None,
                    help="Inclusive start date filter, e.g. 2025-03-01")
parser.add_argument("--end-date", type=str, default=None,
                    help="Inclusive end date filter, e.g. 2025-03-31")
args = parser.parse_args()

COORD_CSV = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
TRIPS_DIR = "/home/jovyan/data/stage/trips/"
OUT_DIR = "/home/jovyan/telco-poc/notebooks"

# ---------------------------------------------------------------------------
# Step 1: Build coord lookup (MCC=310, MNC=410, US AT&T)
# ---------------------------------------------------------------------------
print("Step 1: Loading coordinate dataset...", flush=True)
coord_df = pd.read_csv(COORD_CSV, usecols=["global_cell_id", "latitude", "longitude"])
print(f"  Total rows in CSV: {len(coord_df):,}", flush=True)

coord_lookup: dict[int, tuple[float, float]] = {}
for row in coord_df.itertuples(index=False):
    parts = str(row.global_cell_id).split("-")
    if len(parts) < 3:
        continue
    try:
        if int(parts[0]) == 310 and int(parts[1]) == 410:
            coord_lookup[int(parts[2])] = (float(row.latitude), float(row.longitude))
    except ValueError:
        continue

print(f"  US AT&T cells mapped: {len(coord_lookup):,}", flush=True)

# ---------------------------------------------------------------------------
# Step 2: Fetch top-500 trips by n_cells via DuckDB (avoids loading 23M rows)
# ---------------------------------------------------------------------------
print("\nStep 2: Querying top-500 trips by n_cells...", flush=True)
con = duckdb.connect()
max_minutes = args.max_hours * 60

date_clauses = []
if args.start_date:
    date_clauses.append(f"trip_start >= '{args.start_date}'")
if args.end_date:
    date_clauses.append(f"trip_start < date '{args.end_date}' + interval 1 day")
date_filter = ("AND " + " AND ".join(date_clauses)) if date_clauses else ""

print(f"  Max duration: {args.max_hours}h  "
      f"Min cells/hour: {args.min_cells_per_hour}  "
      f"Date range: {args.start_date or 'any'} → {args.end_date or 'any'}", flush=True)

candidates = con.execute(f"""
    SELECT trip_id, trip_start, trip_end, duration_minutes,
           n_handovers, n_cells, cell_sequence
    FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
    WHERE duration_minutes > 0
      AND duration_minutes <= {max_minutes}
      AND (n_cells / (duration_minutes / 60.0)) >= {args.min_cells_per_hour}
    {date_filter}
    ORDER BY n_cells DESC
    LIMIT 500
""").df()
print(f"  Fetched {len(candidates):,} candidates  (max n_cells={candidates['n_cells'].max()})", flush=True)

# ---------------------------------------------------------------------------
# Step 3: Find top 20 with 100% coordinate coverage
# ---------------------------------------------------------------------------
print("\nStep 3: Resolving coordinates...", flush=True)

top20 = []
checked = 0
for _, row in candidates.iterrows():
    checked += 1
    cell_ids = []
    for tok in str(row["cell_sequence"]).split("->"):
        try:
            cell_ids.append(int(tok.strip()))
        except ValueError:
            pass

    coords = []
    all_mapped = True
    for cid in cell_ids:
        if cid in coord_lookup:
            lat, lon = coord_lookup[cid]
            coords.append({"cell_id": cid, "lat": lat, "lon": lon})
        else:
            all_mapped = False
            break

    if not all_mapped or not coords:
        continue

    top20.append({
        "trip_id": row["trip_id"],
        "trip_start": str(row["trip_start"]),
        "trip_end": str(row["trip_end"]),
        "duration_minutes": float(row["duration_minutes"]) if pd.notna(row["duration_minutes"]) else None,
        "n_handovers": int(row["n_handovers"]) if pd.notna(row["n_handovers"]) else None,
        "n_cells": int(row["n_cells"]),
        "coordinates": coords,
    })

    if len(top20) == 20:
        break

print(f"  Checked: {checked}  Top-20 found: {len(top20)}", flush=True)
if len(top20) < 20:
    print(f"  WARNING: only {len(top20)} trips have 100% coordinate coverage", flush=True)

# ---------------------------------------------------------------------------
# Step 4: Save top20_trips_mapped.json
# ---------------------------------------------------------------------------
print("\nStep 4: Saving top20_trips_mapped.json...", flush=True)
out_json = f"{OUT_DIR}/top20_trips_mapped.json"
with open(out_json, "w") as fh:
    json.dump(top20, fh, indent=2)
print(f"  Saved: {out_json}", flush=True)

# ---------------------------------------------------------------------------
# Step 5: Build top20_trips_map.html
# ---------------------------------------------------------------------------
print("\nStep 5: Building top20_trips_map.html...", flush=True)

COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]

all_lats = [c["lat"] for trip in top20 for c in trip["coordinates"]]
all_lons = [c["lon"] for trip in top20 for c in trip["coordinates"]]
map_center = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]

tm = folium.Map(location=map_center, zoom_start=6)

for i, trip in enumerate(top20):
    colour = COLOURS[i % len(COLOURS)]
    coords_ll = [[c["lat"], c["lon"]] for c in trip["coordinates"]]

    folium.PolyLine(
        locations=coords_ll,
        color=colour,
        weight=3,
        opacity=0.85,
        tooltip=f'{trip["trip_id"]}  n_cells={trip["n_cells"]}',
    ).add_to(tm)

    # Open circle = start
    folium.CircleMarker(
        location=coords_ll[0],
        radius=6,
        color=colour,
        fill=False,
        weight=2,
        tooltip=f'START: {trip["trip_start"]}',
    ).add_to(tm)

    # Filled circle = end
    folium.CircleMarker(
        location=coords_ll[-1],
        radius=6,
        color=colour,
        fill=True,
        fill_color=colour,
        fill_opacity=1.0,
        weight=2,
        tooltip=f'END: {trip["trip_end"]}',
    ).add_to(tm)

out_trips_map = f"{OUT_DIR}/top20_trips_map.html"
tm.save(out_trips_map)
print(f"  Saved: {out_trips_map}", flush=True)

print("\nAll done.", flush=True)
