#!/usr/bin/env python3
"""Rebuild cell_tower_map.html, top20_trips_mapped.json, top20_trips_map.html."""

import glob
import json
import multiprocessing
from collections import Counter

import folium
import pandas as pd
import pyarrow.parquet as pq
from folium.plugins import HeatMap, MarkerCluster

COORD_CSV = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
HANDOVER_DIR = "/home/jovyan/data/stage/handover_events/"
TRIPS_DIR = "/home/jovyan/data/stage/trips/"
OUT_DIR = "/home/jovyan/telco-poc/notebooks"

# ---------------------------------------------------------------------------
# Step 1: Build cell_id -> (lat, lon) lookup (MCC=310, MNC=410, US AT&T)
# ---------------------------------------------------------------------------
print("Step 1: Loading coordinate dataset...", flush=True)
coord_df = pd.read_csv(COORD_CSV, usecols=["global_cell_id", "latitude", "longitude"])
print(f"  Total rows in CSV: {len(coord_df):,}", flush=True)

coord_lookup: dict[int, tuple[float, float]] = {}
skipped = 0
for row in coord_df.itertuples(index=False):
    gid = str(row.global_cell_id)
    parts = gid.split("-")
    if len(parts) < 3:
        skipped += 1
        continue
    try:
        mcc, mnc = int(parts[0]), int(parts[1])
    except ValueError:
        skipped += 1
        continue
    if mcc == 310 and mnc == 410:
        try:
            cell_id = int(parts[2])
            coord_lookup[cell_id] = (float(row.latitude), float(row.longitude))
        except ValueError:
            skipped += 1

print(f"  US AT&T cells mapped: {len(coord_lookup):,}  (skipped malformed: {skipped})", flush=True)

# ---------------------------------------------------------------------------
# Step 2: 4-worker parallel aggregation of handover events -> cell counts
# ---------------------------------------------------------------------------
print("\nStep 2: Aggregating handover events (4 workers)...", flush=True)

handover_files = glob.glob(f"{HANDOVER_DIR}/**/*.parquet", recursive=True)
print(f"  Parquet files found: {len(handover_files):,}", flush=True)


def _count_cells_in_file(path: str) -> dict[int, int]:
    tbl = pq.read_table(path, columns=["cell_id"])
    series = tbl.column("cell_id").to_pylist()
    return dict(Counter(series))


with multiprocessing.Pool(processes=4) as pool:
    results = []
    for i, result in enumerate(pool.imap_unordered(_count_cells_in_file, handover_files, chunksize=10)):
        results.append(result)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(handover_files)} files...", flush=True)

print("  Merging counts...", flush=True)
cell_counts: dict[int, int] = {}
for partial in results:
    for cell_id, cnt in partial.items():
        cell_counts[cell_id] = cell_counts.get(cell_id, 0) + cnt

print(f"  Unique cell IDs in events: {len(cell_counts):,}", flush=True)

# Only keep cells we have coordinates for
mapped_cells = {cid: cnt for cid, cnt in cell_counts.items() if cid in coord_lookup}
print(f"  Cells with coordinates: {len(mapped_cells):,}", flush=True)

# ---------------------------------------------------------------------------
# Step 3: Build cell_tower_map.html
# ---------------------------------------------------------------------------
print("\nStep 3: Building cell_tower_map.html...", flush=True)

rows = [(coord_lookup[cid][0], coord_lookup[cid][1], cnt) for cid, cnt in mapped_cells.items()]
rows.sort(key=lambda x: x[2], reverse=True)

center_lat = sum(r[0] for r in rows) / len(rows)
center_lon = sum(r[1] for r in rows) / len(rows)

m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

# Heatmap layer (weight capped to avoid one tower dominating)
max_weight = rows[0][2]
heat_data = [[r[0], r[1], min(r[2] / max_weight, 1.0)] for r in rows]
HeatMap(heat_data, name="Event heatmap", radius=12, blur=15, min_opacity=0.3).add_to(m)

# Clustered marker layer
cluster = MarkerCluster(name="Cell towers").add_to(m)
for lat, lon, cnt in rows:
    folium.CircleMarker(
        location=[lat, lon],
        radius=4,
        color="#2563eb",
        fill=True,
        fill_opacity=0.7,
        tooltip=f"Events: {cnt:,}",
    ).add_to(cluster)

folium.LayerControl().add_to(m)

out_map = f"{OUT_DIR}/cell_tower_map.html"
m.save(out_map)
print(f"  Saved: {out_map}  ({len(rows):,} towers plotted)", flush=True)

# ---------------------------------------------------------------------------
# Step 4: Load trips, resolve coordinates, find top 20 with 100% coverage
# ---------------------------------------------------------------------------
print("\nStep 4: Loading trips and resolving coordinates...", flush=True)

import pyarrow as pa

# Only use the partitioned parquets (event_date=*/); skip the flat top-level file
trip_files = glob.glob(f"{TRIPS_DIR}/event_date=*/*.parquet")
print(f"  Trip parquet files: {len(trip_files):,}", flush=True)

tables = [pq.read_table(f) for f in trip_files]
all_trips = pa.concat_tables(tables, promote=True).to_pandas()
print(f"  Total trips loaded: {len(all_trips):,}", flush=True)

# Sort by n_cells descending; iterate until we have 20 with 100% coord coverage
all_trips.sort_values("n_cells", ascending=False, inplace=True)

top20 = []
checked = 0
for _, row in all_trips.iterrows():
    checked += 1
    seq_str = str(row["cell_sequence"])
    cell_ids = []
    for tok in seq_str.split("->"):
        tok = tok.strip()
        try:
            cell_ids.append(int(tok))
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

print(f"  Candidates checked: {checked:,}  Top-20 found: {len(top20)}", flush=True)
if len(top20) < 20:
    print(f"  WARNING: only found {len(top20)} trips with 100% coordinate coverage", flush=True)

# ---------------------------------------------------------------------------
# Step 5: Save top20_trips_mapped.json
# ---------------------------------------------------------------------------
print("\nStep 5: Saving top20_trips_mapped.json...", flush=True)
out_json = f"{OUT_DIR}/top20_trips_mapped.json"
with open(out_json, "w") as fh:
    json.dump(top20, fh, indent=2)
print(f"  Saved: {out_json}", flush=True)

# ---------------------------------------------------------------------------
# Step 6: Build top20_trips_map.html
# ---------------------------------------------------------------------------
print("\nStep 6: Building top20_trips_map.html...", flush=True)

COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]

# Centre map on mean of all trip coordinates
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
