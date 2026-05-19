"""
Page 1: Route Map
Top-N trips by cells visited, filtered by duration, quality, and date range.
"""
from datetime import date

import duckdb
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

COORD_CSV = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
TRIPS_DIR = "/home/jovyan/data/stage/trips"

COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]

st.set_page_config(page_title="Route Map", layout="wide")
st.title("🗺️ Top Vehicle Trips")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

max_hours = st.sidebar.slider("Max trip duration (hours)", 1.0, 48.0, 8.0, 0.5)
min_cph   = st.sidebar.slider("Min cells per hour", 1.0, 10.0, 3.0, 0.5)
top_n     = st.sidebar.slider("Number of trips", 5, 20, 20)

st.sidebar.divider()

use_dates  = st.sidebar.toggle("Filter by date range", value=False)
start_date = st.sidebar.date_input("Start date", value=date(2025, 7, 1)) if use_dates else None
end_date   = st.sidebar.date_input("End date",   value=date(2025, 7, 31)) if use_dates else None

# ---------------------------------------------------------------------------
# Cached coordinate lookup
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading cell coordinates...")
def load_coord_lookup() -> dict:
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

coord_lookup = load_coord_lookup()

# ---------------------------------------------------------------------------
# Query trips
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Querying trips...", ttl=300)
def query_trips(max_hours, min_cph, start_date, end_date, top_n):
    max_min = max_hours * 60
    date_clauses = []
    if start_date:
        date_clauses.append(f"trip_start >= '{start_date}'")
    if end_date:
        date_clauses.append(f"trip_start < date '{end_date}' + interval 1 day")
    date_filter = ("AND " + " AND ".join(date_clauses)) if date_clauses else ""

    con = duckdb.connect()
    return con.execute(f"""
        SELECT trip_id, trip_start, trip_end, duration_minutes,
               n_handovers, n_cells, cell_sequence
        FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
        WHERE duration_minutes > 0
          AND duration_minutes <= {max_min}
          AND (n_cells / (duration_minutes / 60.0)) >= {min_cph}
        {date_filter}
        ORDER BY n_cells DESC
        LIMIT 500
    """).df()

candidates = query_trips(max_hours, min_cph, start_date, end_date, top_n)

if candidates.empty:
    st.warning("No trips match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Resolve coordinates
# ---------------------------------------------------------------------------
trips = []
for _, row in candidates.iterrows():
    cell_ids = []
    for tok in str(row["cell_sequence"]).split("->"):
        try:
            cell_ids.append(int(tok.strip()))
        except ValueError:
            pass

    coords, ok = [], True
    for cid in cell_ids:
        if cid in coord_lookup:
            coords.append({"cell_id": cid, "lat": coord_lookup[cid][0], "lon": coord_lookup[cid][1]})
        else:
            ok = False
            break

    if ok and coords:
        trips.append({**row.to_dict(), "coordinates": coords})
    if len(trips) == top_n:
        break

if not trips:
    st.warning("No trips with full coordinate coverage for these filters.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Trips shown", len(trips))
col2.metric("Max cells", max(t["n_cells"] for t in trips))
durations = [t["duration_minutes"] for t in trips if t["duration_minutes"]]
col3.metric("Avg duration", f"{sum(durations)/len(durations)/60:.1f}h" if durations else "—")
col4.metric("Coord coverage", "100%")

st.divider()

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
all_lats = [c["lat"] for t in trips for c in t["coordinates"]]
all_lons = [c["lon"] for t in trips for c in t["coordinates"]]
centre   = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]

m = folium.Map(location=centre, zoom_start=6, tiles="CartoDB positron")

for i, trip in enumerate(trips):
    colour    = COLOURS[i % len(COLOURS)]
    coords_ll = [[c["lat"], c["lon"]] for c in trip["coordinates"]]
    label     = f'{trip["trip_id"]}  |  {trip["n_cells"]} cells  |  {trip["duration_minutes"]/60:.1f}h'

    folium.PolyLine(coords_ll, color=colour, weight=3, opacity=0.85, tooltip=label).add_to(m)

    folium.CircleMarker(
        location=coords_ll[0], radius=6, color=colour,
        fill=False, weight=2, tooltip=f"START — {trip['trip_start']}",
    ).add_to(m)

    folium.CircleMarker(
        location=coords_ll[-1], radius=6, color=colour,
        fill=True, fill_color=colour, fill_opacity=1.0, weight=2,
        tooltip=f"END — {trip['trip_end']}",
    ).add_to(m)

st_folium(m, width=1200, height=600, returned_objects=[])

# ---------------------------------------------------------------------------
# Trips table
# ---------------------------------------------------------------------------
st.subheader("Trip details")
table = pd.DataFrame([{
    "trip_id":           t["trip_id"],
    "start":             str(t["trip_start"])[:16],
    "end":               str(t["trip_end"])[:16],
    "duration (h)":      round(t["duration_minutes"] / 60, 2) if t["duration_minutes"] else None,
    "n_cells":           t["n_cells"],
    "n_handovers":       t["n_handovers"],
    "cells/hour":        round(t["n_cells"] / (t["duration_minutes"] / 60), 1) if t["duration_minutes"] else None,
} for t in trips])
st.dataframe(table, use_container_width=True, hide_index=True)
