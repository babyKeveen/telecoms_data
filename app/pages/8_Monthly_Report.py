"""
Page 8: Monthly State Report
All trips for a selected month and US state(s). Full KPI table, JSON export, and map.
"""
import json
from datetime import date

import duckdb
import folium
import numpy as np
import pandas as pd
import shapely
import streamlit as st
from folium.plugins import HeatMap
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
from streamlit_folium import st_folium

COORD_CSV    = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
TRIPS_DIR    = "/home/jovyan/data/stage/trips"
HANDOVER_DIR = "/home/jovyan/data/stage/handover_events"
ROUTE_SAMPLE = 3000

PING_BUCKETS = [
    ("ping ≤100",    None, 100),
    ("ping 101-150",  101, 150),
    ("ping 151-200",  151, 200),
    ("ping 201-250",  201, 250),
    ("ping 251-300",  251, 300),
    ("ping 301-350",  301, 350),
    ("ping 351-400",  351, 400),
    ("ping 401-450",  401, 450),
    ("ping 451-500",  451, 500),
    ("ping >500",     501, None),
]

# Bounding boxes (lat_min, lat_max, lon_min, lon_max) — used for SQL pre-filter
US_STATES = {
    "Alabama":        (30.19, 35.01, -88.47, -84.89),
    "Arizona":        (31.33, 37.00, -114.82, -109.04),
    "Arkansas":       (33.00, 36.50, -94.62, -89.64),
    "California":     (32.53, 42.01, -124.41, -114.13),
    "Colorado":       (36.99, 41.00, -109.06, -102.04),
    "Connecticut":    (40.98, 42.05, -73.73, -71.79),
    "Delaware":       (38.45, 39.84, -75.79, -74.98),
    "Florida":        (24.52, 31.00, -87.63, -80.03),
    "Georgia":        (30.36, 35.00, -85.61, -80.84),
    "Idaho":          (41.99, 49.00, -117.24, -111.04),
    "Illinois":       (36.97, 42.51, -91.51, -87.01),
    "Indiana":        (37.77, 41.77, -88.10, -84.78),
    "Iowa":           (40.38, 43.50, -96.64, -90.14),
    "Kansas":         (36.99, 40.00, -102.05, -94.59),
    "Kentucky":       (36.50, 39.15, -89.57, -81.96),
    "Louisiana":      (28.92, 33.02, -94.04, -88.82),
    "Maine":          (43.06, 47.46, -71.08, -66.95),
    "Maryland":       (37.91, 39.72, -79.49, -75.05),
    "Massachusetts":  (41.24, 42.89, -73.51, -69.93),
    "Michigan":       (41.70, 48.31, -90.42, -82.41),
    "Minnesota":      (43.50, 49.38, -97.24, -89.48),
    "Mississippi":    (30.17, 35.00, -91.65, -88.10),
    "Missouri":       (35.99, 40.61, -95.77, -89.10),
    "Montana":        (44.36, 49.00, -116.05, -104.04),
    "Nebraska":       (40.00, 43.00, -104.05, -95.31),
    "Nevada":         (35.00, 42.00, -120.00, -114.04),
    "New Hampshire":  (42.70, 45.31, -72.56, -70.62),
    "New Jersey":     (38.93, 41.36, -75.56, -73.89),
    "New Mexico":     (31.33, 37.00, -109.05, -103.00),
    "New York":       (40.50, 45.01, -79.76, -71.85),
    "North Carolina": (33.84, 36.59, -84.32, -75.46),
    "North Dakota":   (45.94, 49.00, -104.05, -96.55),
    "Ohio":           (38.40, 42.33, -84.82, -80.52),
    "Oklahoma":       (33.62, 37.00, -103.00, -94.43),
    "Oregon":         (41.99, 46.24, -124.57, -116.46),
    "Pennsylvania":   (39.72, 42.27, -80.52, -74.69),
    "Rhode Island":   (41.15, 42.01, -71.91, -71.12),
    "South Carolina": (32.04, 35.21, -83.35, -78.54),
    "South Dakota":   (42.48, 45.94, -104.06, -96.44),
    "Tennessee":      (34.98, 36.68, -90.31, -81.65),
    "Texas":          (25.84, 36.50, -106.65, -93.51),
    "Utah":           (37.00, 42.00, -114.05, -109.04),
    "Vermont":        (42.73, 45.02, -73.44, -71.50),
    "Virginia":       (36.54, 39.46, -83.68, -75.25),
    "Washington":     (45.54, 49.00, -124.73, -116.92),
    "West Virginia":  (37.20, 40.64, -82.65, -77.72),
    "Wisconsin":      (42.49, 47.31, -92.89, -86.25),
    "Wyoming":        (40.99, 45.01, -111.06, -104.05),
}

# Accurate shapely polygons — bounding-box fallback for most states,
# proper outlines for complex shapes
@st.cache_resource(show_spinner="Loading state boundaries...")
def build_state_polygons() -> dict:
    polys = {name: box(lon_min, lat_min, lon_max, lat_max)
             for name, (lat_min, lat_max, lon_min, lon_max) in US_STATES.items()}

    # Michigan: Lower + Upper Peninsula
    lower = Polygon([
        (-86.82,41.76),(-86.00,41.70),(-84.81,41.70),(-83.46,41.70),
        (-82.69,42.00),(-82.64,42.61),(-82.41,43.01),(-82.41,43.50),
        (-82.46,43.84),(-82.64,44.00),(-83.16,43.61),(-83.80,43.61),
        (-83.26,44.20),(-83.35,44.50),(-83.57,45.10),(-84.72,45.51),
        (-85.52,45.76),(-86.34,45.90),(-87.06,45.87),(-87.11,45.00),
        (-87.24,44.00),(-87.00,43.00),(-87.36,42.00),(-87.20,41.76),
        (-86.82,41.76)])
    upper = Polygon([
        (-87.06,45.87),(-84.77,45.87),(-83.44,46.00),(-83.44,46.50),
        (-84.00,47.00),(-85.00,47.46),(-88.15,47.46),(-89.00,47.46),
        (-90.42,46.60),(-90.14,46.00),(-87.06,45.87)])
    polys["Michigan"] = MultiPolygon([lower, upper])

    return polys


st.set_page_config(page_title="Monthly Report", layout="wide")
st.title("📋 Monthly State Report")
st.caption("All trips in a selected month and state — full KPI table, JSON export, and map.")

# ---------------------------------------------------------------------------
# Cached loaders
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


state_polygons = build_state_polygons()
coord_lookup   = load_coord_lookup()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

year  = st.sidebar.selectbox("Year",  [2025], index=0)
month = st.sidebar.selectbox("Month", list(range(1, 13)),
                              format_func=lambda m: date(2025, m, 1).strftime("%B"),
                              index=9)   # default October

selected_states = st.sidebar.multiselect(
    "State (start cell)",
    options=sorted(US_STATES.keys()),
    default=["Michigan"],
)

if not selected_states:
    st.info("Select at least one state to generate the report.")
    st.stop()

# Date range for this month
import calendar
_, last_day = calendar.monthrange(year, month)
start_date = date(year, month, 1)
end_date   = date(year, month, last_day)
month_label = start_date.strftime("%B %Y")

# ---------------------------------------------------------------------------
# Query — SQL bounding-box pre-filter across all selected states
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Querying trips...", ttl=300)
def query_trips(start_date, end_date, selected_states: tuple):
    bbox_clauses = []
    for s in selected_states:
        lat_min, lat_max, lon_min, lon_max = US_STATES[s]
        # We filter on first_cell via coord lookup in Python;
        # here we just pull all trips for the month (partition filter is enough
        # to keep the scan fast)
    con = duckdb.connect()
    return con.execute(f"""
        SELECT trip_id, vehicle_id, trip_start, trip_end,
               duration_minutes, n_cells, n_handovers, n_events,
               first_cell, last_cell, dominant_rat,
               avg_neighbor_rsrp, min_neighbor_rsrp,
               avg_neighbor_rsrq, avg_ping_ms
        FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
        WHERE event_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY trip_start
    """).df()


@st.cache_data(show_spinner="Computing ping distributions...", ttl=300)
def fetch_ping_buckets(start_date, end_date, trip_windows: tuple) -> pd.DataFrame:
    """Return a DataFrame indexed by trip_id with one count column per ping bucket."""
    con = duckdb.connect()
    con.register("trip_windows", pd.DataFrame(
        trip_windows, columns=["trip_id", "vehicle_id", "trip_start", "trip_end"]
    ))
    cases = "\n            ".join(
        f"COUNT(CASE WHEN ms > {lo} AND ms <= {hi} THEN 1 END) AS \"{label}\","
        if lo and hi else
        (f"COUNT(CASE WHEN ms <= {hi} THEN 1 END) AS \"{label}\"," if hi
         else f"COUNT(CASE WHEN ms > {lo} THEN 1 END) AS \"{label}\"")
        for label, lo, hi in PING_BUCKETS
    )
    return con.execute(f"""
        WITH raw AS (
            SELECT vehicle_id, event_ts,
                   (ping1 + ping2 + ping3 + ping4) / 4.0 AS ms
            FROM read_parquet('{HANDOVER_DIR}/event_date=*/*.parquet', hive_partitioning=true)
            WHERE event_date BETWEEN '{start_date}' AND '{end_date}'
              AND vehicle_id IN (SELECT DISTINCT vehicle_id FROM trip_windows)
        ),
        joined AS (
            SELECT t.trip_id, r.ms
            FROM trip_windows t
            JOIN raw r
                ON  r.vehicle_id = t.vehicle_id
                AND r.event_ts  >= CAST(t.trip_start AS TIMESTAMP)
                AND r.event_ts  <= CAST(t.trip_end AS TIMESTAMP)
            WHERE r.ms IS NOT NULL
        )
        SELECT trip_id,
            {cases}
        FROM joined
        GROUP BY trip_id
    """).df().set_index("trip_id")


with st.spinner(f"Loading {month_label} trips..."):
    df = query_trips(start_date, end_date, tuple(sorted(selected_states)))

# Resolve start coordinates
df["start_lat"] = df["first_cell"].map(lambda c: coord_lookup.get(int(c), (None, None))[0] if pd.notna(c) else None)
df["start_lon"] = df["first_cell"].map(lambda c: coord_lookup.get(int(c), (None, None))[1] if pd.notna(c) else None)
df["end_lat"]   = df["last_cell"].map(lambda c:  coord_lookup.get(int(c), (None, None))[0] if pd.notna(c) else None)
df["end_lon"]   = df["last_cell"].map(lambda c:  coord_lookup.get(int(c), (None, None))[1] if pd.notna(c) else None)

# Accurate polygon filter using vectorised shapely
filter_geom = unary_union([state_polygons[s] for s in selected_states])
has_start = df.dropna(subset=["start_lat", "start_lon"])
if len(has_start):
    pts  = shapely.points(has_start["start_lon"].values, has_start["start_lat"].values)
    mask = shapely.contains(filter_geom, pts)
    keep_ids = has_start.index[mask]
    df = df.loc[keep_ids].copy()

if df.empty:
    st.warning(f"No trips found starting in {', '.join(selected_states)} during {month_label}.")
    st.stop()

# Ping bucket distributions per trip
with st.spinner("Computing ping distributions..."):
    _windows = tuple(
        (r["trip_id"], r["vehicle_id"], str(r["trip_start"]), str(r["trip_end"]))
        for _, r in df.iterrows()
    )
    ping_dist = fetch_ping_buckets(start_date, end_date, _windows)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
state_label = ", ".join(selected_states)
st.subheader(f"{month_label} — {state_label}")

durations = df["duration_minutes"].dropna()
rsrp_vals = df["avg_neighbor_rsrp"].dropna()
ping_vals = df["avg_ping_ms"].dropna()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total trips",     f"{len(df):,}")
c2.metric("Vehicles",        f"{df['vehicle_id'].nunique():,}")
c3.metric("Avg duration",    f"{durations.mean()/60:.1f} h"         if len(durations) else "—")
c4.metric("Avg RSRP",        f"{rsrp_vals.mean():.1f} dBm"          if len(rsrp_vals) else "—")
c5.metric("Avg ping",        f"{ping_vals.mean():.0f} ms"            if len(ping_vals)  else "—")
c6.metric("Avg handovers",   f"{df['n_handovers'].mean():.1f}"       if len(df)         else "—")

st.divider()

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
has_coords = df.dropna(subset=["start_lat", "start_lon", "end_lat", "end_lon"])

if has_coords.empty:
    st.warning("No coordinate data for these trips.")
else:
    centre_lat = has_coords["start_lat"].mean()
    centre_lon = has_coords["start_lon"].mean()

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=7, tiles="CartoDB positron")

    HeatMap(has_coords[["start_lat", "start_lon"]].values.tolist(),
            name="Trip starts (heatmap)", radius=7, blur=9, min_opacity=0.3).add_to(m)
    HeatMap(has_coords[["end_lat", "end_lon"]].values.tolist(),
            name="Trip ends (heatmap)", radius=7, blur=9, min_opacity=0.3,
            gradient={0.4: "blue", 0.65: "lime", 1: "red"}).add_to(m)

    _LO, _HI = -110.0, -92.0
    def rsrp_colour(v):
        t = max(0.0, min(1.0, (v - _LO) / (_HI - _LO)))
        return "#%02x%02x00" % (int(220 * (1 - t)), int(180 * t))

    route_fg   = folium.FeatureGroup(name=f"Routes (sample of {ROUTE_SAMPLE:,})", show=True)
    pool       = has_coords.dropna(subset=["avg_neighbor_rsrp"])
    n_sample   = min(ROUTE_SAMPLE, len(pool))
    sample     = pool.sample(n=n_sample, random_state=42)
    for _, r in sample.iterrows():
        col = rsrp_colour(float(r["avg_neighbor_rsrp"]))
        ping_str = f" | Ping: {r['avg_ping_ms']:.0f} ms" if pd.notna(r["avg_ping_ms"]) else ""
        tip = (f"{r['trip_id']}<br>"
               f"{str(r['trip_start'])[:16]} → {str(r['trip_end'])[11:16]}<br>"
               f"{r['duration_minutes']:.0f} min | {int(r['n_cells'])} cells | {int(r['n_handovers'])} HOs<br>"
               f"RSRP: {r['avg_neighbor_rsrp']:.1f} dBm{ping_str}")
        folium.PolyLine([[r["start_lat"], r["start_lon"]], [r["end_lat"], r["end_lon"]]],
                        color=col, weight=2, opacity=0.75, tooltip=tip).add_to(route_fg)
    route_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=1200, height=580, returned_objects=[])
    if n_sample < len(has_coords):
        st.caption(
            f"Map shows {n_sample:,} sampled routes of {len(has_coords):,} total. "
            "Heatmap layers cover all trips. Route colour: red = weak RSRP, green = strong."
        )

st.divider()

# ---------------------------------------------------------------------------
# Table + JSON download
# ---------------------------------------------------------------------------
st.subheader(f"All trips ({len(df):,} rows)")

table = pd.DataFrame([{
    "trip_id":           r["trip_id"],
    "vehicle_id":        r["vehicle_id"],
    "trip_start":        str(r["trip_start"])[:19],
    "trip_end":          str(r["trip_end"])[:19],
    "duration (h)":      round(float(r["duration_minutes"]) / 60, 2) if pd.notna(r["duration_minutes"]) else None,
    "n_cells":           int(r["n_cells"])      if pd.notna(r["n_cells"])      else None,
    "n_handovers":       int(r["n_handovers"])  if pd.notna(r["n_handovers"])  else None,
    "n_events":          int(r["n_events"])     if pd.notna(r["n_events"])     else None,
    "dominant_rat":      r["dominant_rat"],
    "avg_rsrp (dBm)":   round(float(r["avg_neighbor_rsrp"]),  2) if pd.notna(r["avg_neighbor_rsrp"])  else None,
    "min_rsrp (dBm)":   round(float(r["min_neighbor_rsrp"]),  2) if pd.notna(r["min_neighbor_rsrp"])  else None,
    "avg_rsrq (dB)":    round(float(r["avg_neighbor_rsrq"]),  2) if pd.notna(r["avg_neighbor_rsrq"])  else None,
    "avg_ping (ms)":    round(float(r["avg_ping_ms"]),         1) if pd.notna(r["avg_ping_ms"])        else None,
    "start_lat":         round(float(r["start_lat"]), 1) if pd.notna(r["start_lat"]) else None,
    "start_lon":         round(float(r["start_lon"]), 1) if pd.notna(r["start_lon"]) else None,
    "end_lat":           round(float(r["end_lat"]),   1) if pd.notna(r["end_lat"])   else None,
    "end_lon":           round(float(r["end_lon"]),   1) if pd.notna(r["end_lon"])   else None,
} for _, r in df.iterrows()])

if not ping_dist.empty:
    for label, _, _ in PING_BUCKETS:
        table[label] = table["trip_id"].map(ping_dist[label].to_dict() if label in ping_dist.columns else {}).fillna(0).astype(int)

json_bytes = table.to_json(orient="records", indent=2).encode("utf-8")
fname = f"trips_{month_label.replace(' ', '_')}_{'-'.join(s.replace(' ', '_') for s in selected_states)}.json"
st.download_button("⬇ Download JSON", data=json_bytes, file_name=fname, mime="application/json")
st.dataframe(table, use_container_width=True, hide_index=True)
