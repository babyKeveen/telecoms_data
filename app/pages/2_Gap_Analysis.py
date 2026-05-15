"""
Page 2: Gap Analysis
Coverage gaps detected from ping-interval anomalies in the real handover data.
A gap is a ping interval significantly above the expected 485s poll cadence.
"""
from datetime import date, timedelta

import duckdb
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

HANDOVER_DIR = "/home/jovyan/data/stage/handover_events"
COORD_CSV    = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
POLL_SECONDS = 485   # observed median ping cadence

st.set_page_config(page_title="Gap Analysis", layout="wide")
st.title("⚠️ Coverage Gap Analysis")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

default_end   = date(2025, 7, 31)
default_start = date(2025, 7, 1)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date   = st.sidebar.date_input("End date",   value=default_end)

min_gap_min = st.sidebar.slider("Min gap duration (minutes)", 5, 60, 15)
max_gap_min = st.sidebar.slider("Max gap duration (minutes — cap overnight parking)", 30, 240, 120)

min_gap_sec = min_gap_min * 60
max_gap_sec = max_gap_min * 60

st.sidebar.divider()
st.sidebar.caption(
    f"Expected poll cadence: ~{POLL_SECONDS}s. "
    f"Gaps >{min_gap_min}m flag likely signal loss."
)

# ---------------------------------------------------------------------------
# Coordinate lookup (shared cache with Route Map page)
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
# Query 1: coverage gaps
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Detecting coverage gaps...", ttl=300)
def query_gaps(start_date, end_date, min_gap_sec, max_gap_sec):
    con = duckdb.connect()
    return con.execute(f"""
        WITH ordered AS (
            SELECT vehicle_id, event_ts, cell_id,
                   LAG(event_ts) OVER (PARTITION BY vehicle_id ORDER BY event_ts) AS prev_ts,
                   LAG(cell_id)  OVER (PARTITION BY vehicle_id ORDER BY event_ts) AS prev_cell
            FROM read_parquet('{HANDOVER_DIR}/**/*.parquet', hive_partitioning=true)
            WHERE event_date BETWEEN '{start_date}' AND '{end_date}'
        )
        SELECT
            vehicle_id,
            prev_cell                                        AS gap_cell,
            DATEDIFF('second', prev_ts, event_ts)           AS gap_seconds,
            event_ts                                         AS resumed_at
        FROM ordered
        WHERE prev_ts IS NOT NULL
          AND DATEDIFF('second', prev_ts, event_ts) BETWEEN {min_gap_sec} AND {max_gap_sec}
        ORDER BY gap_seconds DESC
    """).df()

@st.cache_data(show_spinner="Aggregating gap hotspots...", ttl=300)
def query_gap_hotspots(start_date, end_date, min_gap_sec, max_gap_sec):
    con = duckdb.connect()
    return con.execute(f"""
        WITH ordered AS (
            SELECT vehicle_id, event_ts, cell_id,
                   LAG(event_ts) OVER (PARTITION BY vehicle_id ORDER BY event_ts) AS prev_ts,
                   LAG(cell_id)  OVER (PARTITION BY vehicle_id ORDER BY event_ts) AS prev_cell
            FROM read_parquet('{HANDOVER_DIR}/**/*.parquet', hive_partitioning=true)
            WHERE event_date BETWEEN '{start_date}' AND '{end_date}'
        )
        SELECT
            prev_cell                                                AS cell_id,
            COUNT(*)                                                 AS gap_events,
            COUNT(DISTINCT vehicle_id)                               AS vehicles_affected,
            ROUND(AVG(DATEDIFF('second', prev_ts, event_ts)) / 60, 1) AS avg_gap_min,
            ROUND(MAX(DATEDIFF('second', prev_ts, event_ts)) / 60, 1) AS max_gap_min
        FROM ordered
        WHERE prev_ts IS NOT NULL
          AND DATEDIFF('second', prev_ts, event_ts) BETWEEN {min_gap_sec} AND {max_gap_sec}
        GROUP BY prev_cell
        ORDER BY gap_events DESC
        LIMIT 500
    """).df()

gaps_df     = query_gaps(start_date, end_date, min_gap_sec, max_gap_sec)
hotspots_df = query_gap_hotspots(start_date, end_date, min_gap_sec, max_gap_sec)

if gaps_df.empty:
    st.warning("No gaps found for these filters.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total gap events",       f"{len(gaps_df):,}")
col2.metric("Vehicles affected",      f"{gaps_df['vehicle_id'].nunique():,}")
col3.metric("Unique gap cells",       f"{gaps_df['gap_cell'].nunique():,}")
col4.metric("Avg gap duration",       f"{gaps_df['gap_seconds'].mean()/60:.1f} min")

st.divider()

# ---------------------------------------------------------------------------
# Gap duration histogram
# ---------------------------------------------------------------------------
st.subheader("Gap duration distribution")
hist_df = gaps_df.copy()
hist_df["gap_minutes"] = (hist_df["gap_seconds"] / 60).round(1)
st.bar_chart(
    hist_df["gap_minutes"]
    .value_counts()
    .reindex(range(min_gap_min, max_gap_min + 1), fill_value=0)
    .sort_index(),
    height=200,
)

st.divider()

# ---------------------------------------------------------------------------
# Map — gap hotspot cells
# ---------------------------------------------------------------------------
st.subheader("Coverage gap hotspots")

map_rows = []
for _, row in hotspots_df.iterrows():
    cid = int(row["cell_id"]) if pd.notna(row["cell_id"]) else None
    if cid and cid in coord_lookup:
        lat, lon = coord_lookup[cid]
        map_rows.append({
            "cell_id":          cid,
            "lat":              lat,
            "lon":              lon,
            "gap_events":       int(row["gap_events"]),
            "vehicles":         int(row["vehicles_affected"]),
            "avg_gap_min":      row["avg_gap_min"],
        })

if map_rows:
    all_lats = [r["lat"] for r in map_rows]
    all_lons = [r["lon"] for r in map_rows]
    centre = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]

    m = folium.Map(location=centre, zoom_start=5, tiles="CartoDB positron")

    max_events = max(r["gap_events"] for r in map_rows)
    for r in map_rows:
        radius = 4 + 12 * (r["gap_events"] / max_events)
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=radius,
            color="#D0021B",
            fill=True,
            fill_opacity=0.6,
            tooltip=(
                f'Cell {r["cell_id"]}<br>'
                f'Gap events: {r["gap_events"]:,}<br>'
                f'Vehicles: {r["vehicles"]}<br>'
                f'Avg gap: {r["avg_gap_min"]} min'
            ),
        ).add_to(m)

    st_folium(m, width=1200, height=550, returned_objects=[])
    st.caption(f"Showing {len(map_rows):,} cells with coordinate coverage out of {len(hotspots_df):,} gap cells.")
else:
    st.info("No gap cells could be mapped — coordinate lookup returned no matches.")

st.divider()

# ---------------------------------------------------------------------------
# Worst cells table
# ---------------------------------------------------------------------------
st.subheader("Worst coverage cells")
st.dataframe(
    hotspots_df.head(50).rename(columns={
        "cell_id":          "Cell ID",
        "gap_events":       "Gap events",
        "vehicles_affected":"Vehicles",
        "avg_gap_min":      "Avg gap (min)",
        "max_gap_min":      "Max gap (min)",
    }),
    use_container_width=True,
    hide_index=True,
)
