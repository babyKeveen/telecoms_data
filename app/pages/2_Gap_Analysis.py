"""
Page 2: Gap Analysis
Two complementary views:
  1. Silence gaps — ping intervals > expected cadence (~485s) → dead zones
  2. Quality gaps — RSRP / SINR below threshold → degraded but connected coverage
"""
import sys
sys.path.insert(0, '/home/jovyan')

import math
from datetime import date

import duckdb
import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from branca.element import MacroElement
from jinja2 import Template
from streamlit_folium import st_folium

from pipeline.gaps import RSRP_POOR, SINR_POOR

HANDOVER_DIR = "/home/jovyan/data/stage/handover_events"
TRIPS_DIR    = "/home/jovyan/data/stage/trips"
OOS_DIR      = "/home/jovyan/data/stage/oos_events"
COORD_CSV    = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
POLL_SECONDS = 485
CELL_DOT_MIN_ZOOM = 11


class _ZoomLayer(MacroElement):
    """Show/hide a FeatureGroup based on current map zoom."""
    _template = Template("""
        {% macro script(this, kwargs) %}
        (function(){
            var fg  = {{ this.fg_name }};
            var map = {{ this._parent.get_name() }};
            function _upd() {
                if (map.getZoom() >= {{ this.min_zoom }}) {
                    if (!map.hasLayer(fg)) fg.addTo(map);
                } else {
                    if (map.hasLayer(fg)) map.removeLayer(fg);
                }
            }
            map.on('zoomend', _upd);
            _upd();
        })();
        {% endmacro %}
    """)
    def __init__(self, feature_group, min_zoom):
        super().__init__()
        self.fg_name  = feature_group.get_name()
        self.min_zoom = min_zoom

# Approximate bounding boxes (lat_min, lat_max, lon_min, lon_max) for continental US states
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

st.sidebar.divider()
st.sidebar.subheader("Silence gap")
min_gap_min = st.sidebar.slider("Min gap duration (minutes)", 5, 60, 15)
max_gap_min = st.sidebar.slider("Max gap duration (minutes — cap overnight parking)", 30, 240, 120)

st.sidebar.divider()
st.sidebar.subheader("Signal quality thresholds")
rsrp_thresh = st.sidebar.slider("Neighbour RSRP poor threshold (dBm)", -120, -70, RSRP_POOR)
sinr_thresh = st.sidebar.slider("Neighbour RSRQ poor threshold (dB)",   -20,  -5, SINR_POOR)

st.sidebar.divider()
st.sidebar.subheader("Service outages")
min_oos_occurrences = st.sidebar.slider("Min occurrences per corridor/blip", 1, 50, 3)
min_corridor_km = st.sidebar.slider(
    "Min corridor distance (km)", 0, 20, 2,
    help="Corridors shorter than this are likely adjacent sectors at the same site, "
         "not a real gap between two towers — drawn as blips instead.",
)
min_strength_pct = st.sidebar.slider(
    "Min corridor strength (percentile)", 0, 100, 50,
    help="Strength = (vehicle diversity) × (proximity of the two cells). "
         "Corridors confirmed by many different vehicles at near-neighbour "
         "towers score high; infrequent or far-apart 'phantom' pairs score low. "
         "Higher = fewer, more trustworthy corridors shown.",
)
selected_states = st.sidebar.multiselect(
    "Filter map by state (either endpoint)",
    options=sorted(US_STATES.keys()), default=[], placeholder="All states (national overview)",
    help="At national scale, short corridors render as sub-pixel dots. "
         "Pick one or more states to zoom in and see corridors as road-following lines.",
)

min_gap_sec = min_gap_min * 60
max_gap_sec = max_gap_min * 60

# ---------------------------------------------------------------------------
# Shared: coordinate lookup
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
# Tab 1: Silence gaps
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

# ---------------------------------------------------------------------------
# Tab 2: Signal quality gaps — neighbour RSRP/RSRQ from handover_events
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Analysing signal quality...", ttl=300)
def query_quality_gaps(start_date, end_date, rsrp_thresh, sinr_thresh):
    con = duckdb.connect()
    return con.execute(f"""
        SELECT
            cell_id,
            COUNT(*)                                                            AS total_records,
            SUM(CASE WHEN pci_1_rsrp < {rsrp_thresh}
                          OR pci_1_rsrq < {sinr_thresh} THEN 1 ELSE 0 END)    AS gap_records,
            ROUND(100.0 *
                  SUM(CASE WHEN pci_1_rsrp < {rsrp_thresh}
                                OR pci_1_rsrq < {sinr_thresh} THEN 1 ELSE 0 END)
                  / COUNT(*), 1)                                                AS gap_pct,
            ROUND(AVG(pci_1_rsrp), 1)                                          AS avg_rsrp,
            ROUND(MIN(pci_1_rsrp), 1)                                          AS min_rsrp,
            ROUND(AVG(pci_1_rsrq), 1)                                          AS avg_rsrq
        FROM read_parquet('{HANDOVER_DIR}/**/*.parquet', hive_partitioning=true)
        WHERE event_date BETWEEN '{start_date}' AND '{end_date}'
          AND pci_1_rsrp IS NOT NULL
        GROUP BY cell_id
        HAVING gap_records > 0
        ORDER BY gap_records DESC
        LIMIT 500
    """).df()

# ---------------------------------------------------------------------------
# Load data for both tabs
# ---------------------------------------------------------------------------
gaps_df     = query_gaps(start_date, end_date, min_gap_sec, max_gap_sec)
hotspots_df = query_gap_hotspots(start_date, end_date, min_gap_sec, max_gap_sec)

try:
    quality_df = query_quality_gaps(start_date, end_date, rsrp_thresh, sinr_thresh)
    quality_err = None
except Exception as e:
    quality_df  = pd.DataFrame()
    quality_err = str(e)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Analysing neighbour signal...", ttl=300)
def query_neighbour_signal(start_date, end_date):
    con = duckdb.connect()
    return con.execute(f"""
        SELECT
            first_cell                                   AS cell_id,
            COUNT(*)                                     AS trips,
            ROUND(AVG(avg_neighbor_rsrp), 1)             AS avg_neighbor_rsrp,
            ROUND(MIN(min_neighbor_rsrp), 1)             AS worst_neighbor_rsrp,
            ROUND(AVG(avg_neighbor_rsrq), 1)             AS avg_neighbor_rsrq,
            COUNT(DISTINCT vehicle_id)                   AS vehicles
        FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
        WHERE event_date BETWEEN '{start_date}' AND '{end_date}'
          AND avg_neighbor_rsrp IS NOT NULL
        GROUP BY first_cell
        ORDER BY avg_neighbor_rsrp ASC
        LIMIT 500
    """).df()

neighbour_df = query_neighbour_signal(start_date, end_date)

# ---------------------------------------------------------------------------
# Tab 4: Service outages — BEFORE_OOS/AFTER_OOS paired "not spot" corridors
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Approximate great-circle distance in km between two lat/lon points."""
    dlat = (lat1 - lat2) * 111.0
    dlon = (lon1 - lon2) * 111.0 * math.cos(math.radians(lat1))
    return math.hypot(dlat, dlon)


def bounds_to_zoom(lat_min, lat_max, lon_min, lon_max, width_px, height_px, max_zoom=16, padding=1.3):
    """Compute a folium zoom_start that fits the given lat/lon bounds within
    a width_px x height_px map (plus a little padding). Computed server-side
    so it doesn't depend on Leaflet's fit_bounds(), which can mis-zoom if the
    iframe isn't sized yet when the script runs."""
    if lat_max <= lat_min and lon_max <= lon_min:
        return 12

    def lat_to_merc(lat):
        lat = max(min(lat, 85.05), -85.05)
        s = math.sin(math.radians(lat))
        return math.log((1 + s) / (1 - s)) / 2

    lat_fraction = abs(lat_to_merc(lat_max) - lat_to_merc(lat_min)) / math.pi
    lon_fraction = abs(lon_max - lon_min) / 360.0

    def zoom_for(px, fraction):
        if fraction <= 0:
            return max_zoom
        return math.log2(px / 256 / (fraction * padding))

    zoom = min(zoom_for(height_px, lat_fraction), zoom_for(width_px, lon_fraction))
    return max(3, min(max_zoom, zoom))


@st.cache_data(show_spinner=False, ttl=3600)
def snap_to_road(lat_a, lon_a, lat_b, lon_b):
    """Fetch a driving route between two points via the public OSRM API.
    Returns a list of [lat, lon] points, or None on any failure."""
    url = (f"https://router.project-osrm.org/route/v1/driving/"
           f"{lon_a},{lat_a};{lon_b},{lat_b}?overview=full&geometries=geojson")
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        coords = resp.json()["routes"][0]["geometry"]["coordinates"]
        return [[lat, lon] for lon, lat in coords]
    except Exception:
        return None


@st.cache_data(show_spinner="Pairing BEFORE/AFTER OOS events...", ttl=300)
def query_oos_pairs(start_date, end_date, min_occurrences):
    con = duckdb.connect()

    paired_cte = f"""
        WITH paired AS (
            SELECT
                vehicle_id,
                creation_ts,
                MAX(CASE WHEN test_type='BEFORE_OOS' THEN cell_id END) AS start_cell,
                MAX(CASE WHEN test_type='AFTER_OOS'  THEN cell_id END) AS end_cell,
                MAX(oos_start_time) AS oos_start_time,
                MAX(oos_end_time)   AS oos_end_time
            FROM read_parquet('{OOS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
            WHERE event_date BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY vehicle_id, creation_ts
            HAVING start_cell IS NOT NULL AND end_cell IS NOT NULL
        ),
        with_duration AS (
            SELECT *,
                CASE WHEN EXTRACT(year FROM oos_start_time) = 2025
                      AND EXTRACT(year FROM oos_end_time) = 2025
                      AND DATEDIFF('second', oos_start_time, oos_end_time) BETWEEN 0 AND 172800
                     THEN DATEDIFF('second', oos_start_time, oos_end_time) / 60.0
                END AS duration_min
            FROM paired
        )
    """

    # Corridors: start_cell != end_cell, aggregated as undirected pairs
    corridors = con.execute(paired_cte + f"""
        SELECT
            LEAST(start_cell, end_cell)    AS cell_a,
            GREATEST(start_cell, end_cell) AS cell_b,
            COUNT(*)                       AS n_pairs,
            COUNT(DISTINCT vehicle_id)      AS vehicles,
            ROUND(MEDIAN(duration_min), 1)  AS median_duration_min
        FROM with_duration
        WHERE start_cell != end_cell
        GROUP BY 1, 2
        HAVING COUNT(*) >= {min_occurrences}
        ORDER BY n_pairs DESC
        LIMIT 500
    """).df()

    # Blips: start_cell == end_cell — service dropped and returned at the same tower
    blips = con.execute(paired_cte + f"""
        SELECT
            start_cell                     AS cell_id,
            COUNT(*)                       AS n_blips,
            COUNT(DISTINCT vehicle_id)      AS vehicles,
            ROUND(MEDIAN(duration_min), 1)  AS median_duration_min
        FROM with_duration
        WHERE start_cell = end_cell
        GROUP BY 1
        HAVING COUNT(*) >= {min_occurrences}
        ORDER BY n_blips DESC
        LIMIT 500
    """).df()

    return corridors, blips

corridors_df, blips_df = query_oos_pairs(start_date, end_date, min_oos_occurrences)

tab1, tab2, tab3, tab4 = st.tabs(["Silence Gaps", "Signal Quality", "Neighbour Signal", "Service Outages"])

# ── Tab 1 ────────────────────────────────────────────────────────────────────
with tab1:
    st.sidebar.caption(
        f"Expected poll cadence: ~{POLL_SECONDS}s. "
        f"Gaps >{min_gap_min}m flag likely signal loss."
    )

    if gaps_df.empty:
        st.warning("No gaps found for these filters.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total gap events",  f"{len(gaps_df):,}")
        col2.metric("Vehicles affected", f"{gaps_df['vehicle_id'].nunique():,}")
        col3.metric("Unique gap cells",  f"{gaps_df['gap_cell'].nunique():,}")
        col4.metric("Avg gap duration",  f"{gaps_df['gap_seconds'].mean()/60:.1f} min")

        st.divider()

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

        st.subheader("Coverage gap hotspots")
        map_rows = []
        for _, row in hotspots_df.iterrows():
            cid = int(row["cell_id"]) if pd.notna(row["cell_id"]) else None
            if cid and cid in coord_lookup:
                lat, lon = coord_lookup[cid]
                map_rows.append({
                    "cell_id":     cid,
                    "lat":         lat,
                    "lon":         lon,
                    "gap_events":  int(row["gap_events"]),
                    "vehicles":    int(row["vehicles_affected"]),
                    "avg_gap_min": row["avg_gap_min"],
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
            st.caption(
                f"Showing {len(map_rows):,} cells with coordinate coverage "
                f"out of {len(hotspots_df):,} gap cells."
            )
        else:
            st.info("No gap cells could be mapped — coordinate lookup returned no matches.")

        st.divider()

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

# ── Tab 2 ────────────────────────────────────────────────────────────────────
with tab2:
    st.caption(
        f"Poor signal = neighbour RSRP < {rsrp_thresh} dBm  OR  neighbour RSRQ < {sinr_thresh} dB  "
        f"— sourced from `handover_events` stage Parquet"
    )

    if quality_err:
        st.error(f"Could not load signal quality data: {quality_err}")
    elif quality_df.empty:
        st.warning("No poor-signal records found for this date range and thresholds.")
    else:
        total_records = int(quality_df["total_records"].sum())
        total_gap     = int(quality_df["gap_records"].sum())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total records",       f"{total_records:,}")
        col2.metric("Poor-signal records", f"{total_gap:,}")
        col3.metric("Gap %",               f"{100 * total_gap / total_records:.1f}%")
        col4.metric("Cells affected",      f"{len(quality_df):,}")

        st.divider()

        st.subheader("Poor signal hotspots")
        map_rows_q = []
        for _, row in quality_df.iterrows():
            cid = int(row["cell_id"]) if pd.notna(row["cell_id"]) else None
            if cid and cid in coord_lookup:
                lat, lon = coord_lookup[cid]
                map_rows_q.append({**row.to_dict(), "lat": lat, "lon": lon, "cell_id": cid})

        if map_rows_q:
            centre_q = [
                sum(r["lat"] for r in map_rows_q) / len(map_rows_q),
                sum(r["lon"] for r in map_rows_q) / len(map_rows_q),
            ]
            mq = folium.Map(location=centre_q, zoom_start=5, tiles="CartoDB positron")
            max_gap_q = max(r["gap_records"] for r in map_rows_q)
            for r in map_rows_q:
                radius = 4 + 12 * (r["gap_records"] / max_gap_q)
                folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=radius,
                    color="#FF6B00",
                    fill=True,
                    fill_opacity=0.6,
                    tooltip=(
                        f'Cell {r["cell_id"]}<br>'
                        f'Poor-signal: {r["gap_records"]:,} records ({r["gap_pct"]}%)<br>'
                        f'Avg nbr RSRP: {r["avg_rsrp"]} dBm<br>'
                        f'Min nbr RSRP: {r["min_rsrp"]} dBm<br>'
                        f'Avg nbr RSRQ: {r["avg_rsrq"]} dB'
                    ),
                ).add_to(mq)

            st_folium(mq, width=1200, height=550, returned_objects=[])
        else:
            st.info("No cells could be mapped — coordinate lookup returned no matches.")

        st.divider()

        st.subheader("Worst signal quality cells")
        st.dataframe(
            quality_df.head(50).rename(columns={
                "cell_id":       "Cell ID",
                "total_records": "Total records",
                "gap_records":   "Poor-signal records",
                "gap_pct":       "Gap %",
                "avg_rsrp":      "Avg nbr RSRP (dBm)",
                "min_rsrp":      "Min nbr RSRP (dBm)",
                "avg_rsrq":      "Avg nbr RSRQ (dB)",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ── Tab 3: Neighbour Signal ───────────────────────────────────────────────────
with tab3:
    st.caption(
        "Avg RSRP of the strongest neighbour cell, aggregated per trip origin cell. "
        "Low values indicate weak coverage in the surrounding cell neighbourhood."
    )

    if neighbour_df.empty:
        st.warning("No neighbour signal data found for this date range.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Cells analysed",      f"{len(neighbour_df):,}")
        c2.metric("Avg neighbour RSRP",  f"{neighbour_df['avg_neighbor_rsrp'].mean():.1f} dBm")
        c3.metric("Worst neighbour RSRP",f"{neighbour_df['worst_neighbor_rsrp'].min():.1f} dBm")

        st.divider()

        map_rows_n = []
        for _, row in neighbour_df.iterrows():
            cid = int(row["cell_id"]) if pd.notna(row["cell_id"]) else None
            if cid and cid in coord_lookup:
                lat, lon = coord_lookup[cid]
                map_rows_n.append({**row.to_dict(), "lat": lat, "lon": lon, "cell_id": cid})

        if map_rows_n:
            all_lats = [r["lat"] for r in map_rows_n]
            all_lons = [r["lon"] for r in map_rows_n]
            centre_n = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]
            mn = folium.Map(location=centre_n, zoom_start=5, tiles="CartoDB positron")

            rsrp_min = min(r["avg_neighbor_rsrp"] for r in map_rows_n)
            rsrp_max = max(r["avg_neighbor_rsrp"] for r in map_rows_n)
            rsrp_range = rsrp_max - rsrp_min or 1

            for r in map_rows_n:
                # Colour: red = worst RSRP, green = best
                norm = (r["avg_neighbor_rsrp"] - rsrp_min) / rsrp_range
                red  = int(255 * (1 - norm))
                green= int(180 * norm)
                colour = f"#{red:02x}{green:02x}00"
                folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=5,
                    color=colour,
                    fill=True,
                    fill_opacity=0.7,
                    tooltip=(
                        f'Cell {r["cell_id"]}<br>'
                        f'Avg neighbour RSRP: {r["avg_neighbor_rsrp"]} dBm<br>'
                        f'Worst neighbour RSRP: {r["worst_neighbor_rsrp"]} dBm<br>'
                        f'Avg neighbour RSRQ: {r["avg_neighbor_rsrq"]} dB<br>'
                        f'Trips: {int(r["trips"]):,} | Vehicles: {int(r["vehicles"]):,}'
                    ),
                ).add_to(mn)

            st_folium(mn, width=1200, height=550, returned_objects=[])
            st.caption("Red = weakest neighbour signal, green = strongest.")
        else:
            st.info("No cells could be mapped — coordinate lookup returned no matches.")

        st.divider()

        st.subheader("Weakest neighbour signal cells")
        st.dataframe(
            neighbour_df.head(50).rename(columns={
                "cell_id":             "Cell ID",
                "trips":               "Trips",
                "vehicles":            "Vehicles",
                "avg_neighbor_rsrp":   "Avg neighbour RSRP (dBm)",
                "worst_neighbor_rsrp": "Worst neighbour RSRP (dBm)",
                "avg_neighbor_rsrq":   "Avg neighbour RSRQ (dB)",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ── Tab 4: Service Outages ────────────────────────────────────────────────────
with tab4:
    st.caption(
        "BEFORE_OOS / AFTER_OOS are paired records: BEFORE_OOS captures the cell a "
        "vehicle was attached to right before losing service; AFTER_OOS captures the "
        "cell where connectivity was restored. Pairing them reveals **not spots** — "
        "if start and end cell match, it's a **blip** (brief drop at one tower); if "
        "they differ, it's a **corridor** (the vehicle moved through a coverage gap "
        "between two towers while out of service). Outage duration uses "
        "`oos_start_time`/`oos_end_time` where both have a valid 2025 date and the "
        "gap is under 48h — many raw timestamps are 1980 placeholders."
    )

    if corridors_df.empty and blips_df.empty:
        st.warning("No OOS pairs found for this date range and threshold.")
    else:
        n_corridor_occ = int(corridors_df["n_pairs"].sum()) if not corridors_df.empty else 0
        n_blip_occ     = int(blips_df["n_blips"].sum()) if not blips_df.empty else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Corridor occurrences", f"{n_corridor_occ:,}")
        col2.metric("Distinct corridors",   f"{len(corridors_df):,}")
        col3.metric("Blip occurrences",     f"{n_blip_occ:,}")
        col4.metric("Distinct blip cells",  f"{len(blips_df):,}")

        st.divider()

        st.subheader("Not-spot map")

        def in_any_state(lat, lon):
            if not selected_states:
                return True
            for state in selected_states:
                lat_min, lat_max, lon_min, lon_max = US_STATES[state]
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    return True
            return False

        corridor_rows = []
        colocated_rows = []
        for _, row in corridors_df.iterrows():
            a, b = int(row["cell_a"]), int(row["cell_b"])
            if a in coord_lookup and b in coord_lookup:
                lat_a, lon_a = coord_lookup[a]
                lat_b, lon_b = coord_lookup[b]
                if not (in_any_state(lat_a, lon_a) or in_any_state(lat_b, lon_b)):
                    continue
                dist_km = haversine_km(lat_a, lon_a, lat_b, lon_b)
                entry = {**row.to_dict(), "cell_a": a, "cell_b": b,
                         "lat_a": lat_a, "lon_a": lon_a,
                         "lat_b": lat_b, "lon_b": lon_b,
                         "dist_km": dist_km}
                if dist_km >= min_corridor_km:
                    corridor_rows.append(entry)
                else:
                    colocated_rows.append(entry)

        if corridor_rows:
            max_vehicles = max(r["vehicles"] for r in corridor_rows)
            for r in corridor_rows:
                proximity_norm = max(0.0, 1 - r["dist_km"] / 50.0)
                vehicles_norm = r["vehicles"] / max_vehicles
                r["strength"] = vehicles_norm * proximity_norm

            strength_threshold = np.percentile(
                [r["strength"] for r in corridor_rows], min_strength_pct
            )
            corridor_rows = [r for r in corridor_rows if r["strength"] >= strength_threshold]

            with st.spinner(f"Snapping {len(corridor_rows)} corridors to roads..."):
                for r in corridor_rows:
                    r["route"] = snap_to_road(r["lat_a"], r["lon_a"], r["lat_b"], r["lon_b"])

        if corridor_rows:
            corridor_lats = []
            corridor_lons = []
            for r in corridor_rows:
                pts = r["route"] or [[r["lat_a"], r["lon_a"]], [r["lat_b"], r["lon_b"]]]
                corridor_lats += [p[0] for p in pts]
                corridor_lons += [p[1] for p in pts]

            centre = [sum(corridor_lats) / len(corridor_lats), sum(corridor_lons) / len(corridor_lons)]
            zoom = bounds_to_zoom(min(corridor_lats), max(corridor_lats), min(corridor_lons), max(corridor_lons), 1200, 550)

            m = folium.Map(location=centre, zoom_start=zoom, tiles="CartoDB positron")

            max_pairs = max(r["n_pairs"] for r in corridor_rows)
            for r in corridor_rows:
                weight = 2 + 6 * (r["n_pairs"] / max_pairs)
                dur = f'{r["median_duration_min"]:.0f} min' if pd.notna(r["median_duration_min"]) else "—"
                locations = r["route"] or [[r["lat_a"], r["lon_a"]], [r["lat_b"], r["lon_b"]]]
                folium.PolyLine(
                    locations=locations,
                    color="#FF8C00",
                    weight=weight,
                    opacity=0.6,
                    tooltip=(
                        f'Corridor: cell {r["cell_a"]} ↔ {r["cell_b"]}<br>'
                        f'Occurrences: {int(r["n_pairs"]):,}<br>'
                        f'Vehicles: {int(r["vehicles"]):,}<br>'
                        f'Distance: {r["dist_km"]:.1f} km<br>'
                        f'Strength: {r["strength"]:.2f}<br>'
                        f'Median outage duration: {dur}'
                    ),
                ).add_to(m)

            cell_points = {}
            for r in corridor_rows:
                cell_points[r["cell_a"]] = (r["lat_a"], r["lon_a"])
                cell_points[r["cell_b"]] = (r["lat_b"], r["lon_b"])
            dot_group = folium.FeatureGroup(name="Cell towers", show=False)
            for cell_id, (lat, lon) in cell_points.items():
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=3,
                    color="#000000",
                    fill=True,
                    fill_color="#000000",
                    fill_opacity=1.0,
                    tooltip=f"Cell {cell_id}",
                ).add_to(dot_group)
            dot_group.add_to(m)
            _ZoomLayer(dot_group, CELL_DOT_MIN_ZOOM).add_to(m)

            st_folium(m, width=1200, height=550, returned_objects=[])
            state_note = (
                f" Filtered to {', '.join(selected_states)}; map zoomed to the corridor extent."
                if selected_states else
                " Map zoomed to the corridor extent. Pick a state in the sidebar to focus "
                "on a smaller region if corridors are still hard to distinguish."
            )
            st.caption(
                f"Orange lines = not-spot corridors ({len(corridor_rows):,} of {len(corridors_df):,} "
                f"with both endpoints mapped, ≥{min_corridor_km}km apart, ≥{min_oos_occurrences} "
                f"occurrences, strength ≥ {min_strength_pct}th percentile), snapped to roads via "
                f"OSRM where available; thickness = frequency. "
                f"Black dots = the {len(cell_points):,} cell towers defining these corridors' "
                f"endpoints (visible at zoom ≥ {CELL_DOT_MIN_ZOOM}). "
                f"{len(colocated_rows):,} pairs <{min_corridor_km}km apart (likely adjacent "
                f"sectors at the same site) and all blip cells are excluded from the map — "
                f"see the tables below."
                + state_note
            )
            st.download_button(
                "Download map as HTML",
                data=m.get_root().render(),
                file_name=f"oos_notspots_{start_date}_{end_date}.html",
                mime="text/html",
            )
        elif selected_states:
            st.info(
                f"No corridors fall within {', '.join(selected_states)} "
                "at the current thresholds. Try lowering the strength/occurrence "
                "sliders or selecting different states."
            )
        else:
            st.info("No corridors could be mapped at the current thresholds.")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Top not-spot corridors")
            if corridors_df.empty:
                st.info("No corridors found for this threshold.")
            else:
                st.dataframe(
                    corridors_df.head(50).rename(columns={
                        "cell_a":               "Cell A",
                        "cell_b":               "Cell B",
                        "n_pairs":              "Occurrences",
                        "vehicles":             "Vehicles",
                        "median_duration_min":  "Median outage (min)",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
        with col_b:
            st.subheader("Top blip cells")
            if blips_df.empty:
                st.info("No blip cells found for this threshold.")
            else:
                st.dataframe(
                    blips_df.head(50).rename(columns={
                        "cell_id":              "Cell ID",
                        "n_blips":              "Occurrences",
                        "vehicles":             "Vehicles",
                        "median_duration_min":  "Median outage (min)",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
