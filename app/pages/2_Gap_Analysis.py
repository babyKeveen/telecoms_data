"""
Page 2: Gap Analysis
Two complementary views:
  1. Silence gaps — ping intervals > expected cadence (~485s) → dead zones
  2. Quality gaps — RSRP / SINR below threshold → degraded but connected coverage
"""
import sys
sys.path.insert(0, '/home/jovyan')

from datetime import date

import duckdb
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from pipeline.gaps import RSRP_POOR, SINR_POOR, SIGNAL_BAR_POOR
from pipeline.ingest import PARQUET_PATH

HANDOVER_DIR = "/home/jovyan/data/stage/handover_events"
COORD_CSV    = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
POLL_SECONDS = 485

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
rsrp_thresh   = st.sidebar.slider("RSRP poor threshold (dBm)", -120, -70, RSRP_POOR)
sinr_thresh   = st.sidebar.slider("SINR poor threshold (dB)",   -10,  10,  SINR_POOR)
signal_thresh = st.sidebar.slider("Signal bars threshold",        1,   4,  SIGNAL_BAR_POOR)

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
# Tab 2: Signal quality gaps (uses thresholds from pipeline/gaps.py)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Analysing signal quality...", ttl=300)
def query_quality_gaps(start_date, end_date, rsrp_thresh, sinr_thresh, signal_thresh):
    con = duckdb.connect()
    return con.execute(f"""
        SELECT
            cell_id,
            COUNT(*)                                                              AS total_records,
            SUM(CASE WHEN rsrp  < {rsrp_thresh}
                          OR sinr  < {sinr_thresh}
                          OR signal <= {signal_thresh} THEN 1 ELSE 0 END)        AS gap_records,
            ROUND(100.0 *
                  SUM(CASE WHEN rsrp  < {rsrp_thresh}
                                OR sinr  < {sinr_thresh}
                                OR signal <= {signal_thresh} THEN 1 ELSE 0 END)
                  / COUNT(*), 1)                                                  AS gap_pct,
            ROUND(AVG(rsrp), 1)                                                  AS avg_rsrp,
            ROUND(MIN(rsrp), 1)                                                  AS min_rsrp,
            ROUND(AVG(sinr), 1)                                                  AS avg_sinr,
            ROUND(AVG(lat),  5)                                                  AS lat,
            ROUND(AVG("long"), 5)                                                AS lon
        FROM read_parquet('{str(PARQUET_PATH)}')
        WHERE time::date BETWEEN '{start_date}' AND '{end_date}'
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
    quality_df = query_quality_gaps(start_date, end_date, rsrp_thresh, sinr_thresh, signal_thresh)
    quality_err = None
except Exception as e:
    quality_df  = pd.DataFrame()
    quality_err = str(e)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["Silence Gaps", "Signal Quality"])

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
        f"Poor signal = RSRP < {rsrp_thresh} dBm  OR  SINR < {sinr_thresh} dB  "
        f"OR  signal bars ≤ {signal_thresh}  —  thresholds from `pipeline/gaps.py`"
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
        # Coordinates here are averaged vehicle GPS positions during poor-signal readings,
        # not cell tower locations — they show where in the road network quality degrades.
        map_rows_q = quality_df.dropna(subset=["lat", "lon"]).to_dict("records")

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
                        f'Cell {int(r["cell_id"])}<br>'
                        f'Poor-signal: {r["gap_records"]:,} records ({r["gap_pct"]}%)<br>'
                        f'Avg RSRP: {r["avg_rsrp"]} dBm<br>'
                        f'Min RSRP: {r["min_rsrp"]} dBm<br>'
                        f'Avg SINR: {r["avg_sinr"]} dB'
                    ),
                ).add_to(mq)

            st_folium(mq, width=1200, height=550, returned_objects=[])
        else:
            st.info("No cells could be mapped — lat/lon missing from raw parquet.")

        st.divider()

        st.subheader("Worst signal quality cells")
        st.dataframe(
            quality_df.head(50)
            .drop(columns=["lat", "lon"], errors="ignore")
            .rename(columns={
                "cell_id":       "Cell ID",
                "total_records": "Total records",
                "gap_records":   "Poor-signal records",
                "gap_pct":       "Gap %",
                "avg_rsrp":      "Avg RSRP (dBm)",
                "min_rsrp":      "Min RSRP (dBm)",
                "avg_sinr":      "Avg SINR (dB)",
            }),
            use_container_width=True,
            hide_index=True,
        )
