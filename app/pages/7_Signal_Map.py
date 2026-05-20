"""
Page 7: Signal-Filtered Trip Map
Filter trips by date range, average RSRP, duration, and US state.
Route colour is scaled to the actual data distribution (p10–p90 RSRP).
"""
from datetime import date

import duckdb
import folium
import pandas as pd
import streamlit as st
from branca.element import MacroElement
from jinja2 import Template
from streamlit_folium import st_folium

COORD_CSV  = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
TRIPS_DIR  = "/home/jovyan/data/stage/trips"
MAX_TRIPS  = 200
CELL_DOT_MIN_ZOOM = 10

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


st.set_page_config(page_title="Signal Map", layout="wide")
st.title("📶 Signal-Filtered Trip Map")
st.caption("Filter trips by date range, signal strength, duration, and state. Route colour = signal quality.")

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data bounds...")
def load_bounds():
    con = duckdb.connect()
    return con.execute(f"""
        SELECT
            FLOOR(MIN(avg_neighbor_rsrp))                                                   AS rsrp_min,
            CEIL(MAX(avg_neighbor_rsrp))                                                    AS rsrp_max,
            ROUND(PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY avg_neighbor_rsrp), 1)      AS rsrp_p10,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY avg_neighbor_rsrp), 1)      AS rsrp_p90,
            ROUND(MIN(duration_minutes) / 60.0, 2)                                         AS dur_min_h,
            ROUND(MAX(duration_minutes) / 60.0, 2)                                         AS dur_max_h
        FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
        WHERE avg_neighbor_rsrp IS NOT NULL
          AND duration_minutes  > 0
    """).df().iloc[0]


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


bounds       = load_bounds()
coord_lookup = load_coord_lookup()

rsrp_data_min = int(bounds["rsrp_min"])
rsrp_data_max = int(bounds["rsrp_max"])
# Color scale spans the middle 80% of the actual distribution
_RSRP_LO = float(bounds["rsrp_p10"])
_RSRP_HI = float(bounds["rsrp_p90"])
dur_data_max = min(float(bounds["dur_max_h"]), 48.0)

# Pre-compute per-state cell ID sets (used for state filter)
@st.cache_data(show_spinner=False)
def state_cell_sets(_coord_lookup: dict) -> dict[str, list[int]]:
    result = {}
    for state, (lat_min, lat_max, lon_min, lon_max) in US_STATES.items():
        result[state] = [
            cid for cid, (lat, lon) in _coord_lookup.items()
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
        ]
    return result

_state_cells = state_cell_sets(coord_lookup)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

start_date = st.sidebar.date_input("Start date", value=date(2025, 1, 1))
end_date   = st.sidebar.date_input("End date",   value=date(2025, 12, 31))

st.sidebar.divider()

rsrp_range = st.sidebar.slider(
    "Avg signal strength (RSRP dBm)",
    min_value=rsrp_data_min,
    max_value=rsrp_data_max,
    value=(rsrp_data_min, rsrp_data_max),
    step=1,
    help="Less negative = stronger signal. The scale is coloured to your actual data distribution.",
)

dur_range = st.sidebar.slider(
    "Trip duration (hours)",
    min_value=0.0,
    max_value=dur_data_max,
    value=(0.5, min(8.0, dur_data_max)),
    step=0.5,
)

st.sidebar.divider()

selected_states = st.sidebar.multiselect(
    "Filter by state (start or end cell)",
    options=sorted(US_STATES.keys()),
    default=[],
    placeholder="All states",
)

top_n = st.sidebar.slider("Max trips on map", 10, MAX_TRIPS, 50)

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Querying trips...", ttl=300)
def query_trips(start_date, end_date, rsrp_lo, rsrp_hi, dur_lo_h, dur_hi_h,
                selected_states: tuple, top_n):
    con = duckdb.connect()

    state_filter = ""
    if selected_states:
        cell_ids = []
        for s in selected_states:
            cell_ids.extend(_state_cells.get(s, []))
        if cell_ids:
            con.register("state_cells", pd.DataFrame({"cell_id": cell_ids}))
            state_filter = """
                AND (first_cell IN (SELECT cell_id FROM state_cells)
                  OR last_cell  IN (SELECT cell_id FROM state_cells))
            """

    return con.execute(f"""
        SELECT trip_id, vehicle_id, trip_start, trip_end,
               duration_minutes, n_cells, n_handovers,
               avg_neighbor_rsrp, avg_neighbor_rsrq, avg_ping_ms,
               cell_sequence
        FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
        WHERE event_date BETWEEN '{start_date}' AND '{end_date}'
          AND avg_neighbor_rsrp BETWEEN {rsrp_lo} AND {rsrp_hi}
          AND duration_minutes  BETWEEN {dur_lo_h * 60} AND {dur_hi_h * 60}
          {state_filter}
        ORDER BY duration_minutes DESC
        LIMIT {top_n}
    """).df()


df = query_trips(
    start_date, end_date,
    rsrp_range[0], rsrp_range[1],
    dur_range[0], dur_range[1],
    tuple(sorted(selected_states)),
    top_n,
)

if df.empty:
    st.warning("No trips match the current filters. Try widening the RSRP or duration range.")
    st.stop()

# ---------------------------------------------------------------------------
# Resolve coordinates
# ---------------------------------------------------------------------------
trips = []
for _, row in df.iterrows():
    coords = []
    for tok in str(row["cell_sequence"]).split("->"):
        try:
            cid = int(tok.strip())
        except ValueError:
            continue
        if cid in coord_lookup:
            lat, lon = coord_lookup[cid]
            coords.append({"cell_id": cid, "lat": lat, "lon": lon})
    if coords:
        trips.append({**row.to_dict(), "coordinates": coords})

if not trips:
    st.warning("No coordinate data found for matched trips.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
durations = [t["duration_minutes"] for t in trips]
rsrp_vals = [t["avg_neighbor_rsrp"] for t in trips if pd.notna(t["avg_neighbor_rsrp"])]
ping_vals = [t["avg_ping_ms"]       for t in trips if pd.notna(t["avg_ping_ms"])]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Trips shown",  len(trips))
c2.metric("Avg duration", f"{sum(durations) / len(durations) / 60:.1f} h")
c3.metric("Avg RSRP",     f"{sum(rsrp_vals) / len(rsrp_vals):.1f} dBm" if rsrp_vals else "—")
c4.metric("Best RSRP",    f"{max(rsrp_vals):.1f} dBm"                  if rsrp_vals else "—")
c5.metric("Avg ping",     f"{sum(ping_vals) / len(ping_vals):.0f} ms"  if ping_vals  else "—")

st.divider()

# ---------------------------------------------------------------------------
# Colour scale: red (p10) → green (p90), anchored to actual data distribution
# ---------------------------------------------------------------------------
def rsrp_colour(rsrp: float) -> str:
    t = max(0.0, min(1.0, (rsrp - _RSRP_LO) / (_RSRP_HI - _RSRP_LO)))
    r = int(220 * (1 - t))
    g = int(180 * t)
    return f"#{r:02x}{g:02x}00"

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
all_lats = [c["lat"] for t in trips for c in t["coordinates"]]
all_lons = [c["lon"] for t in trips for c in t["coordinates"]]
centre   = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]

m         = folium.Map(location=centre, zoom_start=6, tiles="CartoDB positron")
dot_group = folium.FeatureGroup(name="Cell towers", show=False)

for trip in trips:
    rsrp      = trip["avg_neighbor_rsrp"]
    colour    = rsrp_colour(float(rsrp)) if pd.notna(rsrp) else "#888888"
    coords_ll = [[c["lat"], c["lon"]] for c in trip["coordinates"]]
    rsrp_str  = f"{rsrp:.1f} dBm" if pd.notna(rsrp) else "N/A"
    label = (
        f'{trip["trip_id"]}<br>'
        f'RSRP: {rsrp_str}<br>'
        f'Duration: {trip["duration_minutes"] / 60:.1f} h &nbsp;|&nbsp; {int(trip["n_cells"])} cells'
    )

    folium.PolyLine(coords_ll, color=colour, weight=3, opacity=0.85, tooltip=label).add_to(m)

    for c in trip["coordinates"][1:-1]:
        folium.CircleMarker(
            location=[c["lat"], c["lon"]], radius=2,
            color="#000000", fill=True, fill_color="#000000", fill_opacity=0.7,
            weight=1, tooltip=f'Cell {c["cell_id"]}',
        ).add_to(dot_group)

    folium.CircleMarker(
        coords_ll[0], radius=5, color=colour, fill=False, weight=2,
        tooltip=f'START — {str(trip["trip_start"])[:16]}',
    ).add_to(m)
    folium.CircleMarker(
        coords_ll[-1], radius=5, color=colour,
        fill=True, fill_color=colour, fill_opacity=1.0, weight=2,
        tooltip=f'END — {str(trip["trip_end"])[:16]}',
    ).add_to(m)

dot_group.add_to(m)
_ZoomLayer(dot_group, CELL_DOT_MIN_ZOOM).add_to(m)

st_folium(m, width=1200, height=600, returned_objects=[])
st.caption(
    f"Route colour: red = weak (≤ {_RSRP_LO:.0f} dBm, p10), green = strong (≥ {_RSRP_HI:.0f} dBm, p90). "
    "Cell dots appear at zoom ≥ 10."
)

st.divider()

# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
st.subheader("Matched trips")
table = pd.DataFrame([{
    "trip_id":        t["trip_id"],
    "vehicle_id":     t["vehicle_id"],
    "start":          str(t["trip_start"])[:16],
    "end":            str(t["trip_end"])[:16],
    "duration (h)":   round(t["duration_minutes"] / 60, 2),
    "n_cells":        int(t["n_cells"]),
    "n_handovers":    int(t["n_handovers"]),
    "Avg RSRP (dBm)": round(t["avg_neighbor_rsrp"], 1) if pd.notna(t["avg_neighbor_rsrp"]) else None,
    "Avg RSRQ (dB)":  round(t["avg_neighbor_rsrq"], 1) if pd.notna(t["avg_neighbor_rsrq"]) else None,
    "Avg ping (ms)":  round(t["avg_ping_ms"],        0) if pd.notna(t["avg_ping_ms"])        else None,
} for t in trips])
st.download_button("⬇ Download JSON", data=table.to_json(orient="records", indent=2), file_name="trips.json", mime="application/json")
st.dataframe(table, use_container_width=True, hide_index=True)
