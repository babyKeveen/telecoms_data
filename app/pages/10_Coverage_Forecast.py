"""
Page 10: Coverage Gap Forecast
Predicts P(poor signal) for cells in a selected state and time context.
Requires: /home/jovyan/data/models/coverage_gap_model.pkl
"""
import sys
from pathlib import Path

import folium
import pandas as pd
import shapely
import streamlit as st
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
from streamlit_folium import st_folium

sys.path.insert(0, "/home/jovyan/telco-poc")
from pipeline.coverage_model import load_model, score_cells

COORD_CSV = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"

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

DOW_LABELS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@st.cache_resource(show_spinner="Loading state boundaries...")
def build_state_polygons():
    polys = {name: box(lon_min, lat_min, lon_max, lat_max)
             for name, (lat_min, lat_max, lon_min, lon_max) in US_STATES.items()}
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


@st.cache_data(show_spinner="Filtering cells for state...", ttl=3600)
def cells_in_state(state_name: str, _coord_lookup: dict) -> dict:
    lat_min, lat_max, lon_min, lon_max = US_STATES[state_name]
    return {
        cid: (lat, lon)
        for cid, (lat, lon) in _coord_lookup.items()
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
    }


@st.cache_data(show_spinner="Computing gap predictions...", ttl=300)
def compute_scores(state: str, hour: int, dow: int) -> pd.DataFrame:
    coord_lookup = load_coord_lookup()
    cell_coords  = cells_in_state(state, coord_lookup)
    return score_cells(cell_coords, hour=hour, dow=dow)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Coverage Forecast", layout="wide")
st.title("📡 Coverage Gap Forecast")
st.caption("ML-predicted probability of poor neighbour RSRP (< −90 dBm) by cell, hour, and day of week. "
           "Most cells in this dataset have consistently weak RSRP — use the threshold slider to isolate the worst.")

# Check model is available
try:
    load_model()
except FileNotFoundError:
    st.error(
        "Model not found. Run `python notebooks/build_coverage_gap_model.py` "
        "inside the container first."
    )
    st.stop()

# Sidebar
st.sidebar.header("Filters")

state = st.sidebar.selectbox("State", sorted(US_STATES.keys()), index=sorted(US_STATES.keys()).index("Michigan"))
hour  = st.sidebar.slider("Hour of day", 0, 23, 9, format="%d:00")
dow   = st.sidebar.selectbox("Day of week", range(7), format_func=lambda d: DOW_LABELS[d], index=1)
threshold = st.sidebar.slider("Show cells with P(gap) ≥", 0.50, 0.99, 0.85, step=0.01,
                               format="%.2f")

# Compute
with st.spinner(f"Scoring cells in {state} at {hour:02d}:00 {DOW_LABELS[dow]}..."):
    scores = compute_scores(state, hour, dow)

above = scores[scores["p_gap"] >= threshold]
st.subheader(f"{state} — {hour:02d}:00 {DOW_LABELS[dow]}")

c1, c2, c3 = st.columns(3)
c1.metric("Cells scored", f"{len(scores):,}")
c2.metric(f"Cells with P(gap) ≥ {threshold:.0%}", f"{len(above):,}")
c3.metric("Avg P(gap) across state", f"{scores['p_gap'].mean():.1%}" if len(scores) else "—")

st.divider()

# Map
if scores.empty:
    st.warning("No cells found for this state.")
else:
    lat_min, lat_max, lon_min, lon_max = US_STATES[state]
    centre = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]
    m = folium.Map(location=centre, zoom_start=7, tiles="CartoDB positron")

    def pgap_colour(p):
        r = int(220 * p)
        g = int(180 * (1 - p))
        return f"#{r:02x}{g:02x}00"

    # Plot cells above threshold
    plot_df = above if len(above) <= 5000 else above.head(5000)
    for _, r in plot_df.iterrows():
        col = pgap_colour(float(r["p_gap"]))
        tip = f"Cell {int(r['cell_id'])}<br>P(gap): {r['p_gap']:.1%}"
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=5,
            color=col, fill=True, fill_color=col, fill_opacity=0.8,
            tooltip=tip,
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=1200, height=560, returned_objects=[])

    if len(above) > 5000:
        st.caption(f"Showing top 5,000 highest-risk cells of {len(above):,} above threshold.")

st.divider()

# Top-risk table
st.subheader(f"Highest-risk cells (P(gap) ≥ {threshold:.0%})")
if above.empty:
    st.info("No cells above threshold. Try lowering the P(gap) slider.")
else:
    display = above.head(200).copy()
    display["cell_id"]  = display["cell_id"].astype(int)
    display["lat"]      = display["lat"].round(4)
    display["lon"]      = display["lon"].round(4)
    display["P(gap)"]   = (display["p_gap"] * 100).round(1).astype(str) + "%"
    st.dataframe(display[["cell_id", "lat", "lon", "P(gap)"]],
                 use_container_width=True, hide_index=True)
