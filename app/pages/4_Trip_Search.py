"""
Page 4: Trip Similarity Search
Find trips similar to a query defined by numeric feature sliders.
Results are shown as a table and mapped as PolyLines.
"""
import sys
sys.path.insert(0, "/home/jovyan/telco-poc")
sys.path.insert(0, "/home/jovyan")

import duckdb
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from pipeline.vectors_np import load, search

COORD_CSV  = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
TRIPS_DIR  = "/home/jovyan/data/stage/trips"

COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]

st.set_page_config(page_title="Trip Search", layout="wide")
st.title("🔍 Trip Similarity Search")
st.caption("Find trips whose characteristics best match your query. Only the features you enable contribute to the search.")

# ---------------------------------------------------------------------------
# Load vector DB (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading vector database...")
def load_db():
    return load()

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

@st.cache_data(show_spinner="Fetching trip sequences...")
def fetch_cell_sequences(trip_ids: tuple) -> pd.DataFrame:
    ids_sql = ", ".join(f"'{t}'" for t in trip_ids)
    con = duckdb.connect()
    return con.execute(f"""
        SELECT trip_id, cell_sequence,
               avg_neighbor_rsrp, avg_neighbor_rsrq, avg_ping_ms
        FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
        WHERE trip_id IN ({ids_sql})
    """).df().set_index("trip_id")

vectors, meta, norm_stats = load_db()
coord_lookup = load_coord_lookup()

# ---------------------------------------------------------------------------
# Sidebar — query builder
# ---------------------------------------------------------------------------

st.sidebar.header("Query")
k = st.sidebar.slider("Results to return", 1, 20, 5)

st.sidebar.divider()
st.sidebar.subheader("Features")
st.sidebar.caption("Toggle each feature on to include it in the search.")

query: dict = {}
active_features: list[str] = []

def _bounds(feature: str) -> tuple[float, float]:
    s = norm_stats[feature]
    return s["min"], s["max"]

# Duration
if st.sidebar.toggle("Duration (minutes)", value=True):
    mn, mx = _bounds("duration_minutes")
    v = st.sidebar.slider("Duration (min)", float(mn), float(mx), float((mn + mx) / 2), step=1.0)
    query["duration_minutes"] = v
    active_features.append("duration_minutes")

# Cells
if st.sidebar.toggle("Number of cells", value=False):
    mn, mx = _bounds("n_cells")
    v = st.sidebar.slider("Cells visited", int(mn), int(mx), int((mn + mx) / 2))
    query["n_cells"] = float(v)
    active_features.append("n_cells")

# Handovers
if st.sidebar.toggle("Number of handovers", value=False):
    mn, mx = _bounds("n_handovers")
    v = st.sidebar.slider("Handovers", float(mn), float(mx), float((mn + mx) / 2), step=1.0)
    query["n_handovers"] = v
    active_features.append("n_handovers")

# Events
if st.sidebar.toggle("Number of events", value=False):
    mn, mx = _bounds("n_events")
    v = st.sidebar.slider("Events", float(mn), float(mx), float((mn + mx) / 2), step=1.0)
    query["n_events"] = v
    active_features.append("n_events")

# Hour of day
if st.sidebar.toggle("Time of day", value=False):
    v = st.sidebar.slider("Hour of day (0–23)", 0, 23, 8)
    query["hour_of_day"] = float(v)
    active_features.append("hour_of_day")

# Day of week
if st.sidebar.toggle("Day of week", value=False):
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    v = st.sidebar.selectbox("Day", options=list(range(7)), format_func=lambda x: day_labels[x])
    query["day_of_week"] = float(v)
    active_features.append("day_of_week")

# ---------------------------------------------------------------------------
# Run search
# ---------------------------------------------------------------------------

if not active_features:
    st.info("Enable at least one feature in the sidebar to search.")
    st.stop()

results = search(
    vectors, meta, norm_stats,
    query=query,
    k=k,
    mask_features=active_features,
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Trips matched",       len(results))
c2.metric("Avg duration",        f"{results['duration_minutes'].mean():.0f} min")
c3.metric("Avg cells",           f"{results['n_cells'].mean():.1f}")
c4.metric("Avg handovers",       f"{results['n_handovers'].mean():.1f}")

sequences = fetch_cell_sequences(tuple(results["trip_id"].tolist()))
avg_rsrp = sequences["avg_neighbor_rsrp"].mean()
avg_ping = sequences["avg_ping_ms"].mean()
c5.metric("Avg neighbour RSRP", f"{avg_rsrp:.1f} dBm" if pd.notna(avg_rsrp) else "N/A")
c6.metric("Avg ping",           f"{avg_ping:.0f} ms"  if pd.notna(avg_ping)  else "N/A")

st.divider()

# ---------------------------------------------------------------------------
# Map — resolve cell sequences for matched trips
# ---------------------------------------------------------------------------


map_trips = []
for _, row in results.iterrows():
    if row["trip_id"] not in sequences.index:
        continue
    cell_ids = []
    for tok in str(sequences.loc[row["trip_id"], "cell_sequence"]).split("->"):
        try:
            cell_ids.append(int(tok.strip()))
        except ValueError:
            pass
    coords = [coord_lookup[c] for c in cell_ids if c in coord_lookup]
    if coords:
        map_trips.append({"meta": row, "coords": coords})

if map_trips:
    all_lats = [lat for t in map_trips for lat, _ in t["coords"]]
    all_lons = [lon for t in map_trips for _, lon in t["coords"]]
    centre = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]

    m = folium.Map(location=centre, zoom_start=7, tiles="CartoDB positron")

    for i, trip in enumerate(map_trips):
        colour = COLOURS[i % len(COLOURS)]
        coords_ll = [[lat, lon] for lat, lon in trip["coords"]]
        r = trip["meta"]
        label = (
            f'{r["trip_id"]}  |  {r["duration_minutes"]:.0f} min  '
            f'|  {int(r["n_cells"])} cells  |  dist {r["distance"]:.3f}'
        )
        folium.PolyLine(coords_ll, color=colour, weight=3, opacity=0.85, tooltip=label).add_to(m)
        folium.CircleMarker(coords_ll[0],  radius=5, color=colour, fill=False, weight=2,
                            tooltip="START").add_to(m)
        folium.CircleMarker(coords_ll[-1], radius=5, color=colour, fill=True,
                            fill_color=colour, fill_opacity=1.0, weight=2,
                            tooltip="END").add_to(m)

    st_folium(m, width=1200, height=500, returned_objects=[])
else:
    st.warning("No coordinate data available for matched trips.")

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

st.subheader("Matched trips")
kpi_cols = sequences[["avg_neighbor_rsrp", "avg_neighbor_rsrq", "avg_ping_ms"]]
results = results.join(kpi_cols, on="trip_id")

display = results[[
    "trip_id", "vehicle_id", "trip_start", "trip_end",
    "duration_minutes", "n_cells", "n_handovers", "n_events", "distance",
    "avg_neighbor_rsrp", "avg_neighbor_rsrq", "avg_ping_ms",
]].copy()
display["trip_start"] = display["trip_start"].astype(str).str[:16]
display["trip_end"]   = display["trip_end"].astype(str).str[:16]
display["duration_minutes"]   = display["duration_minutes"].round(1)
display["distance"]           = display["distance"].round(4)
display["avg_neighbor_rsrp"]  = display["avg_neighbor_rsrp"].round(1)
display["avg_neighbor_rsrq"]  = display["avg_neighbor_rsrq"].round(1)
display["avg_ping_ms"]        = display["avg_ping_ms"].round(0)
st.dataframe(display.rename(columns={
    "avg_neighbor_rsrp": "Nbr RSRP (dBm)",
    "avg_neighbor_rsrq": "Nbr RSRQ (dB)",
    "avg_ping_ms":       "Avg ping (ms)",
}), use_container_width=True, hide_index=True)
