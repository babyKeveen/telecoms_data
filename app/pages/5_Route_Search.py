"""
Page 5: Route Similarity Search
Pick an origin and destination city; find the trips whose start/end cells
most closely match that corridor using the numpy vector DB.
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

COORD_CSV = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
TRIPS_DIR = "/home/jovyan/data/stage/trips"

CITIES = {
    "Ann Arbor":    (42.281, -83.748), "Atlanta":      (33.749, -84.388),
    "Baltimore":    (39.290, -76.612), "Boston":       (42.360, -71.059),
    "Charlotte":    (35.227, -80.843), "Chicago":      (41.878, -87.630),
    "Cincinnati":   (39.103, -84.512), "Cleveland":    (41.499, -81.695),
    "Columbus":     (39.961, -82.999), "Dallas":       (32.776, -96.797),
    "Denver":       (39.739,-104.984), "Detroit":      (42.331, -83.046),
    "Houston":      (29.760, -95.370), "Indianapolis": (39.768, -86.158),
    "Kansas City":  (39.099, -94.578), "Los Angeles":  (34.052,-118.244),
    "Louisville":   (38.252, -85.758), "Memphis":      (35.149, -90.048),
    "Milwaukee":    (43.038, -87.906), "Minneapolis":  (44.977, -93.265),
    "Nashville":    (36.162, -86.781), "New York":     (40.713, -74.006),
    "Philadelphia": (39.952, -75.165), "Pittsburgh":   (40.440, -79.996),
    "Portland":     (45.523,-122.676), "San Francisco":(37.774,-122.419),
    "Seattle":      (47.606,-122.332), "St. Louis":    (38.627, -90.197),
    "Washington":   (38.907, -77.037),
}

COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]

st.set_page_config(page_title="Route Search", layout="wide")
st.title("🛣️ Route Similarity Search")
st.caption("Find trips whose start and end locations best match a city-to-city corridor.")

# ---------------------------------------------------------------------------
# Load resources (cached)
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
def fetch_sequences(trip_ids: tuple) -> pd.DataFrame:
    ids_sql = ", ".join(f"'{t}'" for t in trip_ids)
    con = duckdb.connect()
    return con.execute(f"""
        SELECT trip_id, cell_sequence
        FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
        WHERE trip_id IN ({ids_sql})
    """).df().set_index("trip_id")

vectors, meta, stats = load_db()
coord_lookup = load_coord_lookup()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

city_names = sorted(CITIES.keys())

st.sidebar.header("Route")
origin      = st.sidebar.selectbox("Origin",      city_names, index=city_names.index("Detroit"))
destination = st.sidebar.selectbox("Destination", city_names, index=city_names.index("Ann Arbor"))
k           = st.sidebar.slider("Results", 1, 20, 10)
both_dirs   = st.sidebar.toggle("Include reverse direction", value=False)

if origin == destination:
    st.warning("Origin and destination must be different.")
    st.stop()

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

orig_lat, orig_lon = CITIES[origin]
dest_lat, dest_lon = CITIES[destination]

fwd = search(vectors, meta, stats,
             query={"first_lat": orig_lat, "first_lon": orig_lon,
                    "last_lat":  dest_lat, "last_lon":  dest_lon},
             k=k,
             mask_features=["first_lat", "first_lon", "last_lat", "last_lon"])
fwd["direction"] = f"{origin} → {destination}"

results = fwd
if both_dirs:
    rev = search(vectors, meta, stats,
                 query={"first_lat": dest_lat, "first_lon": dest_lon,
                        "last_lat":  orig_lat, "last_lon":  orig_lon},
                 k=k,
                 mask_features=["first_lat", "first_lon", "last_lat", "last_lon"])
    rev["direction"] = f"{destination} → {origin}"
    results = pd.concat([fwd, rev]).sort_values("distance").reset_index(drop=True)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trips found", len(results))
c2.metric("Avg duration", f"{results['duration_minutes'].mean():.0f} min")
c3.metric("Avg cells", f"{results['n_cells'].mean():.1f}")
c4.metric("Closest match", f"{results['distance'].min():.4f}")

st.divider()

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

sequences = fetch_sequences(tuple(results["trip_id"].tolist()))

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
    # Centre map between the two cities
    centre = [(orig_lat + dest_lat) / 2, (orig_lon + dest_lon) / 2]
    m = folium.Map(location=centre, zoom_start=7, tiles="CartoDB positron")

    # City markers
    for name, (lat, lon) in [(origin, (orig_lat, orig_lon)), (destination, (dest_lat, dest_lon))]:
        folium.Marker(
            location=[lat, lon],
            tooltip=name,
            icon=folium.Icon(color="blue", icon="map-marker"),
        ).add_to(m)

    for i, trip in enumerate(map_trips):
        colour = COLOURS[i % len(COLOURS)]
        coords_ll = [[lat, lon] for lat, lon in trip["coords"]]
        r = trip["meta"]
        label = (f'{r["trip_id"]}  |  {r["duration_minutes"]:.0f} min  '
                 f'|  {int(r["n_cells"])} cells  |  dist {r["distance"]:.4f}')
        folium.PolyLine(coords_ll, color=colour, weight=3, opacity=0.8, tooltip=label).add_to(m)
        folium.CircleMarker(coords_ll[0],  radius=5, color=colour, fill=False, weight=2,
                            tooltip="START").add_to(m)
        folium.CircleMarker(coords_ll[-1], radius=5, color=colour, fill=True,
                            fill_color=colour, fill_opacity=1.0, weight=2,
                            tooltip="END").add_to(m)

    st_folium(m, width=1200, height=520, returned_objects=[])
else:
    st.warning("No coordinate data available for matched trips.")

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

st.subheader("Matched trips")
display = results[[
    "direction", "trip_id", "vehicle_id", "trip_start", "trip_end",
    "duration_minutes", "n_cells", "n_handovers", "distance",
]].copy()
display["trip_start"] = display["trip_start"].astype(str).str[:16]
display["trip_end"]   = display["trip_end"].astype(str).str[:16]
display["duration_minutes"] = display["duration_minutes"].round(1)
display["distance"] = display["distance"].round(4)
st.download_button("⬇ Download JSON", data=display.to_json(orient="records", indent=2), file_name="trips.json", mime="application/json")
st.dataframe(display, use_container_width=True, hide_index=True)
