"""
Page 3: Corridor Analysis
All trips between two selected cities, with silence-gap markers overlaid.
City filtering is pushed into DuckDB so Python never iterates 23M rows.
"""
import math
from datetime import date

import duckdb
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

COORD_CSV    = "/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv"
TRIPS_DIR    = "/home/jovyan/data/stage/trips"
HANDOVER_DIR = "/home/jovyan/data/stage/handover_events"
MAX_TRIPS_ON_MAP = 50

COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]

# (lat, lon) — city centres only; match radius is controlled by the sidebar slider
CITIES = {
    "Ann Arbor, MI":     (42.281, -83.748),
    "Atlanta, GA":       (33.749, -84.388),
    "Baltimore, MD":     (39.290, -76.612),
    "Boston, MA":        (42.360, -71.059),
    "Charlotte, NC":     (35.227, -80.843),
    "Chicago, IL":       (41.878, -87.630),
    "Cincinnati, OH":    (39.103, -84.512),
    "Cleveland, OH":     (41.499, -81.695),
    "Columbus, OH":      (39.961, -82.999),
    "Dallas, TX":        (32.776, -96.797),
    "Denver, CO":        (39.739,-104.984),
    "Detroit, MI":       (42.331, -83.046),
    "Houston, TX":       (29.760, -95.370),
    "Indianapolis, IN":  (39.768, -86.158),
    "Kansas City, MO":   (39.099, -94.578),
    "Los Angeles, CA":   (34.052,-118.244),
    "Louisville, KY":    (38.252, -85.758),
    "Memphis, TN":       (35.149, -90.048),
    "Milwaukee, WI":     (43.038, -87.906),
    "Minneapolis, MN":   (44.977, -93.265),
    "Nashville, TN":     (36.162, -86.781),
    "New York, NY":      (40.713, -74.006),
    "Philadelphia, PA":  (39.952, -75.165),
    "Pittsburgh, PA":    (40.440, -79.996),
    "Portland, OR":      (45.523,-122.676),
    "San Francisco, CA": (37.774,-122.419),
    "Seattle, WA":       (47.606,-122.332),
    "St. Louis, MO":     (38.627, -90.197),
    "Washington, DC":    (38.907, -77.037),
}

st.set_page_config(page_title="Corridor Analysis", layout="wide")
st.title("🛣️ Corridor Analysis")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
city_names = sorted(CITIES.keys())

st.sidebar.header("Corridor")
origin_name = st.sidebar.selectbox("Origin",      city_names, index=city_names.index("Detroit, MI"))
dest_name   = st.sidebar.selectbox("Destination", city_names, index=city_names.index("Ann Arbor, MI"))
both_dirs   = st.sidebar.toggle("Include reverse direction", value=True)

st.sidebar.divider()
st.sidebar.header("Filters")
start_date = st.sidebar.date_input("Start date", value=date(2025, 1, 1))
end_date   = st.sidebar.date_input("End date",   value=date(2025, 12, 31))
max_hours  = st.sidebar.slider("Max trip duration (hours)", 1.0, 24.0, 8.0, 0.5)

match_radius_miles   = st.sidebar.slider("City match radius (miles)", 3, 30, 8)
corridor_width_miles = st.sidebar.slider("Max corridor width (miles)", 5, 60, 15)
# 1° latitude ≈ 69 miles
MATCH_RADIUS_DEG   = match_radius_miles / 69.0
CORRIDOR_WIDTH_DEG = corridor_width_miles / 69.0

st.sidebar.divider()
show_gaps   = st.sidebar.toggle("Show silence gap hotspots", value=True)
min_gap_min = st.sidebar.slider("Min gap duration (minutes)", 5, 60, 15)

if origin_name == dest_name:
    st.warning("Origin and destination must be different cities.")
    st.stop()

origin = CITIES[origin_name]
dest   = CITIES[dest_name]

# Minimum displacement: 60% of straight-line distance between city centres.
# Eliminates short local trips whose first/last cells happen to fall in both boxes.
_city_dist = math.sqrt((origin[0] - dest[0])**2 + (origin[1] - dest[1])**2)
MIN_DISPLACEMENT = _city_dist * 0.6

if MATCH_RADIUS_DEG * 2 >= _city_dist:
    st.warning(
        f"City match radius ({match_radius_miles} miles) is too large — the two city boxes "
        "overlap. Reduce the radius or choose cities further apart."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Coordinate lookup (cached across all pages)
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
# ---------------------------------------------------------------------------
# Corridor geometry helpers
# ---------------------------------------------------------------------------
def perp_distance_deg(lat_p, lon_p, lat_a, lon_a, lat_b, lon_b) -> float:
    """Perpendicular distance from point P to line segment A→B, in degrees."""
    dx, dy = lat_b - lat_a, lon_b - lon_a
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.sqrt((lat_p - lat_a) ** 2 + (lon_p - lon_a) ** 2)
    cross = abs(dx * (lon_a - lon_p) - (lat_a - lat_p) * dy)
    return cross / math.sqrt(length_sq)

def projection_t(lat_p, lon_p, lat_a, lon_a, lat_b, lon_b) -> float:
    """Scalar projection of P onto A→B (0=at A, 1=at B)."""
    dx, dy = lat_b - lat_a, lon_b - lon_a
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return 0.0
    return ((lat_p - lat_a) * dx + (lon_p - lon_a) * dy) / length_sq

def in_corridor(lat, lon, width_deg) -> bool:
    """True if the coordinate is within width_deg of the origin→dest line segment."""
    t = projection_t(lat, lon, origin[0], origin[1], dest[0], dest[1])
    if not (-0.2 <= t <= 1.2):       # allow 20% margin beyond each city centre
        return False
    d = perp_distance_deg(lat, lon, origin[0], origin[1], dest[0], dest[1])
    return d <= width_deg

# City cell sets — which cell IDs fall within the sidebar-controlled bounding box
# ---------------------------------------------------------------------------
def city_cells(city: tuple) -> list[str]:
    clat, clon = city
    return [
        str(cid) for cid, (lat, lon) in coord_lookup.items()
        if abs(lat - clat) <= MATCH_RADIUS_DEG and abs(lon - clon) <= MATCH_RADIUS_DEG
    ]

# ---------------------------------------------------------------------------
# Query trips — filtering pushed into DuckDB via registered DataFrames
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Querying corridor trips...", ttl=300)
def query_corridor_trips(origin_name, dest_name, both_dirs, start_date, end_date, max_hours, match_radius_deg):
    a_cells = [
        str(cid) for cid, (lat, lon) in coord_lookup.items()
        if abs(lat - CITIES[origin_name][0]) <= match_radius_deg
        and abs(lon - CITIES[origin_name][1]) <= match_radius_deg
    ]
    b_cells = [
        str(cid) for cid, (lat, lon) in coord_lookup.items()
        if abs(lat - CITIES[dest_name][0]) <= match_radius_deg
        and abs(lon - CITIES[dest_name][1]) <= match_radius_deg
    ]

    if not a_cells or not b_cells:
        return pd.DataFrame()

    con = duckdb.connect()
    con.register("city_a", pd.DataFrame({"cell_id": a_cells}))
    con.register("city_b", pd.DataFrame({"cell_id": b_cells}))

    direction_sql = """
        (first_cell IN (SELECT cell_id FROM city_a) AND last_cell IN (SELECT cell_id FROM city_b))
        OR
        (first_cell IN (SELECT cell_id FROM city_b) AND last_cell IN (SELECT cell_id FROM city_a))
    """ if both_dirs else """
        first_cell IN (SELECT cell_id FROM city_a) AND last_cell IN (SELECT cell_id FROM city_b)
    """

    return con.execute(f"""
        SELECT trip_id, vehicle_id, trip_start, trip_end, duration_minutes,
               n_handovers, n_cells, first_cell, last_cell, cell_sequence
        FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
        WHERE event_date BETWEEN '{start_date}' AND '{end_date}'
          AND duration_minutes > 0
          AND duration_minutes <= {max_hours * 60}
          AND ({direction_sql})
        ORDER BY n_cells DESC
        LIMIT 2000
    """).df()

trips_df = query_corridor_trips(
    origin_name, dest_name, both_dirs, start_date, end_date, max_hours, MATCH_RADIUS_DEG
)

if trips_df.empty:
    st.warning(
        f"No trips found between **{origin_name}** and **{dest_name}** "
        "for the selected filters. Try widening the date range, increasing "
        "max duration, or enabling reverse direction."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Resolve full cell_sequence coordinates (Python — small result set now)
# ---------------------------------------------------------------------------
_origin_cells_set = set(city_cells(origin))
def direction_label(row) -> str:
    if str(row["first_cell"]).strip() in _origin_cells_set:
        return f"{origin_name} → {dest_name}"
    return f"{dest_name} → {origin_name}"

MIN_CORRIDOR_FRACTION = 0.6   # reject trip if <60% of its cells are within the corridor

corridor_trips = []
for _, row in trips_df.iterrows():
    # Resolve coordinates — skip individual cells not in lookup (don't discard whole trip)
    coords = []
    for tok in str(row["cell_sequence"]).split(" -> "):
        try:
            cid = int(tok.strip())
        except ValueError:
            continue
        if cid in coord_lookup:
            lat, lon = coord_lookup[cid]
            coords.append({"cell_id": cid, "lat": lat, "lon": lon})

    if not coords:
        continue

    # Clip to corridor — keep only cells within the perpendicular width of the A→B line
    clipped = [c for c in coords if in_corridor(c["lat"], c["lon"], CORRIDOR_WIDTH_DEG)]

    # Reject trip if most cells are off-corridor (loop, detour, or misclassified)
    if len(clipped) < MIN_CORRIDOR_FRACTION * len(coords):
        continue

    # Reject trips that don't span enough of the city-to-city distance
    if len(clipped) < 2:
        continue
    span = math.sqrt(
        (clipped[0]["lat"] - clipped[-1]["lat"]) ** 2 +
        (clipped[0]["lon"] - clipped[-1]["lon"]) ** 2
    )
    if span < MIN_DISPLACEMENT:
        continue

    corridor_trips.append({**row.to_dict(), "coordinates": clipped,
                            "direction": direction_label(row)})
    if len(corridor_trips) == MAX_TRIPS_ON_MAP:
        break

if not corridor_trips:
    st.warning(
        f"No trips passed the corridor filter for **{origin_name}** → **{dest_name}**. "
        f"Try increasing the corridor width (currently {corridor_width_miles} miles) "
        "or widening the date range."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Corridor bounding box (for gap filtering)
# ---------------------------------------------------------------------------
all_lats = [c["lat"] for t in corridor_trips for c in t["coordinates"]]
all_lons = [c["lon"] for t in corridor_trips for c in t["coordinates"]]
lat_pad  = max((max(all_lats) - min(all_lats)) * 0.15, 0.1)
lon_pad  = max((max(all_lons) - min(all_lons)) * 0.15, 0.1)
lat_min, lat_max = min(all_lats) - lat_pad, max(all_lats) + lat_pad
lon_min, lon_max = min(all_lons) - lon_pad, max(all_lons) + lon_pad

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
durations = [t["duration_minutes"] for t in corridor_trips if t["duration_minutes"]]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Trips shown",    f"{len(corridor_trips)} of {len(trips_df):,}")
col2.metric("Max cells",      max(t["n_cells"] for t in corridor_trips))
col3.metric("Avg duration",   f"{sum(durations)/len(durations)/60:.1f}h" if durations else "—")
directions = {t["direction"] for t in corridor_trips}
col4.metric("Directions",     " + ".join(sorted(directions)))

st.divider()

# ---------------------------------------------------------------------------
# Gap hotspots in corridor bounding box
# ---------------------------------------------------------------------------
gap_rows = []
if show_gaps:
    @st.cache_data(show_spinner="Fetching gap hotspots...", ttl=300)
    def query_gap_hotspots(start_date, end_date, min_gap_sec):
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
                prev_cell                                                  AS cell_id,
                COUNT(*)                                                   AS gap_events,
                COUNT(DISTINCT vehicle_id)                                 AS vehicles_affected,
                ROUND(AVG(DATEDIFF('second', prev_ts, event_ts)) / 60, 1) AS avg_gap_min
            FROM ordered
            WHERE prev_ts IS NOT NULL
              AND DATEDIFF('second', prev_ts, event_ts) >= {min_gap_sec}
            GROUP BY prev_cell
            ORDER BY gap_events DESC
        """).df()

    gaps_df = query_gap_hotspots(start_date, end_date, min_gap_min * 60)
    for _, row in gaps_df.iterrows():
        try:
            cid = int(row["cell_id"])
        except (ValueError, TypeError):
            continue
        if cid not in coord_lookup:
            continue
        clat, clon = coord_lookup[cid]
        if lat_min <= clat <= lat_max and lon_min <= clon <= lon_max:
            gap_rows.append({**row.to_dict(), "lat": clat, "lon": clon})

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
centre = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]
m = folium.Map(location=centre, zoom_start=8, tiles="CartoDB positron")

# City anchor markers
for name, (clat, clon) in [(origin_name, origin), (dest_name, dest)]:
    folium.CircleMarker(
        location=[clat, clon], radius=10,
        color="#0047AB", fill=True, fill_color="#0047AB", fill_opacity=0.9,
        tooltip=f"<b>{name}</b>",
    ).add_to(m)

# Trip polylines
for i, trip in enumerate(corridor_trips):
    colour    = COLOURS[i % len(COLOURS)]
    coords_ll = [[c["lat"], c["lon"]] for c in trip["coordinates"]]
    label = (
        f'{trip["trip_id"]}<br>'
        f'{trip["direction"]}<br>'
        f'{trip["n_cells"]} cells &nbsp;|&nbsp; {trip["duration_minutes"]/60:.1f}h'
    )
    folium.PolyLine(coords_ll, color=colour, weight=2, opacity=0.75, tooltip=label).add_to(m)
    # Intermediate tower dots
    for c in trip["coordinates"][1:-1]:
        folium.CircleMarker(
            location=[c["lat"], c["lon"]], radius=2,
            color="#000000", fill=True, fill_color="#000000", fill_opacity=0.7,
            weight=1, tooltip=f'Cell {c["cell_id"]}',
        ).add_to(m)
    # Start (open) and end (filled) markers
    folium.CircleMarker(
        coords_ll[0], radius=4, color=colour, fill=False, weight=2,
        tooltip=f'START — {str(trip["trip_start"])[:16]}',
    ).add_to(m)
    folium.CircleMarker(
        coords_ll[-1], radius=4, color=colour,
        fill=True, fill_color=colour, fill_opacity=1.0, weight=2,
        tooltip=f'END — {str(trip["trip_end"])[:16]}',
    ).add_to(m)

# Silence gap markers
if gap_rows:
    max_ev = max(r["gap_events"] for r in gap_rows)
    for r in gap_rows:
        radius = 4 + 10 * (r["gap_events"] / max_ev)
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=radius,
            color="#D0021B",
            fill=True,
            fill_opacity=0.65,
            tooltip=(
                f'⚠️ Cell {int(r["cell_id"])}<br>'
                f'Gap events: {int(r["gap_events"]):,}<br>'
                f'Vehicles: {int(r["vehicles_affected"])}<br>'
                f'Avg gap: {r["avg_gap_min"]} min'
            ),
        ).add_to(m)

st_folium(m, width=1200, height=640, returned_objects=[])

caption_parts = [f"Showing {len(corridor_trips)} trips (capped at {MAX_TRIPS_ON_MAP}, sorted by most cells)."]
if gap_rows:
    caption_parts.append(f"{len(gap_rows):,} silence-gap cells overlaid (red). Blue = city anchors.")
st.caption("  ".join(caption_parts))

st.divider()

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
st.subheader("Matched trips")
st.dataframe(
    pd.DataFrame([{
        "trip_id":      t["trip_id"],
        "direction":    t["direction"],
        "start":        str(t["trip_start"])[:16],
        "end":          str(t["trip_end"])[:16],
        "duration (h)": round(t["duration_minutes"] / 60, 2) if t["duration_minutes"] else None,
        "n_cells":      t["n_cells"],
        "n_handovers":  t["n_handovers"],
    } for t in corridor_trips]),
    use_container_width=True,
    hide_index=True,
)

if gap_rows:
    st.subheader("Gap hotspots on this corridor")
    st.dataframe(
        pd.DataFrame([{
            "cell_id":     int(r["cell_id"]),
            "gap_events":  int(r["gap_events"]),
            "vehicles":    int(r["vehicles_affected"]),
            "avg_gap_min": r["avg_gap_min"],
        } for r in sorted(gap_rows, key=lambda x: -x["gap_events"])]),
        use_container_width=True,
        hide_index=True,
    )
