import math, pandas as pd, duckdb, folium

# -------------------------------------------------------------------
# Coordinate lookup
# -------------------------------------------------------------------
print('Loading coord lookup...')
raw = pd.read_csv('/home/jovyan/data/sim/raw/shared_cell_location_lat_lon.csv',
                  usecols=['global_cell_id','latitude','longitude'])
lookup = {}
for row in raw.itertuples(index=False):
    parts = str(row.global_cell_id).split('-')
    if len(parts) < 3:
        continue
    try:
        if int(parts[0]) == 310 and int(parts[1]) == 410:
            lookup[int(parts[2])] = (float(row.latitude), float(row.longitude))
    except ValueError:
        continue
print(f'  {len(lookup):,} cells')

# -------------------------------------------------------------------
# Michigan cities and corridors
# -------------------------------------------------------------------
CITIES = {
    'Detroit':      (42.331, -83.046),
    'Ann Arbor':    (42.281, -83.748),
    'Dearborn':     (42.322, -83.176),
    'Southfield':   (42.473, -83.221),
    'Troy':         (42.561, -83.147),
    'Grand Rapids': (42.963, -85.668),
    'Holland':      (42.787, -86.109),
    'Lansing':      (42.732, -84.556),
    'East Lansing': (42.736, -84.484),
    'Flint':        (43.013, -83.688),
    'Novi':         (42.480, -83.475),
    'Kalamazoo':    (42.292, -85.587),
}

CORRIDORS = [
    ('Detroit',      'Ann Arbor'),
    ('Detroit',      'Dearborn'),
    ('Detroit',      'Southfield'),
    ('Detroit',      'Troy'),
    ('Grand Rapids', 'Holland'),
    ('Grand Rapids', 'Lansing'),
    ('Lansing',      'East Lansing'),
    ('Flint',        'Detroit'),
    ('Ann Arbor',    'Novi'),
    ('Kalamazoo',    'Grand Rapids'),
]

COLOURS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#9a6324', '#000075',
]

MATCH_RADIUS_DEG   = 10 / 69.0   # ~10 miles
CORRIDOR_WIDTH_DEG = 20 / 69.0   # ~20 miles either side of the A->B line
MAX_TRIPS_PER_CORRIDOR = 20
TRIPS_DIR = '/home/jovyan/data/stage/trips'

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def city_cell_ids(city_name):
    clat, clon = CITIES[city_name]
    return [cid for cid, (lat, lon) in lookup.items()
            if abs(lat - clat) <= MATCH_RADIUS_DEG and abs(lon - clon) <= MATCH_RADIUS_DEG]

def perp_dist(lat_p, lon_p, lat_a, lon_a, lat_b, lon_b):
    dx, dy = lat_b - lat_a, lon_b - lon_a
    lsq = dx * dx + dy * dy
    if lsq == 0:
        return math.sqrt((lat_p - lat_a) ** 2 + (lon_p - lon_a) ** 2)
    return abs(dx * (lon_a - lon_p) - (lat_a - lat_p) * dy) / math.sqrt(lsq)

def proj_t(lat_p, lon_p, lat_a, lon_a, lat_b, lon_b):
    dx, dy = lat_b - lat_a, lon_b - lon_a
    lsq = dx * dx + dy * dy
    if lsq == 0:
        return 0.0
    return ((lat_p - lat_a) * dx + (lon_p - lon_a) * dy) / lsq

# -------------------------------------------------------------------
# Build map
# -------------------------------------------------------------------
con = duckdb.connect()
m = folium.Map(location=[42.6, -84.5], zoom_start=7, tiles='CartoDB positron')
total_plotted = 0
features = []

for (city_a, city_b), colour in zip(CORRIDORS, COLOURS):
    lat_a, lon_a = CITIES[city_a]
    lat_b, lon_b = CITIES[city_b]
    city_dist = math.sqrt((lat_a - lat_b) ** 2 + (lon_a - lon_b) ** 2)
    min_disp  = city_dist * 0.5

    a_ids = city_cell_ids(city_a)
    b_ids = city_cell_ids(city_b)
    if not a_ids or not b_ids:
        print(f'  SKIP {city_a} <-> {city_b}: no cells in lookup')
        continue

    con.register('ca', pd.DataFrame({'cell_id': a_ids}))
    con.register('cb', pd.DataFrame({'cell_id': b_ids}))

    trips_df = con.execute(f"""
        SELECT trip_id, trip_start, duration_minutes, n_cells, cell_sequence
        FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
        WHERE (first_cell IN (SELECT cell_id FROM ca) AND last_cell IN (SELECT cell_id FROM cb))
           OR (first_cell IN (SELECT cell_id FROM cb) AND last_cell IN (SELECT cell_id FROM ca))
        ORDER BY n_cells DESC
        LIMIT 500
    """).df()

    plotted = 0
    for _, row in trips_df.iterrows():
        coords = []
        for tok in str(row['cell_sequence']).split(' -> '):
            try:
                cid = int(tok.strip())
            except ValueError:
                continue
            if cid in lookup:
                lat, lon = lookup[cid]
                coords.append((cid, lat, lon))

        if not coords:
            continue

        clipped = [
            (cid, lat, lon) for cid, lat, lon in coords
            if -0.2 <= proj_t(lat, lon, lat_a, lon_a, lat_b, lon_b) <= 1.2
            and perp_dist(lat, lon, lat_a, lon_a, lat_b, lon_b) <= CORRIDOR_WIDTH_DEG
        ]

        if len(clipped) < 0.5 * len(coords) or len(clipped) < 2:
            continue
        span = math.sqrt((clipped[0][1] - clipped[-1][1]) ** 2 +
                         (clipped[0][2] - clipped[-1][2]) ** 2)
        if span < min_disp:
            continue

        ll = [[lat, lon] for _, lat, lon in clipped]
        label = '{} <-> {}  |  {}min  |  {} cells'.format(
            city_a, city_b, int(row['duration_minutes']), row['n_cells'])
        folium.PolyLine(ll, color=colour, weight=2, opacity=0.65, tooltip=label).add_to(m)

        for _, lat, lon in clipped[1:-1]:
            folium.CircleMarker(
                [lat, lon], radius=2,
                color='#000', fill=True, fill_color='#000', fill_opacity=0.5, weight=1,
            ).add_to(m)

        folium.CircleMarker(ll[0], radius=4, color=colour,
                            fill=False, weight=2).add_to(m)
        folium.CircleMarker(ll[-1], radius=4, color=colour,
                            fill=True, fill_color=colour, fill_opacity=1.0, weight=2).add_to(m)

        features.append({
            "type": "Feature",
            "properties": {
                "trip_id":          row['trip_id'],
                "corridor":         '{} <-> {}'.format(city_a, city_b),
                "origin":           city_a,
                "destination":      city_b,
                "trip_start":       str(row['trip_start']),
                "duration_minutes": round(float(row['duration_minutes']), 1),
                "n_cells":          int(row['n_cells']),
                "cells_plotted":    len(clipped),
                "colour":           colour,
            },
            "geometry": {
                "type": "LineString",
                # GeoJSON uses [lon, lat] order
                "coordinates": [[lon, lat] for _, lat, lon in clipped],
            },
        })

        plotted += 1
        if plotted == MAX_TRIPS_PER_CORRIDOR:
            break

    print('  {:15} <-> {:15}  {} trips plotted'.format(city_a, city_b, plotted))
    total_plotted += plotted

# City markers
for name, (lat, lon) in CITIES.items():
    folium.CircleMarker(
        [lat, lon], radius=7,
        color='#333', fill=True, fill_color='white', fill_opacity=1.0, weight=2,
        tooltip='<b>{}</b>'.format(name),
    ).add_to(m)

# Legend
legend_lines = ['<b>Michigan Corridors</b><br>']
for (a, b), col in zip(CORRIDORS, COLOURS):
    legend_lines.append(
        '<span style="color:{};font-size:16px">&#9644;</span> {} &harr; {}<br>'.format(col, a, b)
    )
legend_html = (
    '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;'
    'padding:12px;border-radius:6px;border:1px solid #ccc;font-size:12px;line-height:1.9">'
    + ''.join(legend_lines) + '</div>'
)
m.get_root().html.add_child(folium.Element(legend_html))

out_html = '/home/jovyan/telco-poc/notebooks/michigan_corridors.html'
m.save(out_html)
print('\nTotal trips plotted: {}'.format(total_plotted))
print('Saved: {}'.format(out_html))

import json
geojson = {"type": "FeatureCollection", "features": features}
out_json = '/home/jovyan/telco-poc/notebooks/michigan_corridors.geojson'
with open(out_json, 'w') as f:
    json.dump(geojson, f, indent=2)
print('Saved: {}'.format(out_json))
