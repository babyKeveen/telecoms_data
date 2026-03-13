"""
Page 1: Route Map
Visualise cell-to-cell handovers on a map, weighted by frequency.
"""
import sys
sys.path.insert(0, "/home/jovyan")

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from pipeline.ingest import load_raw, clean
from pipeline.handovers import extract_handovers, build_graph

st.set_page_config(page_title="Route Map", layout="wide")
st.title("🗺️ Most Used Network Routes")

@st.cache_data
def get_data():
    df = clean(load_raw())
    handovers = extract_handovers(df)
    return df, handovers

df, handovers = get_data()

# Sidebar filters
trips = ["All"] + sorted(df["trip"].unique().tolist())
selected_trip = st.sidebar.selectbox("Filter by Trip", trips)

if selected_trip != "All":
    plot_df = df[df["trip"] == selected_trip]
    plot_ho = handovers[handovers["trip"] == selected_trip]
else:
    plot_df = df
    plot_ho = handovers

# Map
centre = [plot_df["lat"].mean(), plot_df["long"].mean()]
m = folium.Map(location=centre, zoom_start=13, tiles="CartoDB positron")

# Draw GPS track
coords = list(zip(plot_df["lat"], plot_df["long"]))
folium.PolyLine(coords, color="#4A90D9", weight=2, opacity=0.6).add_to(m)

# Mark handover points
for _, row in plot_ho.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=4,
        color="#F5A623",
        fill=True,
        popup=f"Handover: {int(row['from_cell'])} → {int(row['to_cell'])}<br>RSRP: {row['rsrp']} dBm"
    ).add_to(m)

st_folium(m, width=1100, height=600)

col1, col2 = st.columns(2)
col1.metric("Total Handovers", f"{len(plot_ho):,}")
col2.metric("Unique Cell Pairs", f"{plot_ho.groupby(['from_cell','to_cell']).ngroups:,}")

# Top routes table
st.subheader("Top Handover Routes")
top_routes = (
    plot_ho.groupby(["from_cell", "to_cell"])
    .agg(count=("trip", "count"), avg_rsrp=("rsrp", "mean"))
    .reset_index()
    .sort_values("count", ascending=False)
    .head(20)
)
top_routes["avg_rsrp"] = top_routes["avg_rsrp"].round(1)
st.dataframe(top_routes, use_container_width=True)
