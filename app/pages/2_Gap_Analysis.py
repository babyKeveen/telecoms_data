"""
Page 2: Gap Analysis
Show where coverage fails — the 'real gold'.
"""
import sys
sys.path.insert(0, "/home/jovyan")

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from pipeline.ingest import load_raw, clean
from pipeline.gaps import detect_poor_signal, gap_summary

st.set_page_config(page_title="Gap Analysis", layout="wide")
st.title("⚠️ Coverage Gap Analysis")

@st.cache_data
def get_data():
    df = clean(load_raw())
    df = detect_poor_signal(df)
    return df

df = get_data()
summary = gap_summary(df)

# KPI row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{summary['total_records']:,}")
col2.metric("Gap Records", f"{summary['gap_records']:,}", f"{summary['gap_pct']}%")
col3.metric("Worst RSRP", f"{summary['worst_rsrp']} dBm")
col4.metric("Cells with Gaps", f"{summary['unique_gap_cells']}")

st.divider()

# Gap type breakdown
st.subheader("Gap Type Breakdown")
gap_types = {
    "RSRP < -90 dBm": df["rsrp_gap"].sum(),
    "SINR < 0 dB": df["sinr_gap"].sum(),
    "Signal ≤ 2 bars": df["signal_gap"].sum(),
}
st.bar_chart(gap_types)

st.divider()

# Map — good signal vs gaps
st.subheader("Coverage Map")
centre = [df["lat"].mean(), df["long"].mean()]
m = folium.Map(location=centre, zoom_start=13, tiles="CartoDB positron")

good = df[~df["is_gap"]]
bad = df[df["is_gap"]]

# Good signal — blue (every 5th point to keep map fast)
for _, row in good.iloc[::5].iterrows():
    folium.CircleMarker([row["lat"], row["long"]], radius=2,
                        color="#4A90D9", fill=True, opacity=0.4).add_to(m)

# Gap zones — red
for _, row in bad.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=5,
        color="#D0021B",
        fill=True,
        popup=f"RSRP: {row['rsrp']} dBm | SINR: {row['sinr']} dB | Signal: {row['signal']}/5"
    ).add_to(m)

st_folium(m, width=1100, height=600)

# Worst spots table
st.subheader("Worst Coverage Spots")
worst = (
    df[df["is_gap"]]
    .sort_values("rsrp")
    [["trip", "time", "lat", "long", "cell_id", "rsrp", "sinr", "signal"]]
    .head(50)
)
st.dataframe(worst, use_container_width=True)
