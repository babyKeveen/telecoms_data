"""
Home.py — Streamlit entry point
Run from inside container: streamlit run /home/jovyan/app/Home.py
"""
import streamlit as st

APP_VERSION = "1.2.0"

st.set_page_config(
    page_title="Telco Signal POC",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Telco Vehicle Signal Analysis")
st.caption(f"v{APP_VERSION}")

st.markdown("""
### POC — LTE Drive Test Data Explorer

Use the sidebar to navigate between:

- **🗺️ Route Map** — Most used network routes visualised as cell-to-cell handovers
- **⚠️ Gap Analysis** — Coverage silence gaps, neighbour signal quality, and RSRP/RSRQ hotspots
- **🛣️ Corridor Analysis** — Trip density and gap hotspots along a city-pair corridor
- **🔍 Trip Search** — Find trips by feature similarity (duration, cells, handovers, ping, RSRP)
- **🛣️ Route Search** — Find trips matching an origin→destination city pair
- **📅 Temporal Patterns** — Fleet activity, handover rate, ping latency, and neighbour RSRP over time
- **📶 Signal Map** — Filter trips by date range, average RSRP, and duration; routes coloured by signal strength
- **📋 Monthly Report** — All trips for a selected month and state; full KPI table with start/end coordinates and JSON export
""")

st.info("Data source: US AT&T (MCC 310, MNC 410) — 2025 full-year SIM handover events")

