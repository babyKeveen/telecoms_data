"""
Home.py — Streamlit entry point
Run from inside container: streamlit run /home/jovyan/app/Home.py
"""
import streamlit as st

st.set_page_config(
    page_title="Telco Signal POC",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Telco Vehicle Signal Analysis")
st.markdown("""
### POC — LTE Drive Test Data Explorer

Use the sidebar to navigate between:

- **🗺️ Route Map** — Most used network routes visualised as cell-to-cell handovers
- **⚠️ Gap Analysis** — Where coverage fails: poor RSRP, negative SINR, handover stress zones
- **📊 Signal Quality** — Signal metric distributions across the route
""")

st.info("Data source: SRFG Austrian Highway LTE Dataset (IEEE DataPort / CRAWDAD)")

