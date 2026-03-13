import streamlit as st
import pandas as pd

st.title("Telecom Network Route & Gap Analysis")

# Load your sample data
df = pd.read_csv("/home/jovyan/telco-poc/data/SRFG-v1.csv")

# Sidebar for filtering
selected_route = st.sidebar.selectbox("Select Route ID", df['route_id'].unique())

# 1. Visualize Most Used Routes
st.header("Most Used Network Routes")
# Filter data for success logic
success_df = df[df['status'] == 'Success']
st.map(success_df) # Quick map of tower locations

# 2. Visualize the Gaps (The Gold)
st.header("Network Gaps & Failures")
failure_df = df[df['status'] == 'Failure']
st.warning(f"Detected {len(failure_df)} handover failures on this route.")
st.dataframe(failure_df)
