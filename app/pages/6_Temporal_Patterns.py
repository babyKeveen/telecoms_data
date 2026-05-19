"""
Page 6: Temporal Patterns
Fleet activity and network quality broken down by hour, day-of-week, and month.
All data from the trips Parquet — no handover events scan needed.
"""
from datetime import date

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

TRIPS_DIR = "/home/jovyan/data/stage/trips"

DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="Temporal Patterns", layout="wide")
st.title("📅 Temporal Patterns")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
default_start = date(2025, 1, 1)
default_end   = date(2025, 12, 31)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date   = st.sidebar.date_input("End date",   value=default_end)

st.sidebar.divider()
st.sidebar.caption(
    "All times are in the local timezone recorded by the vehicle SIM (US AT&T)."
)

# ---------------------------------------------------------------------------
# Queries — one pass per aggregation level, all cached
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading hourly patterns...", ttl=600)
def query_by_hour(start_date, end_date):
    con = duckdb.connect()
    return con.execute(f"""
        SELECT
            hour(trip_start)                                              AS hour_of_day,
            COUNT(*)                                                      AS trips,
            COUNT(DISTINCT vehicle_id)                                    AS vehicles,
            ROUND(AVG(duration_minutes), 1)                               AS avg_duration_min,
            ROUND(AVG(n_handovers / NULLIF(duration_minutes, 0) * 60), 2) AS avg_ho_per_hour
        FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
        WHERE event_date BETWEEN '{start_date}' AND '{end_date}'
          AND duration_minutes > 0
        GROUP BY 1
        ORDER BY 1
    """).df()


@st.cache_data(show_spinner="Loading day-of-week patterns...", ttl=600)
def query_by_dow(start_date, end_date):
    con = duckdb.connect()
    return con.execute(f"""
        SELECT
            dayofweek(trip_start)                                         AS dow,
            COUNT(*)                                                      AS trips,
            COUNT(DISTINCT vehicle_id)                                    AS vehicles,
            ROUND(AVG(duration_minutes), 1)                               AS avg_duration_min,
            ROUND(AVG(n_handovers / NULLIF(duration_minutes, 0) * 60), 2) AS avg_ho_per_hour
        FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
        WHERE event_date BETWEEN '{start_date}' AND '{end_date}'
          AND duration_minutes > 0
        GROUP BY 1
        ORDER BY 1
    """).df()


@st.cache_data(show_spinner="Loading monthly trends...", ttl=600)
def query_by_month(start_date, end_date):
    con = duckdb.connect()
    return con.execute(f"""
        SELECT
            strftime(trip_start, '%Y-%m')                                 AS month,
            COUNT(*)                                                      AS trips,
            COUNT(DISTINCT vehicle_id)                                    AS vehicles,
            ROUND(AVG(duration_minutes), 1)                               AS avg_duration_min,
            ROUND(AVG(n_handovers / NULLIF(duration_minutes, 0) * 60), 2) AS avg_ho_per_hour,
            ROUND(AVG(n_cells), 1)                                        AS avg_cells_per_trip,
            ROUND(SUM(duration_minutes) / 60, 0)                          AS total_hours
        FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
        WHERE event_date BETWEEN '{start_date}' AND '{end_date}'
          AND duration_minutes > 0
        GROUP BY 1
        ORDER BY 1
    """).df()


@st.cache_data(show_spinner="Loading activity heatmap...", ttl=600)
def query_heatmap(start_date, end_date):
    con = duckdb.connect()
    return con.execute(f"""
        SELECT
            dayofweek(trip_start) AS dow,
            hour(trip_start)      AS hour_of_day,
            COUNT(*)              AS trips
        FROM read_parquet('{TRIPS_DIR}/event_date=*/*.parquet', hive_partitioning=true)
        WHERE event_date BETWEEN '{start_date}' AND '{end_date}'
          AND duration_minutes > 0
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()


hour_df  = query_by_hour(start_date, end_date)
dow_df   = query_by_dow(start_date, end_date)
month_df = query_by_month(start_date, end_date)
heat_df  = query_heatmap(start_date, end_date)

if hour_df.empty:
    st.warning("No trips found for the selected date range.")
    st.stop()

# ---------------------------------------------------------------------------
# Top-level KPIs
# ---------------------------------------------------------------------------
total_trips    = int(month_df["trips"].sum())
total_vehicles = int(month_df["vehicles"].max())   # peak active in any month
total_hours    = int(month_df["total_hours"].sum())
avg_duration   = round(month_df["avg_duration_min"].mean(), 1)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total trips",        f"{total_trips:,}")
k2.metric("Peak active vehicles", f"{total_vehicles:,}")
k3.metric("Total drive hours",  f"{total_hours:,}")
k4.metric("Avg trip duration",  f"{avg_duration} min")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Hour of Day", "Day of Week", "Monthly Trends", "Activity Heatmap"]
)

# ── Tab 1: Hour of Day ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Trip volume by hour of day")
    col_left, col_right = st.columns(2)

    with col_left:
        fig = px.bar(
            hour_df, x="hour_of_day", y="trips",
            labels={"hour_of_day": "Hour (local)", "trips": "Trip starts"},
            color="trips",
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False, height=320)
        fig.update_xaxes(tickvals=list(range(0, 24)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Trip start time. Peak hours show when the fleet is most active.")

    with col_right:
        fig2 = px.bar(
            hour_df, x="hour_of_day", y="avg_duration_min",
            labels={"hour_of_day": "Hour (local)", "avg_duration_min": "Avg duration (min)"},
            color="avg_duration_min",
            color_continuous_scale="Greens",
        )
        fig2.update_layout(coloraxis_showscale=False, height=320)
        fig2.update_xaxes(tickvals=list(range(0, 24)))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Trips starting later in the day tend to be longer (overnight / long-haul).")

    st.divider()
    st.subheader("Network stress by hour (handovers per hour of drive time)")
    st.caption(
        "Higher handover rate = more cell transitions per driving hour. "
        "Elevated values at peak traffic hours may indicate congested cells forcing re-selection."
    )
    fig3 = px.line(
        hour_df, x="hour_of_day", y="avg_ho_per_hour",
        labels={"hour_of_day": "Hour (local)", "avg_ho_per_hour": "Avg handovers / drive-hour"},
        markers=True,
    )
    fig3.update_layout(height=280)
    fig3.update_xaxes(tickvals=list(range(0, 24)))
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 2: Day of Week ────────────────────────────────────────────────────────
with tab2:
    dow_df["day_name"] = dow_df["dow"].apply(lambda d: DAYS[int(d)])
    dow_df["is_weekend"] = dow_df["dow"].isin([0, 6])

    st.subheader("Trip volume and duration by day of week")
    col_left, col_right = st.columns(2)

    with col_left:
        fig = px.bar(
            dow_df, x="day_name", y="trips",
            category_orders={"day_name": DAYS},
            color="is_weekend",
            color_discrete_map={True: "#4363d8", False: "#e6194b"},
            labels={"day_name": "", "trips": "Trips", "is_weekend": "Weekend"},
        )
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red = weekday, blue = weekend.")

    with col_right:
        fig2 = px.bar(
            dow_df, x="day_name", y="avg_duration_min",
            category_orders={"day_name": DAYS},
            color="is_weekend",
            color_discrete_map={True: "#4363d8", False: "#e6194b"},
            labels={"day_name": "", "avg_duration_min": "Avg duration (min)", "is_weekend": "Weekend"},
        )
        fig2.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Active vehicles by day of week")
        fig3 = px.bar(
            dow_df, x="day_name", y="vehicles",
            category_orders={"day_name": DAYS},
            labels={"day_name": "", "vehicles": "Distinct vehicles"},
            color_discrete_sequence=["#42d4f4"],
        )
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Distinct vehicles that started at least one trip on this day of week.")

    with col_b:
        st.subheader("Network stress by day of week")
        fig4 = px.bar(
            dow_df, x="day_name", y="avg_ho_per_hour",
            category_orders={"day_name": DAYS},
            labels={"day_name": "", "avg_ho_per_hour": "Avg handovers / drive-hour"},
            color_discrete_sequence=["#f58231"],
        )
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)

# ── Tab 3: Monthly Trends ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Fleet activity over time")

    col_left, col_right = st.columns(2)

    with col_left:
        fig = px.bar(
            month_df, x="month", y="trips",
            labels={"month": "", "trips": "Trips"},
            color_discrete_sequence=["#3cb44b"],
        )
        fig.update_layout(height=300, title="Trips per month")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig2 = px.line(
            month_df, x="month", y="vehicles",
            labels={"month": "", "vehicles": "Active vehicles"},
            markers=True,
            color_discrete_sequence=["#e6194b"],
        )
        fig2.update_layout(height=300, title="Active vehicles per month")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Network quality over time")

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        fig3 = px.line(
            month_df, x="month", y="avg_ho_per_hour",
            labels={"month": "", "avg_ho_per_hour": "Avg handovers / drive-hour"},
            markers=True,
            color_discrete_sequence=["#911eb4"],
        )
        fig3.update_layout(height=300, title="Handover rate per month")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            "Rising handover rate can signal cell congestion, seasonal RF changes "
            "(foliage, weather), or fleet expansion into new geographies."
        )

    with col_right2:
        fig4 = px.line(
            month_df, x="month", y="avg_cells_per_trip",
            labels={"month": "", "avg_cells_per_trip": "Avg cells per trip"},
            markers=True,
            color_discrete_sequence=["#f032e6"],
        )
        fig4.update_layout(height=300, title="Avg cells visited per trip")
        st.plotly_chart(fig4, use_container_width=True)
        st.caption(
            "Avg unique cells per trip reflects route diversity / trip length mix. "
            "Increasing trend = longer or more varied trips over time."
        )

    st.divider()
    st.subheader("Monthly summary table")
    st.dataframe(
        month_df.rename(columns={
            "month":             "Month",
            "trips":             "Trips",
            "vehicles":          "Active vehicles",
            "avg_duration_min":  "Avg duration (min)",
            "avg_ho_per_hour":   "Avg HO/drive-hr",
            "avg_cells_per_trip":"Avg cells/trip",
            "total_hours":       "Total drive hours",
        }),
        use_container_width=True,
        hide_index=True,
    )

# ── Tab 4: Activity Heatmap ───────────────────────────────────────────────────
with tab4:
    st.subheader("Trip intensity: hour of day × day of week")
    st.caption(
        "Each cell shows the total trip starts for that hour + day combination "
        "across the selected date range. Reveals the fleet's operating schedule at a glance."
    )

    # Pivot to hour × day matrix
    pivot = heat_df.pivot(index="hour_of_day", columns="dow", values="trips").fillna(0)
    pivot.columns = [DAYS[int(c)] for c in pivot.columns]
    pivot = pivot[DAYS]  # ensure Monday-first reading order left-to-right

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[f"{h:02d}:00" for h in pivot.index],
        colorscale="YlOrRd",
        hoverongaps=False,
        hovertemplate="<b>%{x}</b> at <b>%{y}</b><br>Trips: %{z:,}<extra></extra>",
    ))
    fig.update_layout(
        height=600,
        yaxis=dict(autorange="reversed"),
        xaxis_title="",
        yaxis_title="Hour of day (local)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Peak hours per day")
    peak_rows = []
    for dow_val in range(7):
        day_data = heat_df[heat_df["dow"] == dow_val].sort_values("trips", ascending=False)
        if not day_data.empty:
            top = day_data.iloc[0]
            peak_rows.append({
                "Day":        DAYS[dow_val],
                "Peak hour":  f"{int(top['hour_of_day']):02d}:00",
                "Trips":      int(top["trips"]),
            })
    st.dataframe(pd.DataFrame(peak_rows), use_container_width=True, hide_index=True)
