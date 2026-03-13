"""
gaps.py
-------
Detect coverage gaps and weak signal zones from signal logs.

Three gap types:
  1. RSRP gap     — signal strength below threshold (default: -90 dBm)
  2. SINR gap     — signal-to-noise ratio negative (interference dominating)
  3. Handover stress zone — rapid successive handovers with low signal
"""
import pandas as pd
from loguru import logger

# Industry thresholds
RSRP_POOR = -90       # dBm — below this is poor coverage
SINR_POOR = 0         # dB  — negative SINR means noise dominates
SIGNAL_BAR_POOR = 2   # bars (out of 5)


def detect_poor_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where signal quality is poor across multiple metrics."""
    df = df.copy()
    df["rsrp_gap"] = df["rsrp"] < RSRP_POOR
    df["sinr_gap"] = df["sinr"] < SINR_POOR
    df["signal_gap"] = df["signal"] <= SIGNAL_BAR_POOR
    df["is_gap"] = df["rsrp_gap"] | df["sinr_gap"] | df["signal_gap"]

    n_gaps = df["is_gap"].sum()
    pct = 100 * n_gaps / len(df)
    logger.info(f"Poor signal rows: {n_gaps:,} ({pct:.1f}% of all records)")
    return df


def detect_handover_stress(handovers: pd.DataFrame,
                            window_seconds: int = 30,
                            min_handovers: int = 3) -> pd.DataFrame:
    """
    Find zones where N+ handovers occur within a short time window.
    Indicates unstable coverage / ping-pong between cells.
    """
    df = handovers.sort_values(["trip", "time"]).copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    stress_rows = []
    for trip, group in df.groupby("trip"):
        rolling = group.rolling(f"{window_seconds}s").count()["from_cell"]
        stressed = group[rolling >= min_handovers].copy()
        stressed["trip"] = trip
        stress_rows.append(stressed)

    if stress_rows:
        result = pd.concat(stress_rows).reset_index()
        logger.info(f"Handover stress zones: {len(result):,} events")
        return result
    return pd.DataFrame()


def gap_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict for dashboard display."""
    gap_df = df[df["is_gap"]]
    return {
        "total_records": len(df),
        "gap_records": len(gap_df),
        "gap_pct": round(100 * len(gap_df) / len(df), 1),
        "worst_rsrp": round(df["rsrp"].min(), 1),
        "avg_rsrp_in_gaps": round(gap_df["rsrp"].mean(), 1),
        "unique_gap_cells": gap_df["cell_id"].nunique(),
    }
