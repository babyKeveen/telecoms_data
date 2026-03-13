"""
handovers.py
------------
Derive handover events from raw signal logs.
A handover = cell_id changes between consecutive rows within the same trip.
"""
import pandas as pd
import networkx as nx
from loguru import logger


def extract_handovers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame of handover events:
    trip, time, from_cell, to_cell, from_lat, from_long, rsrp_at_handover, signal_at_handover
    """
    logger.info("Extracting handover events...")

    df = df.sort_values(["trip", "time"]).copy()
    df["prev_cell"] = df.groupby("trip")["cell_id"].shift(1)
    df["prev_lat"] = df.groupby("trip")["lat"].shift(1)
    df["prev_long"] = df.groupby("trip")["long"].shift(1)

    handovers = df[df["cell_id"] != df["prev_cell"]].dropna(subset=["prev_cell"]).copy()
    handovers = handovers.rename(columns={
        "prev_cell": "from_cell",
        "cell_id": "to_cell",
        "prev_lat": "from_lat",
        "prev_long": "from_long",
    })

    handovers["from_cell"] = handovers["from_cell"].astype(int)
    result = handovers[["trip", "time", "from_cell", "to_cell",
                         "from_lat", "from_long", "lat", "long",
                         "rsrp", "sinr", "signal"]].reset_index(drop=True)

    logger.info(f"Found {len(result):,} handover events across {result['trip'].nunique()} trips")
    return result


def build_graph(handovers: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed graph where:
      nodes = cell IDs
      edges = handover pairs, weighted by frequency
    """
    G = nx.DiGraph()

    edge_counts = handovers.groupby(["from_cell", "to_cell"]).size().reset_index(name="count")
    edge_rsrp = handovers.groupby(["from_cell", "to_cell"])["rsrp"].mean().reset_index(name="avg_rsrp")
    edges = edge_counts.merge(edge_rsrp, on=["from_cell", "to_cell"])

    for _, row in edges.iterrows():
        G.add_edge(
            int(row["from_cell"]),
            int(row["to_cell"]),
            weight=row["count"],
            avg_rsrp=round(row["avg_rsrp"], 1)
        )

    logger.info(f"Graph: {G.number_of_nodes()} cells, {G.number_of_edges()} handover edges")
    return G
