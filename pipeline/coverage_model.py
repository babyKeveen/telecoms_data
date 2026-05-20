"""
Coverage gap prediction — model loader and on-demand cell scorer.

Loads the trained HistGradientBoostingClassifier and scores a set of cells
for a given (hour_of_day, day_of_week) context.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = Path("/home/jovyan/data/models/coverage_gap_model.pkl")

_cache = {}   # module-level cache so Streamlit cache_resource isn't required


def load_model() -> dict:
    if "artifact" not in _cache:
        with open(MODEL_PATH, "rb") as f:
            _cache["artifact"] = pickle.load(f)
    return _cache["artifact"]


def score_cells(cell_coords: dict, hour: int, dow: int, month: int = 10) -> pd.DataFrame:
    """
    Score a dict of {cell_id: (lat, lon)} for P(gap) at a given time.

    Returns a DataFrame with columns: cell_id, lat, lon, p_gap
    sorted descending by p_gap.
    """
    if not cell_coords:
        return pd.DataFrame(columns=["cell_id", "lat", "lon", "p_gap"])

    artifact = load_model()
    model    = artifact["model"]
    features = artifact["features"]

    def _cyc(v, max_val):
        return np.sin(2 * np.pi * v / max_val), np.cos(2 * np.pi * v / max_val)

    hour_sin, hour_cos = _cyc(hour,  24)
    dow_sin,  dow_cos  = _cyc(dow,    7)
    mon_sin,  mon_cos  = _cyc(month, 12)

    rows = [(cid, lat, lon) for cid, (lat, lon) in cell_coords.items()
            if lat is not None and lon is not None]
    if not rows:
        return pd.DataFrame(columns=["cell_id", "lat", "lon", "p_gap"])

    df = pd.DataFrame(rows, columns=["cell_id", "lat", "lon"])
    df["hour_sin"] = hour_sin
    df["hour_cos"] = hour_cos
    df["dow_sin"]  = dow_sin
    df["dow_cos"]  = dow_cos
    df["mon_sin"]  = mon_sin
    df["mon_cos"]  = mon_cos

    df["p_gap"] = model.predict_proba(df[features])[:, 1]
    return df[["cell_id", "lat", "lon", "p_gap"]].sort_values("p_gap", ascending=False)
