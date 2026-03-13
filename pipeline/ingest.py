"""
ingest.py
---------
Load raw CSV, clean, convert to Parquet for fast downstream reads.
Run once: python -m pipeline.ingest
"""
import os
import pandas as pd
from pathlib import Path
from loguru import logger

RAW_PATH = Path(os.getenv("RAW_DATA_PATH", "/home/jovyan/telco-poc/data/SRFG-v1.csv"))
PARQUET_PATH = Path(os.getenv("PARQUET_PATH", "/home/jovyan/telco-poc/data/SRFG-v1.parquet"))


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path, parse_dates=["time"])
    logger.info(f"Loaded {len(df):,} rows")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data...")

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()

    # Drop rows with no cell_id or location
    df = df.dropna(subset=["cell_id", "lat", "long"])

    # Sort within each trip by time
    df = df.sort_values(["trip", "time"]).reset_index(drop=True)

    # Cast types
    df["cell_id"] = df["cell_id"].astype(int)
    df["rsrp"] = pd.to_numeric(df["rsrp"], errors="coerce")
    df["sinr"] = pd.to_numeric(df["sinr"], errors="coerce")
    df["rsrq"] = pd.to_numeric(df["rsrq"], errors="coerce")
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce")

    logger.info(f"Clean data: {len(df):,} rows, {df['trip'].nunique()} trips")
    return df


def to_parquet(df: pd.DataFrame, path: Path = PARQUET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved to {path}")


if __name__ == "__main__":
    df = load_raw()
    df = clean(df)
    to_parquet(df)
