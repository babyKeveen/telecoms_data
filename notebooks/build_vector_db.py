"""
build_vector_db.py
------------------
One-time build of the FAISS trip vector database.
Re-run whenever the trips Parquet is rebuilt with new parameters.

Usage:
    python notebooks/build_vector_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.vectors import build, VECTOR_DIR

if __name__ == "__main__":
    build(VECTOR_DIR)
