"""Build the numpy vector database from the staged trips parquet."""
import sys
sys.path.insert(0, "/home/jovyan/telco-poc")

from pipeline.vectors_np import build

if __name__ == "__main__":
    build()
