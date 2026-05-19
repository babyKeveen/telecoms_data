#!/usr/bin/env python3
"""
Rebuild the trips stage from scratch.

Clears all existing per-date trip partitions under data/stage/trips/,
then executes build_trips_duckDB.ipynb cell by cell using nbconvert.

Usage:
    python rebuild_trips_stage.py [--dry-run]

    --dry-run   Show what would be deleted without deleting or running the notebook.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

STAGE_DIR   = Path("/home/jovyan/data/stage/trips")
NOTEBOOK    = Path("/home/jovyan/telco-poc/notebooks/build_trips_duckDB.ipynb")
OUTPUT_NB   = Path("/home/jovyan/telco-poc/notebooks/build_trips_duckDB_out.ipynb")

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 1. Validate
# ---------------------------------------------------------------------------
if not NOTEBOOK.exists():
    sys.exit(f"ERROR: notebook not found at {NOTEBOOK}")

if not STAGE_DIR.exists():
    print(f"Stage directory does not exist yet ({STAGE_DIR}) — nothing to clear.")
else:
    partitions = sorted(STAGE_DIR.glob("event_date=*"))
    print(f"Found {len(partitions)} existing partitions in {STAGE_DIR}")
    for p in partitions[:5]:
        print(f"  {p.name}")
    if len(partitions) > 5:
        print(f"  ... and {len(partitions) - 5} more")

    if args.dry_run:
        print("\n--dry-run: would delete all partitions listed above. Exiting.")
        sys.exit(0)

    # ---------------------------------------------------------------------------
    # 2. Clear existing partitions
    # ---------------------------------------------------------------------------
    print(f"\nDeleting {len(partitions)} partitions...")
    t0 = time.time()
    for p in partitions:
        shutil.rmtree(p)
    print(f"Cleared in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# 3. Execute notebook
# ---------------------------------------------------------------------------
print(f"\nExecuting {NOTEBOOK.name} ...")
print(f"Output will be written to {OUTPUT_NB.name}")
print("This will take 20-40 minutes.\n")

import subprocess
result = subprocess.run(
    [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=7200",
        "--output", str(OUTPUT_NB),
        str(NOTEBOOK),
    ],
    capture_output=False,
)

if result.returncode != 0:
    sys.exit(f"\nERROR: notebook execution failed (exit code {result.returncode}).")

print(f"\nDone. Executed notebook saved to {OUTPUT_NB}")
print("Check the output notebook for per-date progress and any errors.")
