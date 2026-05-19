#!/usr/bin/env python3
"""
build_trips_v2.py
-----------------
Rebuilds the trip stage without calendar-day boundary splits.

The original pipeline processed one day at a time, causing trips that cross
midnight to be falsely split into two. This script processes each vehicle as
a continuous stream across all dates so the stationary-period logic determines
trip boundaries regardless of calendar day.

Same trip detection logic as v1:
  - STATIONARY_POLLS = 3  (3 x ~450s = ~22 min on same cell = parked)
  - MIN_HANDOVERS   = 4
  - MIN_CELLS       = 10

Output: /home/jovyan/data/stage/trips_v2/
        hive-partitioned by DATE(trip_start), same schema as trips_v1.

To activate after verification:
    mv /home/jovyan/data/stage/trips   /home/jovyan/data/stage/trips_v1_bak
    mv /home/jovyan/data/stage/trips_v2 /home/jovyan/data/stage/trips

Run: python notebooks/build_trips_v2.py [--resume]
"""
import argparse
import os
import time
from pathlib import Path

import duckdb

HANDOVER_GLOB    = "/home/jovyan/data/stage/handover_events/**/*.parquet"
OUT_DIR          = Path("/home/jovyan/data/stage/trips_v2")
TMP_DIR          = "/tmp/duckdb_tmp_v2"
CHECKPOINT_FILE  = OUT_DIR / "_checkpoint.txt"

STATIONARY_POLLS = 3
MIN_HANDOVERS    = 4
MIN_CELLS        = 10
VEHICLE_BATCH    = 500
MAX_GAP_HOURS    = 4    # gap > 4h between events = forced trip boundary
MAX_TRIP_HOURS   = 48   # hard cap — no single trip segment can exceed this


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def make_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{TMP_DIR}'")
    con.execute("SET memory_limit='16GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    con.execute(f"""
        CREATE OR REPLACE VIEW handover_events AS
        SELECT * FROM read_parquet('{HANDOVER_GLOB}', hive_partitioning=true)
    """)
    return con


# ---------------------------------------------------------------------------
# Per-batch trip extraction
# ---------------------------------------------------------------------------

def process_batch(con: duckdb.DuckDBPyConnection, vehicles: list[str], tmp_out: Path) -> int:
    ids_sql = ", ".join(f"'{v}'" for v in vehicles)

    # Step A: stationary detection and trip segmentation across all dates
    con.execute("DROP TABLE IF EXISTS trips_raw")
    con.execute(f"""
    CREATE TABLE trips_raw AS
    WITH
    ranked AS (
        SELECT
            vehicle_id, imsi, event_ts, cell_id, rat,
            LAG(cell_id) OVER (PARTITION BY vehicle_id ORDER BY event_ts) AS prev_cell,
            CASE
                WHEN cell_id != LAG(cell_id) OVER (PARTITION BY vehicle_id ORDER BY event_ts)
                  OR LAG(cell_id) OVER (PARTITION BY vehicle_id ORDER BY event_ts) IS NULL
                THEN 1 ELSE 0
            END AS cell_changed,
            -- gap > MAX_GAP_HOURS since last event = vehicle was dark; force trip boundary
            CASE
                WHEN DATEDIFF('hour',
                         LAG(event_ts) OVER (PARTITION BY vehicle_id ORDER BY event_ts),
                         event_ts) >= {MAX_GAP_HOURS}
                THEN TRUE ELSE FALSE
            END AS idle_gap
        FROM handover_events
        WHERE vehicle_id IN ({ids_sql})
    ),
    with_run AS (
        SELECT *,
            SUM(cell_changed) OVER (PARTITION BY vehicle_id ORDER BY event_ts) AS cell_run_id
        FROM ranked
    ),
    with_poll_count AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY vehicle_id, cell_run_id ORDER BY event_ts) AS same_cell_poll_count
        FROM with_run
    ),
    with_stationary AS (
        SELECT *,
            same_cell_poll_count >= {STATIONARY_POLLS} AS stationary,
            LAG(same_cell_poll_count >= {STATIONARY_POLLS})
                OVER (PARTITION BY vehicle_id ORDER BY event_ts) AS prev_stationary
        FROM with_poll_count
    ),
    with_trip_flags AS (
        SELECT *,
            CASE
                WHEN idle_gap                                       -- dark for MAX_GAP_HOURS+
                  OR (cell_changed = 1
                      AND (prev_stationary OR prev_stationary IS NULL))
                THEN 1 ELSE 0
            END AS trip_start_flag
        FROM with_stationary
    ),
    with_trip_seq AS (
        SELECT *,
            SUM(trip_start_flag) OVER (PARTITION BY vehicle_id ORDER BY event_ts) AS trip_seq,
            NOT stationary AS in_trip
        FROM with_trip_flags
    )
    SELECT *,
        CASE WHEN NOT stationary
            THEN vehicle_id || '_trip_' || LPAD(CAST(trip_seq AS VARCHAR), 3, '0')
            ELSE NULL
        END AS trip_id
    FROM with_trip_seq
    """)

    # Step B: aggregate one row per trip
    con.execute("DROP TABLE IF EXISTS trip_agg")
    con.execute(f"""
    CREATE TABLE trip_agg AS
    SELECT
        vehicle_id, imsi, trip_id, trip_seq,
        DATE(MIN(event_ts))                                    AS trip_date,
        MIN(event_ts)                                          AS trip_start,
        MAX(event_ts)                                          AS trip_end,
        COUNT(*)                                               AS n_events,
        COUNT(DISTINCT cell_id)                                AS n_cells,
        SUM(CASE WHEN cell_changed = 1 THEN 1 ELSE 0 END)     AS n_handovers,
        FIRST(cell_id ORDER BY event_ts)                       AS first_cell,
        LAST(cell_id ORDER BY event_ts)                        AS last_cell,
        MODE(rat)                                              AS dominant_rat
    FROM trips_raw
    WHERE in_trip
    GROUP BY vehicle_id, imsi, trip_id, trip_seq
    HAVING n_handovers >= {MIN_HANDOVERS}
       AND COUNT(DISTINCT cell_id) >= {MIN_CELLS}
       AND DATEDIFF('hour', MIN(event_ts), MAX(event_ts)) <= {MAX_TRIP_HOURS}
    """)

    con.execute("DROP TABLE IF EXISTS trips_raw")

    # Step C: cell sequences — join spans all event_dates for cross-midnight trips
    con.execute("DROP TABLE IF EXISTS cell_sequences")
    con.execute(f"""
    CREATE TABLE cell_sequences AS
    SELECT
        a.trip_id,
        STRING_AGG(h.cell_id, ' -> ' ORDER BY h.event_ts) AS cell_sequence
    FROM handover_events h
    JOIN trip_agg a
        ON  h.vehicle_id = a.vehicle_id
        AND h.event_ts  >= a.trip_start
        AND h.event_ts  <= a.trip_end
    GROUP BY a.trip_id
    """)

    # Step D: write flat batch parquet (will be repartitioned in final step)
    n_trips = con.execute(f"""
    COPY (
        SELECT
            a.vehicle_id,
            a.imsi,
            a.trip_date   AS event_date,
            a.trip_id,
            a.trip_seq,
            a.trip_start,
            a.trip_end,
            ROUND(DATEDIFF('second', a.trip_start, a.trip_end) / 60.0, 1) AS duration_minutes,
            a.n_handovers,
            a.n_cells,
            a.n_events,
            a.first_cell,
            a.last_cell,
            a.dominant_rat,
            s.cell_sequence
        FROM trip_agg a
        LEFT JOIN cell_sequences s ON a.trip_id = s.trip_id
        WHERE a.trip_id IS NOT NULL
        ORDER BY vehicle_id, trip_start
    ) TO '{tmp_out}' (FORMAT PARQUET)
    """).fetchone()[0]

    con.execute("DROP TABLE IF EXISTS trip_agg")
    con.execute("DROP TABLE IF EXISTS cell_sequences")

    return n_trips


# ---------------------------------------------------------------------------
# Final repartition: flat batch files → hive layout by trip_start date
# ---------------------------------------------------------------------------

def repartition(con: duckdb.DuckDBPyConnection, tmp_dir: Path, out_dir: Path) -> int:
    batch_glob = str(tmp_dir / "batch_*.parquet")
    print(f"\nRepartitioning batch files → {out_dir}/event_date=*/trips.parquet ...")

    # Remove any existing hive partitions so we start clean
    import shutil
    for d in out_dir.glob("event_date=*"):
        shutil.rmtree(d)

    n_total = con.execute(f"""
    COPY (
        SELECT * FROM read_parquet('{batch_glob}')
        ORDER BY vehicle_id, trip_start
    ) TO '{out_dir}' (
        FORMAT PARQUET,
        PARTITION_BY (event_date),
        FILENAME_PATTERN 'trips'
    )
    """).fetchone()[0]

    return n_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip vehicle batches already written to tmp dir")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("/tmp/trips_v2_tmp")  # outside OUT_DIR so COPY TO sees an empty dir
    tmp_dir.mkdir(exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    con = make_con()

    # Get all vehicles
    print("Fetching vehicle list...")
    vehicles = [
        r[0] for r in
        con.execute("SELECT DISTINCT vehicle_id FROM handover_events ORDER BY vehicle_id").fetchall()
    ]
    print(f"  {len(vehicles):,} vehicles to process")

    # Split into batches
    batches = [vehicles[i:i + VEHICLE_BATCH] for i in range(0, len(vehicles), VEHICLE_BATCH)]
    total_batches = len(batches)
    print(f"  {total_batches} batches of up to {VEHICLE_BATCH} vehicles each\n")

    total_trips = 0
    skipped     = 0
    run_start   = time.time()

    for batch_num, batch in enumerate(batches, 1):
        tmp_out = tmp_dir / f"batch_{batch_num:04d}.parquet"

        if args.resume and tmp_out.exists():
            skipped += 1
            print(f"[{batch_num}/{total_batches}] SKIP (already written)")
            continue

        t0 = time.time()
        n = process_batch(con, batch, tmp_out)
        total_trips += n
        elapsed = time.time() - t0
        print(f"[{batch_num}/{total_batches}] {len(batch)} vehicles → {n:,} trips ({elapsed:.1f}s)")

    print(f"\nBatch pass complete. {total_batches - skipped} processed, {skipped} skipped.")
    print(f"Trips so far: {total_trips:,}")

    # Repartition into hive layout
    t0 = time.time()
    n_final = repartition(con, tmp_dir, OUT_DIR)
    print(f"Repartitioned {n_final:,} trips in {time.time()-t0:.1f}s")

    # Remove tmp batch files
    import shutil
    shutil.rmtree(tmp_dir)
    print("Temp batch files cleaned up.")

    total_elapsed = time.time() - run_start
    print(f"\nDone. Total runtime: {total_elapsed/60:.1f} min")
    print(f"Output: {OUT_DIR}")
    print()
    print("To activate:")
    print(f"  mv /home/jovyan/data/stage/trips     /home/jovyan/data/stage/trips_v1_bak")
    print(f"  mv /home/jovyan/data/stage/trips_v2  /home/jovyan/data/stage/trips")


if __name__ == "__main__":
    main()
