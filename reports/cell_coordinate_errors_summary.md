# AT&T Cell Coordinate Data Quality Report

**Prepared:** 2026-05-17  
**Dataset:** Ford SIM vehicle LTE handover events — AT&T (MCC 310, MNC 410)  
**Scope:** Full calendar year 2025

---

## Executive Summary

A neighbor-consistency analysis of 2,133,183 vehicle trips across 93,568 vehicles
identified **3,285 AT&T cell IDs** whose coordinates in the
`shared_cell_location_lat_lon.csv` lookup are inconsistent with their observed
geographic context in trip data. Of these, **1,087 are high-confidence
errors** (>500 mile deviation, ≥10 independent trip observations).

The errors manifest as cell IDs mapped to the wrong US region — most commonly an
east-coast cell mapped to a west-coast coordinate or vice versa, and a smaller number
mapped to Hawaii or Alaska. This causes trip routes to appear to "teleport" across the
country when visualised.

One cell (`310-410-22504712`, San Bernardino CA → Kershaw County SC) has already been
corrected in the lookup as a validated example.

---

## Methodology

### Data sources
- **Trip sequences:** 2,133,183 trips, each containing an ordered `cell_sequence`
  of cell IDs visited during the trip (avg ~15 cells/trip)
- **Cell lookup:** `shared_cell_location_lat_lon.csv` — 1,059,347 AT&T (310-410)
  cell records with lat/lon, state, and county

### Detection algorithm

For each cell ID that appears in at least one trip sequence:

1. **Collect neighbor coordinates.** For every position the cell appears in a trip
   sequence, record the coordinates of the cells immediately before and after it.
   These are its geographic neighbors.

2. **Compute median neighbor position.** Take the median latitude and median longitude
   across all neighbor observations. The median is used (not mean) to be robust against
   a small number of other bad cells in the same sequences.

3. **Compute deviation.** Calculate the distance between the cell's actual coordinate
   (from the CSV) and its inferred coordinate (median neighbor position):

   ```
   distance_miles = sqrt((actual_lat − inferred_lat)² + (actual_lon − inferred_lon)²) × 69
   ```

4. **Flag if deviation > 100 miles with ≥5 neighbor samples.**

### Why this works

A vehicle travelling a continuous road route visits geographically adjacent cells in
sequence. If cell A consistently appears between cells that are in South Carolina across
hundreds of independent trips, its correct location is in South Carolina — regardless of
what the lookup CSV records. The more trips a cell appears in, the more confident the
inferred position.

---

## Findings

### Deviation distribution (flagged cells)

| Deviation range  | Cell count |
|------------------|------------|
| 100–200 miles    | 1,390      |
| 200–500 miles    | 669        |
| 500–1,000 miles  | 264        |
| 1,000–5,000 miles| 959        |
| > 5,000 miles    | 3          |
| **Total**        | **3,285** |

### Worst offenders (top 5 by deviation)

```
   global_cell_id current_state inferred_state  deviation_miles  n_neighbor_samples
 310-410-59376137        Hawaii          Maine           6246.3                  24
 310-410-59182351        Hawaii          Maine           6163.0                 368
 310-410-64290824        Alaska       Arkansas           5156.9                  79
310-410-125703696        Alaska      Tennessee           4961.0                   8
310-410-253955849     Tennessee         Alaska           4951.5                  20
```

### Common error patterns

- **East ↔ West coast swap:** Cell IDs appear to be reused across AT&T regions.
  The same numeric cell ID is used in both a New England market and a Pacific Northwest
  market; the lookup resolves to the wrong region.
- **Continental US ↔ Hawaii/Alaska:** A small number of cell IDs resolve to Hawaii
  or Alaska despite the vehicles consistently operating in the continental US.

---

## Deliverables

| File | Description |
|------|-------------|
| `cell_coordinate_errors.csv` | All 3,285 flagged cells with current coordinates, inferred coordinates, deviation, and neighbor sample count |
| `cell_coordinate_errors_map.html` | Interactive map showing 1,087 high-confidence errors (>500 mi, ≥10 samples) with correction vectors |

### CSV schema

| Column | Description |
|--------|-------------|
| `global_cell_id` | Lookup key (format: `310-410-<cell_id>`) |
| `cell_id` | Numeric cell ID |
| `current_lat` / `current_lon` | Coordinate currently in the CSV |
| `current_state` / `current_county` | Location metadata currently in the CSV |
| `inferred_lat` / `inferred_lon` | Median neighbor coordinate from trip data |
| `inferred_state` | Most common state among neighbor cells |
| `deviation_miles` | Distance between current and inferred coordinate |
| `n_neighbor_samples` | Number of trip observations (confidence indicator) |

---

## Recommended Actions

1. **Validate the inferred coordinates** against an authoritative cell tower database
   (e.g. OpenCelliD, internal AT&T network inventory) before bulk-applying corrections.
   The inferred coordinates are accurate to within a few miles for high-confidence cells
   but should not be applied blindly to low-confidence ones (< 10 neighbor samples).

2. **Prioritise high-confidence corrections first:** the 1,087 cells with
   >500 mile deviation and ≥10 neighbor samples are unambiguously wrong and safe to correct.

3. **Investigate the reuse pattern:** the east↔west coast cell ID collisions suggest the
   lookup was built by merging regional exports without deduplicating numeric cell IDs.
   The root fix is to enforce uniqueness on `global_cell_id` (MCC-MNC-cell_id) rather
   than cell_id alone.

4. **Re-validate periodically:** as the fleet grows into new geographies, new cell IDs
   will enter the lookup. The neighbor-consistency check can be run as a scheduled
   validation step against updated lookup exports.
