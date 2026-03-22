"""
merge_gps.py
============
Combines all per-session frame_gps_map.csv files into one master CSV.

Run this AFTER extract_frames.py has processed all sessions in a batch.

Input:
  data/processed/batch_1/
    2622_Farwell_Ave-.../frame_gps_map.csv
    2647_Glendale_Blvd-.../frame_gps_map.csv
    ...

Output:
  data/processed/batch_1/master_gps_map.csv
    session, frame_file, frame_index, video_time_s, frame_abs_ms,
    lat, lon, altitude, speed, bearing, horizontalAccuracy, in_gps_range

Usage:
  python scripts/merge_gps.py               # merges DEFAULT_BATCH
  python scripts/merge_gps.py --batch batch_2
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# ──────────────────────────────────────────────
# ✏️  EDIT THESE TWO LINES TO CONFIGURE
# ──────────────────────────────────────────────
PROJECT_ROOT  = Path("/Users/nandiyang/Documents/robot-navigable-city")
DEFAULT_BATCH = "batch_1"
# ──────────────────────────────────────────────


def resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def merge_batch(batch: str) -> None:
    batch_processed = resolve_path(f"data/processed/{batch}")

    if not batch_processed.exists():
        sys.exit(f"[ERROR] Processed batch folder not found: {batch_processed}\n"
                 f"        Run extract_frames.py first.")

    # Find all session frame_gps_map.csv files
    csv_files = sorted(batch_processed.glob("*/frame_gps_map.csv"))
    if not csv_files:
        sys.exit(f"[ERROR] No frame_gps_map.csv files found in {batch_processed}\n"
                 f"        Run extract_frames.py first.")

    print(f"\nBatch   : {batch}")
    print(f"Found   : {len(csv_files)} session GPS files")
    print()

    dfs = []
    for csv_path in csv_files:
        session_name = csv_path.parent.name
        df = pd.read_csv(csv_path)

        # Add session column as first column
        df.insert(0, "session", session_name)

        # QC report per session
        oor = (~df["in_gps_range"]).sum()
        print(f"  {session_name}")
        print(f"    Frames : {len(df)}  |  In GPS range: {df['in_gps_range'].sum()}  "
              f"|  Out of range: {oor}")
        if oor > 0:
            print(f"    WARN   : {oor} frames outside GPS window")

        dfs.append(df)

    # Combine all sessions
    master = pd.concat(dfs, ignore_index=True)

    # Save master CSV
    out_path = batch_processed / "master_gps_map.csv"
    master.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"  Master GPS map saved")
    print(f"  Path    : {out_path}")
    print(f"  Total   : {len(master)} frames across {len(csv_files)} sessions")
    print(f"  Columns : {list(master.columns)}")
    print(f"{'='*60}\n")

    # Sanity check — confirm all frame_files are unique
    dupes = master["frame_file"].duplicated().sum()
    if dupes > 0:
        print(f"  WARN : {dupes} duplicate frame filenames detected.")
        print(f"         This means two sessions produced identically named frames.")
        print(f"         Check make_session_prefix() in extract_frames.py.")
    else:
        print(f"  OK   : All {len(master)} frame filenames are unique.\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge per-session frame_gps_map.csv files into one master CSV.",
    )
    parser.add_argument(
        "--batch", "-b",
        default=None,
        help=f"Batch name (default: {DEFAULT_BATCH})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch = args.batch or DEFAULT_BATCH
    merge_batch(batch)
