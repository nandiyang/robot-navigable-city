"""
postprocess.py
==============
Phase 1 — Spatial Deduplication of Detections
Robot-Navigable City Project

Problem:
  Walking past an obstacle at 1 FPS generates ~4-6 detections of the same
  object at nearly identical GPS coordinates. This creates clusters of dots
  on the map that represent one real-world object.

Solution:
  Group detections of the same class within MERGE_DISTANCE_M meters of each
  other → keep the detection with the highest confidence score as the
  representative point.

Input:
  data/processed/batch_1/inference/detections.geojson

Output:
  data/processed/batch_1/inference/detections_clean.geojson
  data/processed/batch_1/inference/detections_clean.csv

Usage:
  python scripts/postprocess.py              # uses DEFAULT_BATCH
  python scripts/postprocess.py --dist 10    # increase merge distance
  python scripts/postprocess.py --batch batch_2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# ✏️  EDIT THESE TO CONFIGURE
# ──────────────────────────────────────────────
MIN_CONF           = 0.50
PROJECT_ROOT       = Path("/Users/nandiyang/Documents/robot-navigable-city")
DEFAULT_BATCH      = "batch_1"
MERGE_DISTANCE_M   = 10.0   # merge detections within this many meters
                            # increase to 10m if map still looks cluttered
                            # decrease to 3m if merging too aggressively
# ──────────────────────────────────────────────


def resolve_path(p) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """
    Calculate distance in meters between two GPS points.
    Uses Haversine formula — accurate for small distances.
    """
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi       = np.radians(lat2 - lat1)
    dlambda    = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def deduplicate(df: pd.DataFrame, merge_dist_m: float) -> pd.DataFrame:
    """
    Greedy spatial deduplication per class:
      1. Sort by confidence descending
      2. For each detection, check if any already-kept detection of the
         same class is within merge_dist_m meters
      3. If yes — discard (it's a duplicate of the same real object)
      4. If no  — keep as a new unique detection

    Returns deduplicated DataFrame.
    """
    kept_rows = []

    for class_name, group in df.groupby("class_name"):
        # Sort by confidence — keep best detection when merging
        group = group.sort_values("confidence", ascending=False).reset_index(drop=True)

        kept_lats = []
        kept_lons = []
        kept_indices = []

        for idx, row in group.iterrows():
            if pd.isna(row["lat"]) or pd.isna(row["lon"]):
                continue  # skip unmatched GPS

            is_duplicate = False
            for klat, klon in zip(kept_lats, kept_lons):
                dist = haversine_m(row["lat"], row["lon"], klat, klon)
                if dist <= merge_dist_m:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept_lats.append(row["lat"])
                kept_lons.append(row["lon"])
                kept_indices.append(idx)

        kept_rows.append(group.loc[kept_indices])

    if not kept_rows:
        return pd.DataFrame()

    result = pd.concat(kept_rows, ignore_index=True)
    return result


def load_detections(batch: str) -> pd.DataFrame:
    """Load detections.csv from inference output."""
    csv_path = resolve_path(f"data/processed/{batch}/inference/detections.csv")
    if not csv_path.exists():
        sys.exit(f"[ERROR] detections.csv not found: {csv_path}\n"
                 f"        Run inference.py first.")
    df = pd.read_csv(csv_path)
    print(f"  Loaded  : {len(df)} raw detections")
    return df


def to_geojson(df: pd.DataFrame) -> dict:
    """Convert detections DataFrame to GeoJSON FeatureCollection."""
    features = []
    for _, row in df.iterrows():
        if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
            continue
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["lon"], row["lat"]]
            },
            "properties": {
                "frame_file":   row.get("frame_file", ""),
                "session":      row.get("session", ""),
                "class_name":   row["class_name"],
                "confidence":   row["confidence"],
                "altitude":     row.get("altitude", None),
                "speed":        row.get("speed", None),
                "bearing":      row.get("bearing", None),
                "bbox":         [row["x1"], row["y1"], row["x2"], row["y2"]],
            }
        }
        features.append(feature)
    return {"type": "FeatureCollection", "features": features}


def run(batch: str, merge_dist_m: float):
    print(f"\n{'='*60}")
    print(f"  Robot-Navigable City — Spatial Deduplication")
    print(f"{'='*60}")
    print(f"  Batch         : {batch}")
    print(f"  Merge distance: {merge_dist_m}m")
    print(f"{'='*60}\n")

    # Load raw detections
    print("[1/3] Loading detections...")
    df = load_detections(batch)

    # Filter by confidence
    before_conf = len(df)
    df = df[df["confidence"] >= MIN_CONF]
    print(f"  Filtered: {before_conf} → {len(df)} detections (conf ≥ {MIN_CONF})")

    # Print before stats
    print(f"\n  Before deduplication:")
    for cls in df["class_name"].unique():
        n = (df["class_name"] == cls).sum()
        print(f"    {cls:25s}: {n}")

    # Deduplicate
    print(f"\n[2/3] Deduplicating within {merge_dist_m}m radius...")
    df_clean = deduplicate(df, merge_dist_m)

    # Print after stats
    print(f"\n  After deduplication:")
    for cls in df_clean["class_name"].unique():
        n = (df_clean["class_name"] == cls).sum()
        print(f"    {cls:25s}: {n}")

    reduction = (1 - len(df_clean) / len(df)) * 100
    print(f"\n  Reduced : {len(df)} → {len(df_clean)} detections ({reduction:.0f}% removed)")

    # Save outputs
    print(f"\n[3/3] Saving outputs...")
    out_dir = resolve_path(f"data/processed/{batch}/inference")

    csv_path     = out_dir / "detections_clean.csv"
    geojson_path = out_dir / "detections_clean.geojson"

    df_clean.to_csv(csv_path, index=False)
    geojson = to_geojson(df_clean)
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    print(f"  detections_clean.csv     → {csv_path}")
    print(f"  detections_clean.geojson → {geojson_path}")
    print(f"\n  Load detections_clean.geojson into Kepler.gl")
    print(f"  If map still looks cluttered, re-run with --dist 10")
    print(f"  If merging too aggressively, re-run with --dist 3")
    print(f"{'='*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spatially deduplicate detections from inference.py."
    )
    parser.add_argument("--batch", default=DEFAULT_BATCH,
                        help=f"Batch name (default: {DEFAULT_BATCH})")
    parser.add_argument("--dist",  type=float, default=MERGE_DISTANCE_M,
                        help=f"Merge distance in meters (default: {MERGE_DISTANCE_M})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(batch=args.batch, merge_dist_m=args.dist)
