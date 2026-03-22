"""
inference.py
============
Phase 1 — Run YOLOv8 Inference + GPS Join
Robot-Navigable City Project

What this does:
  1. Runs trained YOLO model on all frames in processed/batch_N/
  2. Joins each detection to GPS coordinates via master_gps_map.csv
  3. Outputs:
       detections.csv     — all detections with GPS
       detections.geojson — map-ready format for Kepler.gl / QGIS

Usage:
  python scripts/inference.py                    # uses defaults below
  python scripts/inference.py --batch batch_2    # different batch
  python scripts/inference.py --conf 0.25        # lower confidence threshold
"""

import json
import sys
from pathlib import Path

import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────
# ✏️  EDIT THESE TO CONFIGURE
# ──────────────────────────────────────────────
PROJECT_ROOT  = Path("/Users/nandiyang/Documents/robot-navigable-city")
DEFAULT_BATCH = "batch_1"
DEFAULT_CONF  = 0.35    # confidence threshold
DEFAULT_IOU   = 0.45    # NMS IOU threshold — leave as-is

# Model weights — uses best.pt from most recent training run
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs/detect/train_v1/weights/best.pt"
# ──────────────────────────────────────────────


def resolve_path(p) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_master_gps(batch: str) -> pd.DataFrame:
    """Load master_gps_map.csv for the batch."""
    gps_path = resolve_path(f"data/processed/{batch}/master_gps_map.csv")
    if not gps_path.exists():
        sys.exit(f"[ERROR] master_gps_map.csv not found: {gps_path}\n"
                 f"        Run merge_gps.py first.")
    df = pd.read_csv(gps_path)
    print(f"  GPS map : {len(df)} frames loaded from {gps_path.name}")
    return df


def collect_frames(batch: str) -> list[Path]:
    """Collect all frame JPEGs from all sessions in processed/batch/"""
    batch_path = resolve_path(f"data/processed/{batch}")
    frames = sorted(batch_path.glob("*/frames/*.jpg"))
    if not frames:
        sys.exit(f"[ERROR] No frames found in {batch_path}\n"
                 f"        Run extract_frames.py first.")
    print(f"  Frames  : {len(frames)} found across all sessions")
    return frames


def run_inference(
    weights: Path,
    frames: list[Path],
    conf: float,
    iou: float,
) -> pd.DataFrame:
    """Run YOLO inference on all frames. Returns raw detections DataFrame."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERROR] ultralytics not found. Run: pip install ultralytics")

    if not weights.exists():
        sys.exit(f"[ERROR] Model weights not found: {weights}\n"
                 f"        Run train.py first.")

    print(f"\n  Loading model: {weights.name}")
    model = YOLO(str(weights))

    records = []
    print(f"  Running inference at conf={conf}...")

    for frame_path in tqdm(frames, desc="  Inference", unit="frame"):
        results = model.predict(
            source    = str(frame_path),
            conf      = conf,
            iou       = iou,
            device    = "mps",      # Apple Silicon GPU
            verbose   = False,
            save      = False,      # don't save annotated images (saves disk space)
        )

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                records.append({
                    "frame_file":  frame_path.name,
                    "class_id":    int(box.cls.item()),
                    "class_name":  model.names[int(box.cls.item())],
                    "confidence":  round(float(box.conf.item()), 4),
                    "x1":          round(float(box.xyxy[0][0].item()), 1),
                    "y1":          round(float(box.xyxy[0][1].item()), 1),
                    "x2":          round(float(box.xyxy[0][2].item()), 1),
                    "y2":          round(float(box.xyxy[0][3].item()), 1),
                })

    print(f"  Found   : {len(records)} detections across {len(frames)} frames")
    return pd.DataFrame(records)


def join_gps(detections: pd.DataFrame, gps_map: pd.DataFrame) -> pd.DataFrame:
    """Join detections with GPS coordinates via frame_file."""
    if detections.empty:
        print("  WARN: No detections to join.")
        return detections

    # Keep only needed GPS columns
    gps_cols = ["frame_file", "session", "lat", "lon",
                 "altitude", "speed", "bearing", "in_gps_range"]
    gps_slim = gps_map[gps_cols].copy()

    merged = detections.merge(gps_slim, on="frame_file", how="left")

    # Flag detections that couldn't be matched to GPS
    unmatched = merged["lat"].isna().sum()
    if unmatched > 0:
        print(f"  WARN: {unmatched} detections could not be matched to GPS.")
        print(f"        Check that frame filenames match master_gps_map.csv.")

    out_of_range = (~merged["in_gps_range"].fillna(False)).sum()
    if out_of_range > 0:
        print(f"  WARN: {out_of_range} detections are outside GPS window.")

    print(f"  Joined  : {len(merged)} detections with GPS coordinates")
    return merged


def to_geojson(detections: pd.DataFrame) -> dict:
    """Convert detections DataFrame to GeoJSON FeatureCollection."""
    features = []
    for _, row in detections.iterrows():
        if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
            continue
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["lon"], row["lat"]]
            },
            "properties": {
                "frame_file":   row["frame_file"],
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

    return {
        "type": "FeatureCollection",
        "features": features
    }


def run(batch: str, weights: Path, conf: float, iou: float):
    print(f"\n{'='*60}")
    print(f"  Robot-Navigable City — Inference Pipeline")
    print(f"{'='*60}")
    print(f"  Batch   : {batch}")
    print(f"  Weights : {weights.name}")
    print(f"  Conf    : {conf}")
    print(f"{'='*60}\n")

    # Output directory
    out_dir = resolve_path(f"data/processed/{batch}/inference")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GPS map
    print("[1/4] Loading GPS map...")
    gps_map = load_master_gps(batch)

    # Collect frames
    print("\n[2/4] Collecting frames...")
    frames = collect_frames(batch)

    # Run inference
    print("\n[3/4] Running inference...")
    detections = run_inference(weights, frames, conf, iou)

    if detections.empty:
        print("\n  No detections found. Try lowering --conf threshold.")
        return

    # Join GPS
    print("\n[4/4] Joining GPS coordinates...")
    detections_geo = join_gps(detections, gps_map)

    # Save outputs
    csv_path     = out_dir / "detections.csv"
    geojson_path = out_dir / "detections.geojson"

    detections_geo.to_csv(csv_path, index=False)
    geojson = to_geojson(detections_geo)
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"  Total detections    : {len(detections_geo)}")
    for cls in detections_geo["class_name"].unique():
        n = (detections_geo["class_name"] == cls).sum()
        print(f"    {cls:25s}: {n}")
    print(f"\n  Outputs saved to: {out_dir}")
    print(f"    detections.csv     — {len(detections_geo)} rows")
    print(f"    detections.geojson — ready for Kepler.gl / QGIS")
    print(f"\n  Next step: open detections.geojson in Kepler.gl")
    print(f"    → https://kepler.gl/demo")
    print(f"{'='*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference and join detections to GPS."
    )
    parser.add_argument("--batch",   default=DEFAULT_BATCH,
                        help=f"Batch to run inference on (default: {DEFAULT_BATCH})")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help="Path to model weights (default: train_v1/best.pt)")
    parser.add_argument("--conf",    type=float, default=DEFAULT_CONF,
                        help=f"Confidence threshold (default: {DEFAULT_CONF})")
    parser.add_argument("--iou",     type=float, default=DEFAULT_IOU,
                        help=f"NMS IOU threshold (default: {DEFAULT_IOU})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        batch   = args.batch,
        weights = Path(args.weights),
        conf    = args.conf,
        iou     = args.iou,
    )
