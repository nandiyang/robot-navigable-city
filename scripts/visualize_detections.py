"""
visualize_detections.py
=======================
Phase 1 — Annotated Frame Visualization
Robot-Navigable City Project

What this does:
  Draws bounding boxes and class labels on frames that have detections,
  saves them as annotated JPEGs. Only saves frames WITH detections —
  clean frames are skipped to save disk space.

Input:
  data/processed/batch_1/inference/detections_clean.csv
  data/processed/batch_1/*/frames/*.jpg

Output:
  data/processed/batch_1/inference/annotated/
    2622_Farwell_00043.jpg   <- original frame with boxes drawn
    2622_Farwell_00051.jpg
    ...

Usage:
  python scripts/visualize_detections.py
  python scripts/visualize_detections.py --batch batch_2
  python scripts/visualize_detections.py --raw   # use raw detections instead of clean
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

# ──────────────────────────────────────────────
# ✏️  EDIT THESE TO CONFIGURE
# ──────────────────────────────────────────────
PROJECT_ROOT = Path("/Users/nandiyang/Documents/robot-navigable-city")
DEFAULT_BATCH = "batch_1"

# Visual style
CLASS_COLORS = {
    "obstacle":          (0,  165, 255),   # orange (BGR)
    "path_discontinuity":(0,  0,   220),   # red    (BGR)
}
DEFAULT_COLOR  = (180, 180, 180)           # grey for unknown classes
BOX_THICKNESS  = 3
FONT_SCALE     = 0.7
FONT_THICKNESS = 2
JPEG_QUALITY   = 92
# ──────────────────────────────────────────────


def resolve_path(p) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def find_frame(batch: str, frame_file: str) -> Path | None:
    """Search all session frame folders for a given filename."""
    batch_processed = resolve_path(f"data/processed/{batch}")
    matches = list(batch_processed.glob(f"*/frames/{frame_file}"))
    return matches[0] if matches else None


def draw_detections(img, rows: pd.DataFrame) -> object:
    """Draw bounding boxes and labels on an image. Returns annotated image."""
    for _, row in rows.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        cls   = row["class_name"]
        conf  = row["confidence"]
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Label background
        label      = f"{cls}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       FONT_SCALE, FONT_THICKNESS)
        label_y = max(y1 - 8, th + 8)
        cv2.rectangle(img,
                      (x1, label_y - th - 6),
                      (x1 + tw + 6, label_y + 2),
                      color, -1)

        # Label text (white)
        cv2.putText(img, label,
                    (x1 + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, (255, 255, 255), FONT_THICKNESS,
                    cv2.LINE_AA)
    return img


def run(batch: str, use_raw: bool):
    print(f"\n{'='*60}")
    print(f"  Robot-Navigable City — Annotated Frame Visualization")
    print(f"{'='*60}")
    print(f"  Batch   : {batch}")
    suffix = "" if use_raw else "_clean"
    print(f"  Source  : detections{suffix}.csv")
    print(f"{'='*60}\n")

    # Load detections
    csv_path = resolve_path(
        f"data/processed/{batch}/inference/detections{suffix}.csv"
    )
    if not csv_path.exists():
        sys.exit(f"[ERROR] Not found: {csv_path}\n"
                 f"        Run {'inference' if use_raw else 'postprocess'}.py first.")

    df = pd.read_csv(csv_path)
    print(f"  Loaded  : {len(df)} detections across "
          f"{df['frame_file'].nunique()} unique frames")

    # Output directory
    out_dir = resolve_path(f"data/processed/{batch}/inference/annotated")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process each unique frame that has detections
    unique_frames = df["frame_file"].unique()
    saved   = 0
    missing = 0

    print(f"\n  Drawing bounding boxes...")
    for frame_file in unique_frames:
        frame_path = find_frame(batch, frame_file)
        if frame_path is None:
            missing += 1
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            missing += 1
            continue

        # Get all detections for this frame
        frame_rows = df[df["frame_file"] == frame_file]
        img_annotated = draw_detections(img, frame_rows)

        # Save
        out_path = out_dir / frame_file
        cv2.imwrite(str(out_path), img_annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        saved += 1

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    print(f"  Annotated frames saved : {saved}")
    if missing > 0:
        print(f"  Frames not found       : {missing}")
    print(f"  Output folder : {out_dir}")
    print(f"\n  These frames are used by the HTML demo.")
    print(f"  Next step: run demo.py to generate the interactive HTML page.")
    print(f"{'='*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes on detected frames."
    )
    parser.add_argument("--batch", default=DEFAULT_BATCH,
                        help=f"Batch name (default: {DEFAULT_BATCH})")
    parser.add_argument("--raw",   action="store_true",
                        help="Use raw detections instead of deduplicated clean version")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(batch=args.batch, use_raw=args.raw)
