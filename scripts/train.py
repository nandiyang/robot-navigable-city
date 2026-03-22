"""
train.py
========
Phase 1 — YOLOv8 Model Training
Robot-Navigable City Project

Classes:
  0: obstacle          — objects blocking the sidewalk path
  1: path_discontinuity — vertical drops, uplifts, sudden grade changes

Usage:
  python scripts/train.py           # uses defaults below
  python scripts/train.py --epochs 100 --model yolov8s.pt

Output:
  runs/detect/train_v1/
    weights/
      best.pt       <- use this for inference
      last.pt
    results.csv
    confusion_matrix.png
    PR_curve.png
    val_batch0_pred.jpg  <- visual check of predictions
"""

import argparse
import shutil
import sys
from pathlib import Path

# ──────────────────────────────────────────────
# ✏️  EDIT THESE TO CONFIGURE
# ──────────────────────────────────────────────
PROJECT_ROOT = Path("/Users/nandiyang/Documents/robot-navigable-city")
DATASET_PATH = PROJECT_ROOT / "data/datasets/robot-navigable-city.v1i.yolov8"
RUNS_DIR     = PROJECT_ROOT / "runs"

# Training config
DEFAULT_MODEL  = "yolov8n.pt"    # nano — fast, good for first run
                                 # upgrade to yolov8s.pt after confirming pipeline works
DEFAULT_EPOCHS = 100             # good starting point for ~476 labeled images
DEFAULT_IMGSZ  = 640             # matches Roboflow export size
DEFAULT_BATCH  = 32              # reduce to 8 if you get out-of-memory errors
DEFAULT_NAME   = "train_v1"      # output folder name under runs/detect/
# ──────────────────────────────────────────────


def fix_yaml_paths(dataset_path: Path) -> Path:
    """
    Roboflow exports data.yaml with relative paths (../train/images).
    YOLO needs absolute paths to work reliably regardless of working directory.
    Creates a fixed copy: data_fixed.yaml
    """
    import yaml

    yaml_path       = dataset_path / "data.yaml"
    yaml_fixed_path = dataset_path / "data_fixed.yaml"

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Replace relative paths with absolute
    data["train"] = str(dataset_path / "train" / "images")
    data["val"]   = str(dataset_path / "valid" / "images")
    data["test"]  = str(dataset_path / "test"  / "images")

    with open(yaml_fixed_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"  Fixed data.yaml saved → {yaml_fixed_path}")
    print(f"  train : {data['train']}")
    print(f"  val   : {data['val']}")
    print(f"  test  : {data['test']}")
    print(f"  nc    : {data['nc']}")
    print(f"  names : {data['names']}")

    return yaml_fixed_path


def train(model_name: str, epochs: int, imgsz: int, batch: int, run_name: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERROR] ultralytics not found. Run: pip install ultralytics")

    if not DATASET_PATH.exists():
        sys.exit(f"[ERROR] Dataset not found: {DATASET_PATH}\n"
                 f"        Download from Roboflow and place in data/datasets/")

    print(f"\n{'='*60}")
    print(f"  Robot-Navigable City — YOLOv8 Training")
    print(f"{'='*60}")
    print(f"  Model   : {model_name}")
    print(f"  Epochs  : {epochs}")
    print(f"  ImgSz   : {imgsz}")
    print(f"  Batch   : {batch}")
    print(f"  Run     : {run_name}")
    print(f"{'='*60}\n")

    # Fix yaml paths
    print("Fixing data.yaml paths...")
    yaml_path = fix_yaml_paths(DATASET_PATH)

    # Load model
    print(f"\nLoading model: {model_name}")
    model = YOLO(model_name)

    # Train
    print(f"\nStarting training...")
    results = model.train(
        data    = str(yaml_path),
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        name    = run_name,
        project = str(RUNS_DIR / "detect"),
        exist_ok= True,           # overwrite if run_name already exists
        patience= 20,             # early stopping if no improvement for 20 epochs
        save    = True,
        plots   = True,           # saves training curves and confusion matrix
        verbose = True,
        device  = "mps",
    )

    # Results
    run_dir   = RUNS_DIR / "detect" / run_name
    best_pt   = run_dir / "weights" / "best.pt"

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best weights : {best_pt}")
    print(f"  Results dir  : {run_dir}")
    print(f"\n  Next step: run inference.py with best.pt")
    print(f"{'='*60}\n")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on robot-navigable-city dataset."
    )
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help=f"YOLO model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--imgsz",  type=int, default=DEFAULT_IMGSZ,
                        help=f"Image size (default: {DEFAULT_IMGSZ})")
    parser.add_argument("--batch",  type=int, default=DEFAULT_BATCH,
                        help=f"Batch size (default: {DEFAULT_BATCH})")
    parser.add_argument("--name",   default=DEFAULT_NAME,
                        help=f"Run name (default: {DEFAULT_NAME})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name = args.model,
        epochs     = args.epochs,
        imgsz      = args.imgsz,
        batch      = args.batch,
        run_name   = args.name,
    )
