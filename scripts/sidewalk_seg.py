"""
sidewalk_seg.py
===============
Stage 1: Sidewalk segmentation using Mask2Former (Mapillary Vistas pretrained)
Mapillary Vistas has 124 classes including: sidewalk, curb, curb cut, road, etc.
Much better for pedestrian-perspective sidewalk footage than Cityscapes.

Just run:  python sidewalk_seg.py
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import imageio.v2 as imageio

# ── PATHS ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser("~/Documents/robot-navigable-city")
INPUT_DIR = os.path.join(PROJECT_ROOT, "data/processed/round2_labeling/combined")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/processed/round2_labeling/segmentation")
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
SEGMAP_DIR = os.path.join(OUTPUT_DIR, "seg_maps")   # full seg maps for Stage 2a
GIF_PATH = os.path.join(OUTPUT_DIR, "sidewalk_segmentation.gif")
GIF_FULL_PATH = os.path.join(OUTPUT_DIR, "sidewalk_segmentation_full_palette.gif")

# ── SETTINGS ──────────────────────────────────────────────────────────
MODEL_NAME = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
SIDEWALK_ALPHA = 0.5
FULL_ALPHA = 0.45
GIF_FPS = 2
GIF_MAX_WIDTH = 640
SAVE_MASKS = True
MAKE_SIDEWALK_GIF = True
MAKE_FULL_GIF = True
TEST_LIMIT = None           # set to None to process ALL frames

# ── Mapillary Vistas classes we care about ────────────────────────────
# Colors for the classes we want to highlight (RGBA for overlay)
HIGHLIGHT_COLORS = {
    "sidewalk":        (0, 255, 128, 180),    # bright green
    "curb":            (0, 200, 255, 150),    # cyan
    "curb cut":        (255, 200, 0, 160),    # yellow — accessibility feature!
    "utility pole":    (255, 100, 255, 150),  # magenta
    "pole":            (200, 80, 200, 120),   # lighter magenta
    "vegetation":      (40, 180, 40, 120),    # forest green
    "terrain":         (120, 200, 120, 100),  # light green (grass/landscape)
    "trash can":       (255, 128, 0, 170),    # orange
    "street light":    (180, 100, 220, 130),  # purple
}

# Colors for full palette mode (key classes only, rest gets gray)
FULL_PALETTE_COLORS = {
    "sidewalk":    (0, 255, 128),       # bright green
    "curb":        (0, 200, 255),       # cyan
    "curb cut":    (255, 200, 0),       # yellow
    "road":        (128, 64, 128),      # purple
    "car":         (255, 200, 0),       # gold
    "truck":       (200, 150, 0),       # darker gold
    "bus":         (180, 120, 0),       # brown-gold
    "vegetation":  (40, 90, 30),        # dark green
    "terrain":     (152, 200, 152),     # light green
    "building":    (70, 70, 70),        # dark gray
    "person":      (220, 20, 60),       # red
    "pole":        (200, 80, 200),      # magenta
    "utility pole": (255, 100, 255),    # bright magenta
    "street light": (180, 100, 220),    # purple
    "trash can":   (255, 128, 0),       # orange
    "fence":       (190, 153, 153),     # pinkish gray
    "wall":        (102, 102, 156),     # blue-gray
    "sky":         (70, 130, 180),      # steel blue
    "bicycle":     (119, 11, 32),       # dark red
    "motorcycle":  (0, 0, 230),         # blue
}

DEFAULT_COLOR = (100, 100, 100)  # muted gray for everything else


# ── MODEL ─────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading model: {MODEL_NAME}")
    print("  (This is a larger model — first download may take a few minutes)")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = model.to(device).eval()
    print(f"Model loaded on {device}")
    if device == "mps":
        print("  (Apple Silicon GPU — expect ~1-3s per frame)")

    # Build class name → id mapping
    id2label = model.config.id2label
    label2id = {v.lower(): int(k) for k, v in id2label.items()}

    print(f"  Model has {len(id2label)} classes")

    # Find our key classes
    sidewalk_classes = {}
    for name in HIGHLIGHT_COLORS:
        matches = [k for k, v in label2id.items() if name in k]
        if matches:
            for m in matches:
                sidewalk_classes[m] = label2id[m]
                print(f"  Found class: '{m}' → id {label2id[m]}")
        else:
            print(f"  WARNING: class '{name}' not found in model")

    return processor, model, device, id2label, label2id, sidewalk_classes


# ── SEGMENTATION ──────────────────────────────────────────────────────
def segment_frame(image, processor, model, device):
    """Run Mask2Former semantic segmentation on a single image."""
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process to get semantic map
    seg_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0].cpu().numpy()

    return seg_map


def create_sidewalk_overlay(image, seg_map, sidewalk_classes):
    """
    Overlay ONLY sidewalk-related classes on the original image.
    sidewalk = green, curb = cyan, curb cut = yellow
    """
    overlay = image.convert("RGBA").copy()
    color_layer = np.zeros((*seg_map.shape, 4), dtype=np.uint8)

    for class_name, class_id in sidewalk_classes.items():
        mask = (seg_map == class_id)
        if mask.any():
            # Find matching highlight color
            for key, color in HIGHLIGHT_COLORS.items():
                if key in class_name:
                    color_layer[mask] = color
                    break

    color_img = Image.fromarray(color_layer, "RGBA")
    overlay = Image.alpha_composite(overlay, color_img)
    return overlay.convert("RGB")


def create_full_overlay(image, seg_map, id2label, alpha=FULL_ALPHA):
    """All classes colored — dramatic visualization."""
    h, w = seg_map.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in np.unique(seg_map):
        class_name = id2label.get(class_id, "unknown").lower()
        mask = (seg_map == class_id)

        # Find color
        color = DEFAULT_COLOR
        for key, c in FULL_PALETTE_COLORS.items():
            if key in class_name:
                color = c
                break

        color_mask[mask] = color

    color_img = Image.fromarray(color_mask)
    blended = Image.blend(image.convert("RGB"), color_img, alpha)
    return blended


# ── MAIN ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_MASKS:
        os.makedirs(MASK_DIR, exist_ok=True)
        os.makedirs(SEGMAP_DIR, exist_ok=True)

    # Find frames
    frame_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        frame_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    frame_paths.sort()

    if not frame_paths:
        print(f"ERROR: No images found in {INPUT_DIR}")
        return

    # Apply test limit
    if TEST_LIMIT:
        frame_paths = frame_paths[:TEST_LIMIT]
        print(f"TEST MODE: processing first {TEST_LIMIT} frames")

    print(f"Processing {len(frame_paths)} frames from {INPUT_DIR}")
    print(f"Output → {OUTPUT_DIR}\n")

    # Load model
    processor, model, device, id2label, label2id, sidewalk_classes = load_model()
    print()

    # Process frames
    sidewalk_gif_frames = []
    full_gif_frames = []
    stats = []

    for i, fpath in enumerate(frame_paths):
        fname = os.path.basename(fpath)
        print(f"  [{i+1}/{len(frame_paths)}] {fname}", end="")

        image = Image.open(fpath).convert("RGB")
        seg_map = segment_frame(image, processor, model, device)

        # Sidewalk-only overlay
        sidewalk_overlay = create_sidewalk_overlay(image, seg_map, sidewalk_classes)
        sidewalk_overlay.save(os.path.join(OUTPUT_DIR, f"seg_{fname}"))

        # Full-palette overlay
        if MAKE_FULL_GIF:
            full_overlay = create_full_overlay(image, seg_map, id2label)
            full_overlay.save(os.path.join(OUTPUT_DIR, f"full_{fname}"))

        # Binary sidewalk mask (combining sidewalk + curb cut)
        if SAVE_MASKS:
            sw_mask = np.zeros(seg_map.shape, dtype=np.uint8)
            for class_name, class_id in sidewalk_classes.items():
                sw_mask[seg_map == class_id] = 255
            mask_img = Image.fromarray(sw_mask)
            mask_img.save(os.path.join(MASK_DIR, f"mask_{fname}"))

            # Save full segmentation map for Stage 2a (width + blockage analysis)
            base_name = os.path.splitext(fname)[0]
            np.savez_compressed(
                os.path.join(SEGMAP_DIR, f"{base_name}.npz"),
                seg_map=seg_map.astype(np.int16)
            )

        # Stats — show what classes were found in this frame
        detected = []
        for class_name, class_id in sidewalk_classes.items():
            pct = (seg_map == class_id).sum() / seg_map.size
            if pct > 0.005:  # only show if > 0.5% of image
                detected.append(f"{class_name}:{pct:.0%}")
        fraction = sum(
            (seg_map == cid).sum() for cid in sidewalk_classes.values()
        ) / seg_map.size
        stats.append(fraction)
        det_str = ", ".join(detected) if detected else "none"
        print(f"  — [{det_str}]")

        # GIF frames
        if MAKE_SIDEWALK_GIF:
            gif_frame = sidewalk_overlay.copy()
            gif_frame.thumbnail((GIF_MAX_WIDTH, GIF_MAX_WIDTH), Image.LANCZOS)
            sidewalk_gif_frames.append(np.array(gif_frame))

        if MAKE_FULL_GIF:
            gif_frame = full_overlay.copy()
            gif_frame.thumbnail((GIF_MAX_WIDTH, GIF_MAX_WIDTH), Image.LANCZOS)
            full_gif_frames.append(np.array(gif_frame))

    # Summary
    print(f"\n{'='*50}")
    print(f"Processed {len(frame_paths)} frames")
    print(f"Sidewalk coverage: mean={np.mean(stats):.1%}, "
          f"min={np.min(stats):.1%}, max={np.max(stats):.1%}")

    # Save GIFs
    if MAKE_SIDEWALK_GIF and sidewalk_gif_frames:
        print(f"\nSaving sidewalk GIF → {GIF_PATH}")
        imageio.mimsave(GIF_PATH, sidewalk_gif_frames, fps=GIF_FPS, loop=0)
        size_mb = os.path.getsize(GIF_PATH) / (1024 * 1024)
        print(f"  Done ({len(sidewalk_gif_frames)} frames, {size_mb:.1f} MB)")

    if MAKE_FULL_GIF and full_gif_frames:
        print(f"\nSaving full-palette GIF → {GIF_FULL_PATH}")
        imageio.mimsave(GIF_FULL_PATH, full_gif_frames, fps=GIF_FPS, loop=0)
        size_mb = os.path.getsize(GIF_FULL_PATH) / (1024 * 1024)
        print(f"  Done ({len(full_gif_frames)} frames, {size_mb:.1f} MB)")

    print(f"\n{'='*50}")
    print(f"Sidewalk overlays:     {OUTPUT_DIR}/seg_*.jpg")
    if MAKE_FULL_GIF:
        print(f"Full palette:          {OUTPUT_DIR}/full_*.jpg")
    if SAVE_MASKS:
        print(f"Binary masks:          {MASK_DIR}/mask_*.jpg")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
