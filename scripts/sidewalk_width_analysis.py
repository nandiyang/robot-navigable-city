"""
sidewalk_width_analysis.py
==========================
Stage 2a: Perspective-corrected corridor width & blockage analysis.

Uses:
  - Full segmentation maps from sidewalk_seg.py (Stage 1)
  - Depth Anything V2 (Metric Outdoor) for perspective correction
  - Rule-based logic to detect narrow corridors and object blockages

Output:
  - White corridor boundary + green/yellow/red heatmap overlay per frame
  - GIF animation
  - CSV with per-frame passability scores

Just run:  python sidewalk_width_analysis.py

Prerequisites: run sidewalk_seg.py first (with SAVE_MASKS=True)
"""

import os
import glob
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch.nn.functional as F
import imageio.v2 as imageio

# ── PATHS ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser("~/Documents/robot-navigable-city")
INPUT_DIR = os.path.join(PROJECT_ROOT, "data/processed/round2_labeling/combined")
SEG_DIR = os.path.join(PROJECT_ROOT, "data/processed/round2_labeling/segmentation")
SEGMAP_DIR = os.path.join(SEG_DIR, "seg_maps")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/processed/round2_labeling/width_analysis")
GIF_PATH = os.path.join(OUTPUT_DIR, "width_analysis.gif")

# ── SETTINGS ──────────────────────────────────────────────────────────
TEST_LIMIT = None
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"
GIF_FPS = 2
GIF_MAX_WIDTH = 640

# ADA wheelchair minimum clear width = 91.5 cm (36 inches)
# Comfortable passing width = 150 cm (60 inches)
ADA_MIN_WIDTH_CM = 91.5
COMFORTABLE_WIDTH_CM = 150.0

# Classes that count as "obstruction" when overlapping the sidewalk
# These are infrastructure barriers (not temporary like people/animals)
OBSTRUCTION_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bicycle",          # vehicles
    "pole", "utility pole", "street light",                   # poles
    "trash can", "bench", "fire hydrant", "mailbox",          # street furniture
    "vegetation", "potted plant",                             # overgrown plants
    "fence", "wall", "barrier", "guard rail",                 # barriers
    "railing", "bollard", "rail",                             # railings & bollards
    "traffic sign", "sign", "signage",                        # sign structures
    "construction", "jersey barrier",                         # construction
    "static",                                                 # Mapillary catch-all for fixed objects
}

# Classes to EXCLUDE from obstruction (temporary/moving or flush with surface)
EXCLUDE_CLASSES = {
    "person", "rider", "dog", "animal", "bird",              # temporary
    "manhole", "utility hole", "drain", "catch basin",        # flush covers
    "water valve", "ground",                                  # ground-level
}

# Sidewalk classes — anything that is walkable surface
SIDEWALK_CLASSES = {
    "sidewalk", "curb cut",                                   # primary
    "manhole", "utility hole", "drain", "catch basin",        # flush covers
    "water valve", "ground",                                  # ground-level surface
}

# Heatmap colors (RGBA)
COLOR_COMFORTABLE = (0, 200, 80, 140)     # green
COLOR_CONSTRAINED = (255, 200, 0, 160)    # yellow
COLOR_BLOCKED = (255, 40, 40, 180)        # red
COLOR_BOUNDARY = (255, 255, 255, 200)     # white


# ── DEPTH MODEL ───────────────────────────────────────────────────────
def load_depth_model():
    print(f"Loading depth model: {DEPTH_MODEL}")
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = model.to(device).eval()
    print(f"Depth model loaded on {device}")
    return processor, model, device


def estimate_depth(image, processor, model, device):
    """
    Returns depth map in meters (H, W).
    Metric Outdoor model outputs actual distance in meters.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth

    # Resize to original image size
    depth = F.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    return depth


# ── SEGMENTATION MAP LOADING ─────────────────────────────────────────
def load_seg_map(fname):
    """Load the full segmentation map saved by sidewalk_seg.py"""
    base_name = os.path.splitext(fname)[0]
    npz_path = os.path.join(SEGMAP_DIR, f"{base_name}.npz")
    if os.path.exists(npz_path):
        return np.load(npz_path)["seg_map"]
    return None


def get_class_masks(seg_map, id2label):
    """
    From the full seg map, extract:
    - sidewalk_mask: bool array, True where sidewalk/curb cut
    - obstruction_mask: bool array, True where infrastructure obstacles
    - obstruction_labels: dict mapping (y,x) regions to class names
    """
    H, W = seg_map.shape
    sidewalk_mask = np.zeros((H, W), dtype=bool)
    obstruction_mask = np.zeros((H, W), dtype=bool)
    obstruction_info = {}  # class_id -> class_name for detected obstructions

    for class_id in np.unique(seg_map):
        class_name = id2label.get(int(class_id), "unknown").lower()
        pixel_mask = (seg_map == class_id)

        # Check if this class is a sidewalk class
        for sw_name in SIDEWALK_CLASSES:
            if sw_name in class_name:
                sidewalk_mask |= pixel_mask
                break

        # Check if this class is an obstruction
        for obs_name in OBSTRUCTION_CLASSES:
            if obs_name in class_name:
                # Check it's not excluded
                excluded = False
                for exc_name in EXCLUDE_CLASSES:
                    if exc_name in class_name:
                        excluded = True
                        break
                if not excluded:
                    obstruction_mask |= pixel_mask
                    if pixel_mask.any():
                        obstruction_info[int(class_id)] = class_name
                break

    return sidewalk_mask, obstruction_mask, obstruction_info


# ── WIDTH ANALYSIS ────────────────────────────────────────────────────
def isolate_camera_side_sidewalk(sidewalk_mask):
    """
    Keep only the sidewalk on the camera's side of the street.

    Strategy: In egocentric footage, "our" sidewalk is the largest
    connected component that touches the bottom portion of the image.
    The far-side sidewalk is smaller and sits in the upper half.

    Uses connected components to find the dominant sidewalk region.
    """
    from scipy import ndimage

    H, W = sidewalk_mask.shape

    # Label connected components
    labeled, num_features = ndimage.label(sidewalk_mask)
    if num_features == 0:
        return sidewalk_mask

    # Score each component: prefer ones that are large AND touch bottom half
    bottom_half_start = H // 2
    best_label = 0
    best_score = 0

    for label_id in range(1, num_features + 1):
        component = (labeled == label_id)
        total_pixels = component.sum()
        bottom_pixels = component[bottom_half_start:, :].sum()

        # Score = total size weighted by how much is in the bottom half
        # Components entirely in the top half (far sidewalk) score low
        if total_pixels > 0:
            bottom_fraction = bottom_pixels / total_pixels
            score = total_pixels * (0.3 + 0.7 * bottom_fraction)

            if score > best_score:
                best_score = score
                best_label = label_id

    if best_label > 0:
        return (labeled == best_label)
    return sidewalk_mask


def compute_corridor_width(sidewalk_mask, obstruction_mask, depth_map):
    """
    For each row, compute the effective corridor width in centimeters.

    Method:
    1. Isolate camera-side sidewalk only
    2. Subtract any obstruction pixels that overlap the sidewalk
    3. Find the widest continuous run of clear sidewalk
    4. Convert pixel width to real-world cm using depth

    The key insight: pixel_width_cm = pixel_width_px * (depth_m / focal_length_px)
    We approximate focal_length_px from the image width assuming ~70° horizontal FOV
    (typical for iPhone wide camera).
    """
    H, W = sidewalk_mask.shape

    # Step 1: Keep only camera-side sidewalk
    sidewalk_mask = isolate_camera_side_sidewalk(sidewalk_mask)

    # Effective sidewalk = sidewalk minus obstructions
    clear_sidewalk = sidewalk_mask & ~obstruction_mask
    blocked_sidewalk = sidewalk_mask & obstruction_mask

    # Approximate focal length in pixels (iPhone ~70° horizontal FOV)
    fov_deg = 70.0
    focal_px = (W / 2) / np.tan(np.radians(fov_deg / 2))

    widths_cm = np.zeros(H, dtype=np.float32)
    widths_px = np.zeros(H, dtype=np.int32)
    left_edges = np.zeros(H, dtype=np.int32)
    right_edges = np.zeros(H, dtype=np.int32)
    row_depths = np.zeros(H, dtype=np.float32)

    for y in range(H):
        row = clear_sidewalk[y].astype(np.uint8)
        if row.sum() == 0:
            continue

        # Find continuous runs
        diffs = np.diff(np.concatenate(([0], row, [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        run_lengths = ends - starts

        # Widest run
        best = np.argmax(run_lengths)
        px_width = run_lengths[best]
        left = starts[best]
        right = ends[best]

        widths_px[y] = px_width
        left_edges[y] = left
        right_edges[y] = right

        # Get median depth along this row's sidewalk pixels
        row_depth_vals = depth_map[y, left:right]
        if len(row_depth_vals) > 0:
            median_depth = np.median(row_depth_vals)
            if median_depth > 0.1:  # sanity check
                # Convert pixel width to real-world width
                width_m = (px_width * median_depth) / focal_px
                widths_cm[y] = width_m * 100  # to cm
                row_depths[y] = median_depth

    return widths_cm, widths_px, left_edges, right_edges, row_depths, blocked_sidewalk


def detect_full_blockages(sidewalk_mask, obstruction_mask, seg_map, id2label,
                          widths_px, look_ahead=80):
    """
    Detect rows where the sidewalk is COMPLETELY blocked — i.e., sidewalk
    exists above and below, but disappears in between because an object
    fully covers it.

    Returns:
        blockage_regions: list of (y_start, y_end, blocking_class_name)
        full_block_mask: bool array (H, W) marking fully blocked zones
    """
    H, W = sidewalk_mask.shape
    full_block_mask = np.zeros((H, W), dtype=bool)
    blockage_regions = []

    # Find rows that have sidewalk
    has_sidewalk = np.array([widths_px[y] > 0 for y in range(H)])

    # Only scan within the LOWER portion of the sidewalk zone.
    # The top of the sidewalk (furthest away) naturally disappears into
    # perspective — that's not a blockage. Real blockages happen in the
    # middle and bottom of the image (closer to the camera).
    sidewalk_rows = np.where(has_sidewalk)[0]
    if len(sidewalk_rows) == 0:
        return blockage_regions, full_block_mask

    # Don't scan above the midpoint of the sidewalk's vertical range.
    # This eliminates false positives at the vanishing point.
    sidewalk_mid = (sidewalk_rows.min() + sidewalk_rows.max()) // 2
    scan_top = sidewalk_mid  # start scanning from the middle, not the top
    scan_bottom = min(H, sidewalk_rows.max() + 20)

    # Scan for gaps in sidewalk within the sidewalk zone only.
    # In egocentric view: bottom of image = close, top = far.
    # A blockage means: sidewalk exists BELOW the gap (between you and the object).
    in_gap = False
    gap_start = 0

    for y in range(scan_top, scan_bottom):
        if has_sidewalk[y]:
            if in_gap and (y - gap_start) >= 10:  # minimum gap size
                # Is there sidewalk BELOW (closer to camera)?
                # "Below" = higher y values in image coordinates
                has_below = has_sidewalk[y:min(H, y + look_ahead)].any()

                if has_below:
                    # This is a full blockage. Find what's blocking it.
                    gap_seg = seg_map[gap_start:y, :]
                    blocking_classes = []
                    for class_id in np.unique(gap_seg):
                        name = id2label.get(int(class_id), "unknown").lower()
                        # Check if it's an obstruction class
                        for obs_name in OBSTRUCTION_CLASSES:
                            if obs_name in name:
                                excluded = any(
                                    exc in name for exc in EXCLUDE_CLASSES
                                )
                                if not excluded:
                                    pct = (gap_seg == class_id).sum() / gap_seg.size
                                    if pct > 0.05:  # at least 5% of gap area
                                        blocking_classes.append(name)
                                break

                    if blocking_classes:
                        blockage_regions.append(
                            (gap_start, y, ", ".join(set(blocking_classes)))
                        )
                        full_block_mask[gap_start:y, :] = True

            in_gap = False
        else:
            if not in_gap:
                gap_start = y
                in_gap = True

    # ── DEAD-END DETECTION ────────────────────────────────────────
    # If the sidewalk exists in the bottom portion but terminates
    # (no sidewalk above the topmost sidewalk row), and there's an
    # obstruction right where it ends — that's a dead-end blockage.
    #
    # Logic: find the topmost row of sidewalk. Look at the region
    # immediately above it (where the sidewalk should continue but
    # doesn't). If obstruction classes dominate that region, flag it.
    top_sw_row = sidewalk_rows.min()
    bottom_sw_row = sidewalk_rows.max()
    sw_vertical_span = bottom_sw_row - top_sw_row

    # Only flag a dead-end if:
    # 1. The sidewalk's top edge is in the BOTTOM HALF of the image
    #    (if it reaches the upper half, it's just perspective fade)
    # 2. There's a substantial vertical span of sidewalk visible
    # 3. The region above is DOMINATED by a single obstruction (>30%)
    #    (not just scattered wall/fence/veg which is normal scenery)
    if sw_vertical_span > 80 and top_sw_row > H * 0.50:
        # Look at a narrow zone just above where sidewalk ends
        check_height = min(60, top_sw_row)
        check_top = top_sw_row - check_height

        # Only check the horizontal range where sidewalk was
        # (not the full image width — that catches side walls/fences)
        sw_cols = np.where(sidewalk_mask[top_sw_row:top_sw_row + 30, :].any(axis=0))[0]
        if len(sw_cols) > 10:
            col_left = sw_cols.min()
            col_right = sw_cols.max()
            check_region = seg_map[check_top:top_sw_row, col_left:col_right]

            # What's blocking the path?
            blocking_classes = []
            for class_id in np.unique(check_region):
                name = id2label.get(int(class_id), "unknown").lower()
                for obs_name in OBSTRUCTION_CLASSES:
                    if obs_name in name:
                        excluded = any(exc in name for exc in EXCLUDE_CLASSES)
                        if not excluded:
                            pct = (check_region == class_id).sum() / check_region.size
                            if pct > 0.30:  # must dominate the check region
                                blocking_classes.append(name)
                        break

            if blocking_classes:
                blockage_regions.append(
                    (check_top, top_sw_row,
                     "DEAD END: " + ", ".join(set(blocking_classes)))
                )
                full_block_mask[check_top:top_sw_row, col_left:col_right] = True

    return blockage_regions, full_block_mask


def classify_rows(widths_cm):
    """
    Classify each row: 2=comfortable, 1=constrained, 0=blocked/missing
    """
    labels = np.zeros(len(widths_cm), dtype=np.uint8)
    labels[widths_cm >= COMFORTABLE_WIDTH_CM] = 2
    labels[(widths_cm >= ADA_MIN_WIDTH_CM) &
           (widths_cm < COMFORTABLE_WIDTH_CM)] = 1
    # 0 stays for below ADA minimum or no sidewalk
    return labels


def smooth_labels(labels, widths_cm, min_run=25):
    """
    Remove short yellow/red segments that are likely perspective noise.
    If a constrained/blocked run is shorter than min_run rows,
    promote it back to comfortable (green).

    Real passability issues persist over many consecutive rows.
    Perspective artifacts create tiny slivers at the vanishing point.
    """
    smoothed = labels.copy()
    in_problem = False
    problem_start = 0

    for y in range(len(smoothed)):
        is_problem = (smoothed[y] <= 1) and (widths_cm[y] > 0)
        if is_problem:
            if not in_problem:
                problem_start = y
                in_problem = True
        else:
            if in_problem:
                run_length = y - problem_start
                if run_length < min_run:
                    # Too short — promote back to comfortable
                    smoothed[problem_start:y] = 2
                in_problem = False

    # Handle edge case at end of array
    if in_problem:
        run_length = len(smoothed) - problem_start
        if run_length < min_run:
            smoothed[problem_start:] = 2

    return smoothed


def find_pinch_points(widths_cm, labels, min_run=15):
    """Find continuous regions of constrained/blocked sidewalk."""
    pinch_points = []
    current_start = None
    current_severity = None

    for y in range(len(labels)):
        if labels[y] <= 1 and widths_cm[y] > 0:
            severity = "blocked" if labels[y] == 0 else "constrained"
            if current_start is None:
                current_start = y
                current_severity = severity
            elif severity != current_severity:
                if y - current_start >= min_run:
                    pinch_points.append((current_start, y, current_severity))
                current_start = y
                current_severity = severity
        else:
            if current_start is not None and y - current_start >= min_run:
                pinch_points.append((current_start, y, current_severity))
            current_start = None
            current_severity = None

    if current_start is not None and len(labels) - current_start >= min_run:
        pinch_points.append((current_start, len(labels), current_severity))

    return pinch_points


# ── VISUALIZATION ─────────────────────────────────────────────────────
def create_overlay(image, sidewalk_mask, blocked_sidewalk,
                   widths_cm, widths_px, labels, left_edges, right_edges,
                   pinch_points, obstruction_info,
                   blockage_regions=None, full_block_mask=None,
                   partial_blockages=None):
    """
    White corridor boundary + green/yellow/red heatmap on sidewalk.
    Blocked areas shown in red with obstruction labels.
    """
    overlay = image.convert("RGBA").copy()
    H, W = sidewalk_mask.shape

    # Heatmap layer
    heatmap = np.zeros((H, W, 4), dtype=np.uint8)

    for y in range(H):
        if widths_px[y] == 0:
            continue
        left = left_edges[y]
        right = right_edges[y]

        if labels[y] == 2:
            heatmap[y, left:right] = COLOR_COMFORTABLE
        elif labels[y] == 1:
            heatmap[y, left:right] = COLOR_CONSTRAINED
        else:
            heatmap[y, left:right] = COLOR_BLOCKED

    # Mark blocked sidewalk areas (obstruction overlap) in red
    blocked_layer = np.zeros((H, W, 4), dtype=np.uint8)
    blocked_layer[blocked_sidewalk] = (255, 60, 60, 160)
    heatmap = np.maximum(heatmap, blocked_layer)

    heatmap_img = Image.fromarray(heatmap, "RGBA")
    overlay = Image.alpha_composite(overlay, heatmap_img)

    # Draw corridor boundary (white lines)
    draw = ImageDraw.Draw(overlay)
    prev_left = None
    prev_right = None
    step = 2
    for y in range(0, H, step):
        if widths_px[y] == 0:
            prev_left = None
            prev_right = None
            continue
        left = int(left_edges[y])
        right = int(right_edges[y])
        if prev_left is not None:
            draw.line([(prev_left, y - step), (left, y)],
                      fill=COLOR_BOUNDARY, width=2)
            draw.line([(prev_right, y - step), (right, y)],
                      fill=COLOR_BOUNDARY, width=2)
        prev_left = left
        prev_right = right

    # Label pinch points
    for y_start, y_end, severity in pinch_points:
        y_mid = (y_start + y_end) // 2

        if widths_px[y_mid] > 0:
            x_mid = (left_edges[y_mid] + right_edges[y_mid]) // 2
        else:
            # Find nearest row with sidewalk
            for dy in range(1, 50):
                for yy in [y_mid - dy, y_mid + dy]:
                    if 0 <= yy < H and widths_px[yy] > 0:
                        x_mid = (left_edges[yy] + right_edges[yy]) // 2
                        break
                else:
                    continue
                break
            else:
                x_mid = W // 2

        # Get width at narrowest point in this region
        region_widths = widths_cm[y_start:y_end]
        valid_widths = region_widths[region_widths > 0]

        if severity == "blocked":
            color = (255, 40, 40, 255)
            label = "BLOCKED"
        else:
            if len(valid_widths) > 0:
                min_w = valid_widths.min()
                label = f"{min_w:.0f}cm"
            else:
                label = "NARROW"
            color = (255, 200, 0, 255)

        # Background box for label
        text_w = len(label) * 8 + 16
        draw.rectangle(
            [x_mid - text_w // 2, y_mid - 12,
             x_mid + text_w // 2, y_mid + 12],
            fill=(0, 0, 0, 180)
        )
        draw.text((x_mid - text_w // 2 + 8, y_mid - 8),
                  label, fill=color)

    # Mark full blockages (sidewalk completely eliminated by object)
    if blockage_regions and full_block_mask is not None:
        # Red stripe across the full blockage zone
        block_layer = np.zeros((H, W, 4), dtype=np.uint8)
        block_layer[full_block_mask] = (255, 0, 0, 120)
        block_img = Image.fromarray(block_layer, "RGBA")
        overlay = Image.alpha_composite(overlay, block_img)
        draw = ImageDraw.Draw(overlay)  # refresh draw after composite

        for y_start, y_end, blocking_class in blockage_regions:
            y_mid = (y_start + y_end) // 2
            label = f"BLOCKED by {blocking_class}"
            text_w = len(label) * 7 + 16
            x_mid = W // 2
            draw.rectangle(
                [x_mid - text_w // 2, y_mid - 12,
                 x_mid + text_w // 2, y_mid + 12],
                fill=(0, 0, 0, 200)
            )
            draw.text((x_mid - text_w // 2 + 8, y_mid - 8),
                      label, fill=(255, 60, 60, 255))

    # Mark partial blockages (object ON sidewalk but doesn't eliminate it)
    if partial_blockages:
        for y_start, y_end, blocking_class in partial_blockages:
            y_mid = (y_start + y_end) // 2
            # Find x position in the blocked area
            blocked_cols = np.where(blocked_sidewalk[y_mid, :])[0]
            if len(blocked_cols) > 0:
                x_mid = int(np.mean(blocked_cols))
            else:
                x_mid = W // 2
            label = f"INTRUDING: {blocking_class}"
            text_w = len(label) * 7 + 16
            draw.rectangle(
                [x_mid - text_w // 2, y_mid - 12,
                 x_mid + text_w // 2, y_mid + 12],
                fill=(0, 0, 0, 200)
            )
            draw.text((x_mid - text_w // 2 + 8, y_mid - 8),
                      label, fill=(255, 165, 0, 255))  # orange

    # Legend in top-left corner
    legend_y = 10
    for label_text, color in [
        ("Comfortable (>150cm)", COLOR_COMFORTABLE),
        ("Constrained (91-150cm)", COLOR_CONSTRAINED),
        ("Below ADA min (<91cm)", COLOR_BLOCKED),
    ]:
        draw.rectangle([10, legend_y, 26, legend_y + 14],
                       fill=color)
        draw.text((32, legend_y), label_text,
                  fill=(255, 255, 255, 255))
        legend_y += 18

    return overlay.convert("RGB")


def compute_scene_score(widths_cm, labels, H):
    """Passability score 0-100 for the frame."""
    # Focus on bottom 2/3 (closer = more relevant for navigation)
    start = H // 3
    relevant_cm = widths_cm[start:]
    relevant_labels = labels[start:]

    has_sidewalk = relevant_cm > 0
    if has_sidewalk.sum() == 0:
        return 0, "no sidewalk"

    comfortable_frac = (relevant_labels[has_sidewalk] == 2).mean()
    constrained_frac = (relevant_labels[has_sidewalk] == 1).mean()
    blocked_frac = (relevant_labels[has_sidewalk] == 0).mean()

    score = int(comfortable_frac * 100 - constrained_frac * 20 - blocked_frac * 50)
    score = max(0, min(100, score))

    if score >= 70:
        return score, "comfortable"
    elif score >= 40:
        return score, "constrained"
    else:
        return score, "failure"


# ── MAIN ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check that Stage 1 was run
    if not os.path.exists(SEGMAP_DIR):
        print(f"ERROR: No seg_maps found at {SEGMAP_DIR}")
        print(f"  Run sidewalk_seg.py first with SAVE_MASKS=True")
        return

    # Find frames
    frame_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        frame_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    frame_paths.sort()

    if TEST_LIMIT:
        frame_paths = frame_paths[:TEST_LIMIT]

    if not frame_paths:
        print(f"ERROR: No images found in {INPUT_DIR}")
        return

    print(f"Processing {len(frame_paths)} frames")
    print(f"Seg maps from: {SEGMAP_DIR}")
    print(f"Output → {OUTPUT_DIR}\n")

    # Load depth model
    depth_processor, depth_model, depth_device = load_depth_model()

    # Get id2label from a sample seg map to resolve class names
    # We need to reload the segmentation model config for this
    from transformers import AutoConfig
    seg_config = AutoConfig.from_pretrained(
        "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    )
    id2label = seg_config.id2label
    print(f"  Loaded {len(id2label)} class labels from seg model config\n")

    gif_frames = []
    results = []

    for i, fpath in enumerate(frame_paths):
        fname = os.path.basename(fpath)
        print(f"  [{i+1}/{len(frame_paths)}] {fname}", end="")

        # Load original image
        image = Image.open(fpath).convert("RGB")

        # Load segmentation map from Stage 1
        seg_map = load_seg_map(fname)
        if seg_map is None:
            print(f"  — SKIP (no seg map)")
            continue

        # Run depth estimation
        depth_map = estimate_depth(image, depth_processor, depth_model, depth_device)

        # Extract class masks
        sidewalk_mask, obstruction_mask, obstruction_info = get_class_masks(
            seg_map, id2label
        )

        if not sidewalk_mask.any():
            print(f"  — SKIP (no sidewalk detected)")
            continue

        # Compute perspective-corrected corridor width
        (widths_cm, widths_px, left_edges, right_edges,
         row_depths, blocked_sidewalk) = compute_corridor_width(
            sidewalk_mask, obstruction_mask, depth_map
        )

        # Classify rows
        labels = classify_rows(widths_cm)

        # Smooth out short yellow/red runs (perspective noise at vanishing point)
        labels = smooth_labels(labels, widths_cm)

        # Find pinch points (after smoothing, so only real issues remain)
        pinch_points = find_pinch_points(widths_cm, labels)

        # Detect full blockages (sidewalk completely eliminated)
        blockage_regions, full_block_mask = detect_full_blockages(
            sidewalk_mask, obstruction_mask, seg_map, id2label, widths_px
        )

        # Detect partial blockages (object ON the sidewalk but doesn't
        # fully eliminate it — like a car parked halfway on the sidewalk)
        partial_blockages = []
        if blocked_sidewalk.any():
            # Find vertical runs where obstruction overlaps sidewalk
            blocked_rows = blocked_sidewalk.any(axis=1)
            in_block = False
            block_start = 0
            for y in range(len(blocked_rows)):
                if blocked_rows[y]:
                    if not in_block:
                        block_start = y
                        in_block = True
                else:
                    if in_block and (y - block_start) >= 15:
                        # Find what's blocking
                        block_seg = seg_map[block_start:y, :]
                        block_sw = blocked_sidewalk[block_start:y, :]
                        blocking = []
                        for cid in np.unique(block_seg[block_sw]):
                            name = id2label.get(int(cid), "unknown").lower()
                            for obs_name in OBSTRUCTION_CLASSES:
                                if obs_name in name:
                                    blocking.append(name)
                                    break
                        if blocking:
                            partial_blockages.append(
                                (block_start, y, ", ".join(set(blocking)))
                            )
                    in_block = False
            if in_block and (len(blocked_rows) - block_start) >= 15:
                block_seg = seg_map[block_start:, :]
                block_sw = blocked_sidewalk[block_start:, :]
                blocking = []
                for cid in np.unique(block_seg[block_sw]):
                    name = id2label.get(int(cid), "unknown").lower()
                    for obs_name in OBSTRUCTION_CLASSES:
                        if obs_name in name:
                            blocking.append(name)
                            break
                if blocking:
                    partial_blockages.append(
                        (block_start, len(blocked_rows),
                         ", ".join(set(blocking)))
                    )

        # Create visualization
        viz = create_overlay(
            image, sidewalk_mask, blocked_sidewalk,
            widths_cm, widths_px, labels, left_edges, right_edges,
            pinch_points, obstruction_info,
            blockage_regions, full_block_mask,
            partial_blockages
        )
        viz.save(os.path.join(OUTPUT_DIR, f"width_{fname}"))

        # Score
        score, scene_label = compute_scene_score(
            widths_cm, labels, sidewalk_mask.shape[0]
        )

        # Stats
        valid_widths = widths_cm[widths_cm > 0]
        min_w = valid_widths.min() if len(valid_widths) > 0 else 0
        med_w = np.median(valid_widths) if len(valid_widths) > 0 else 0
        obstructions = list(obstruction_info.values())
        n_blockages = len(blockage_regions)
        obs_str = ", ".join(obstructions[:3]) if obstructions else "none"
        block_str = f", FULL BLOCKS:{n_blockages}" if n_blockages > 0 else ""

        print(f"  — score:{score} ({scene_label}), "
              f"min:{min_w:.0f}cm, med:{med_w:.0f}cm, "
              f"obstructions:[{obs_str}]{block_str}")

        results.append({
            "frame": fname,
            "score": score,
            "label": scene_label,
            "min_width_cm": round(min_w, 1),
            "median_width_cm": round(med_w, 1),
            "pinch_points": len(pinch_points),
            "full_blockages": n_blockages,
            "obstructions": "|".join(obstructions),
        })

        # GIF frame
        gif_frame = viz.copy()
        gif_frame.thumbnail((GIF_MAX_WIDTH, GIF_MAX_WIDTH), Image.LANCZOS)
        gif_frames.append(np.array(gif_frame))

    # Summary
    print(f"\n{'='*60}")
    scores = [r["score"] for r in results]
    dist = {}
    for r in results:
        dist[r["label"]] = dist.get(r["label"], 0) + 1
    print(f"Processed {len(results)} frames")
    if scores:
        print(f"Mean score: {np.mean(scores):.0f}/100")
    print(f"Distribution: {dist}")

    # Save GIF
    if gif_frames:
        print(f"\nSaving GIF → {GIF_PATH}")
        imageio.mimsave(GIF_PATH, gif_frames, fps=GIF_FPS, loop=0)
        size_mb = os.path.getsize(GIF_PATH) / (1024 * 1024)
        print(f"  Done ({len(gif_frames)} frames, {size_mb:.1f} MB)")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "width_analysis_results.csv")
    with open(csv_path, "w") as f:
        f.write("frame,score,label,min_width_cm,median_width_cm,"
                "pinch_points,full_blockages,obstructions\n")
        for r in results:
            f.write(f"{r['frame']},{r['score']},{r['label']},"
                    f"{r['min_width_cm']},{r['median_width_cm']},"
                    f"{r['pinch_points']},{r['full_blockages']},"
                    f"{r['obstructions']}\n")
    print(f"CSV saved → {csv_path}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
