"""
extract_frames.py
=================
Phase 1 — Frame Extraction & GPS Synchronization
Robot-Navigable City Project

Input structure:
  data/raw/batch_1/
    2622_Farwell_Ave-2026-03-14_16-40-32/
      video/front.mp4        <- or front.mp4 directly in session folder
      Location.csv           <- Sensor Logger (or inside gps/ subfolder)
      Metadata.csv           <- Sensor Logger (or inside gps/ subfolder)
    2748_2756_Waverly_Dr-2026-03-14_16-11-39/
      ...

Output (mirrors input exactly):
  data/processed/batch_1/
    2622_Farwell_Ave-2026-03-14_16-40-32/
      frames/
        2622_Farwell_00000.jpg     <- prefixed with short session ID
        2622_Farwell_00001.jpg
        ...
      frame_gps_map.csv      <- one row per frame: filename, timestamp, lat, lon, speed...
      session_summary.json
    2748_2756_Waverly_Dr-2026-03-14_16-11-39/
      frames/
        2748_Waverly_00000.jpg
        ...

After all sessions are processed, run merge_gps.py to combine all frame_gps_map.csv
files into one master CSV for inference.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ──────────────────────────────────────────────
# ✏️  EDIT THESE TWO LINES TO CONFIGURE THE SCRIPT
# ──────────────────────────────────────────────

PROJECT_ROOT  = Path("/Users/nandiyang/Documents/robot-navigable-city")
DEFAULT_BATCH = "batch_1"   # change to "batch_2" etc. when processing new recordings

# ──────────────────────────────────────────────
# CONFIG  (adjust if needed)
# ──────────────────────────────────────────────

DEFAULT_FPS    = 1      # frames per second to extract  (2 fps × ~56s ≈ 112 frames/session)
JPEG_QUALITY   = 92     # 0–100
TRIM_START_S   = 2.0    # seconds to skip at video start (removes fingers-in-frame)
TRIM_END_S     = 2.0    # seconds to skip at video end
SYNC_OFFSET_MS = 0      # ms shift if video start ≠ sensor start
                        #   positive → video started AFTER sensor
                        #   negative → video started BEFORE sensor
                        #   increase if frame_gps_map.csv shows many in_gps_range=False


# ──────────────────────────────────────────────
# PATH HELPERS
# ──────────────────────────────────────────────

def resolve_path(p: str) -> Path:
    """
    Resolve a path against PROJECT_ROOT if relative, or as-is if absolute.
    Means you can always use short paths like "data/raw/batch_1" regardless
    of what directory PyCharm or the terminal is running from.
    """
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def output_path_for(session_path: Path) -> Path:
    """
    Given:  PROJECT_ROOT/data/raw/batch_1/2622_Farwell_Ave-.../
    Return: PROJECT_ROOT/data/processed/batch_1/2622_Farwell_Ave-.../

    Structure:
      session_path.name        = "2622_Farwell_Ave-..."      (session folder name)
      session_path.parent.name = "batch_1"                   (batch name)
      session_path.parent.parent.parent = PROJECT_ROOT/data  (data root)
    """
    batch_name = session_path.parent.name                          # "batch_1"
    data_root  = session_path.parent.parent.parent                 # .../data/
    return data_root / "processed" / batch_name / session_path.name


def make_session_prefix(session_path: Path) -> str:
    """
    Derive a short, readable prefix from the session folder name.
    e.g. "2622_Farwell_Ave-2026-03-14_16-40-32"  ->  "2622_Farwell"
         "2748_2756_Waverly_Dr-2026-03-14_16-11-39" -> "2748_Waverly"

    Rule: take parts before the date (split on '-20'), then keep
    the first token (street number) and first word of street name.
    Strips common suffixes: Ave, Blvd, Dr, St, Rd.
    """
    folder = session_path.name
    # Drop date portion (everything from '-20' onward)
    base = folder.split("-20")[0]                  # e.g. "2622_Farwell_Ave"
    parts = base.split("_")                         # ["2622", "Farwell", "Ave"]
    # Drop street type suffixes
    suffixes = {"Ave", "Blvd", "Dr", "St", "Rd", "Ln", "Way", "Pl", "Ct"}
    parts = [p for p in parts if p not in suffixes]
    # Keep first two meaningful parts: number + street name
    prefix = "_".join(parts[:2])                   # "2622_Farwell"
    return prefix


# ──────────────────────────────────────────────
# FILE DISCOVERY
# ──────────────────────────────────────────────

def find_video(session_path: Path) -> Path | None:
    """Find video in Camera/ subfolder, video/ subfolder, or session root."""
    for folder in [session_path / "Camera", session_path / "video", session_path]:
        for ext in ["*.mp4", "*.mov", "*.MP4", "*.MOV", "*.m4v", "*.M4V"]:
            matches = list(folder.glob(ext))
            if matches:
                return matches[0]
    return None


def find_csv(session_path: Path, filename: str) -> Path | None:
    """Find a Sensor Logger CSV in gps/ subfolder or session root."""
    for candidate in [session_path / "gps" / filename, session_path / filename]:
        if candidate.exists():
            return candidate
    return None


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_metadata(meta_csv: Path) -> dict:
    """Parse Sensor Logger Metadata.csv."""
    df  = pd.read_csv(meta_csv)
    row = df.iloc[0]
    return {
        "device":             str(row.get("device name", "unknown")),
        "recording_epoch_ms": int(row["recording epoch time"]),   # UTC ms
        "recording_time":     str(row.get("recording time", "")),
        "timezone":           str(row.get("recording timezone", "UTC")),
    }


def load_gps(location_csv: Path) -> pd.DataFrame:
    """
    Load Sensor Logger Location.csv.
    'time' column is Unix nanoseconds → converted to ms.
    Rows with lat == 0 (no fix) are dropped.
    """
    df = pd.read_csv(location_csv)
    df["time_ms"] = df["time"] / 1_000_000
    df = df[df["latitude"] != 0].copy()
    df = df.sort_values("time_ms").reset_index(drop=True)

    print(f"  GPS   : {len(df)} fixes  |  "
          f"{df['seconds_elapsed'].min():.1f}s → {df['seconds_elapsed'].max():.1f}s elapsed")
    print(f"  Bounds: lat [{df['latitude'].min():.5f}, {df['latitude'].max():.5f}]  "
          f"lon [{df['longitude'].min():.5f}, {df['longitude'].max():.5f}]")
    return df


# ──────────────────────────────────────────────
# FRAME EXTRACTION
# ──────────────────────────────────────────────

def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float,
    img_size: tuple | None,
    session_prefix: str = "frame",
) -> pd.DataFrame:
    """
    Extract frames at target_fps, skipping TRIM_START_S at start and TRIM_END_S at end.

    Frames are named: {session_prefix}_{index:05d}.jpg
    e.g. 2622_Farwell_00001.jpg

    video_time_s in output = position in the ORIGINAL (untrimmed) video.
    This is intentional: GPS sync uses recording_epoch_ms + video_time_s,
    so we must NOT re-zero the clock after trimming or GPS would shift by 2 seconds.

    Returns DataFrame: frame_file, frame_index, video_time_s
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    native_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / native_fps

    trim_start_frame = int(TRIM_START_S * native_fps)
    trim_end_frame   = total_frames - int(TRIM_END_S * native_fps)
    active_s         = max(0.0, (trim_end_frame - trim_start_frame) / native_fps)

    if trim_end_frame <= trim_start_frame:
        print(f"  WARN  : Video too short ({duration_s:.1f}s) to trim {TRIM_START_S}+{TRIM_END_S}s. "
              f"Trim skipped.")
        trim_start_frame = 0
        trim_end_frame   = total_frames
        active_s         = duration_s

    print(f"  Video : {video_path.name}")
    print(f"          {native_fps:.1f} fps  |  {duration_s:.1f}s total  "
          f"|  {active_s:.1f}s after trim  "
          f"|  ~{int(active_s * target_fps)} frames @ {target_fps} fps")

    output_dir.mkdir(parents=True, exist_ok=True)

    interval     = native_fps / target_fps
    next_extract = float(trim_start_frame)   # skip everything before trim start
    native_idx   = 0
    extract_idx  = 0
    records      = []

    with tqdm(total=total_frames, desc="  Extracting", unit="f", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if native_idx >= trim_end_frame:         # stop at trim end
                pbar.update(total_frames - native_idx)
                break

            if native_idx >= next_extract:
                if img_size is not None:
                    frame = cv2.resize(frame, img_size, interpolation=cv2.INTER_AREA)

                filename = f"{session_prefix}_{extract_idx:05d}.jpg"
                cv2.imwrite(
                    str(output_dir / filename),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )
                records.append({
                    "frame_file":   filename,
                    "frame_index":  extract_idx,
                    "video_time_s": native_idx / native_fps,   # original clock
                })
                extract_idx  += 1
                next_extract += interval

            native_idx += 1
            pbar.update(1)

    cap.release()
    print(f"  Saved : {extract_idx} frames → {output_dir}")
    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# GPS SYNCHRONIZATION
# ──────────────────────────────────────────────

def assign_timestamps(frame_df: pd.DataFrame, recording_epoch_ms: int) -> pd.DataFrame:
    """
    Compute absolute UTC timestamp for each frame:
      frame_abs_ms = recording_epoch_ms + video_time_s * 1000 + SYNC_OFFSET_MS
    """
    frame_df = frame_df.copy()
    frame_df["frame_abs_ms"] = (
        recording_epoch_ms
        + frame_df["video_time_s"] * 1000
        + SYNC_OFFSET_MS
    )
    return frame_df


def interpolate_gps(gps_df: pd.DataFrame, query_ms: np.ndarray) -> pd.DataFrame:
    """
    Linearly interpolate GPS fields at each frame timestamp.
    in_gps_range = False means the frame fell outside the GPS recording window
    (GPS values will be clamped to the nearest boundary — check SYNC_OFFSET_MS).
    """
    t    = gps_df["time_ms"].values
    cols = ["latitude", "longitude", "altitude", "speed", "bearing", "horizontalAccuracy"]

    out = pd.DataFrame(
        {col: np.interp(query_ms, t, gps_df[col].values) for col in cols}
    ).rename(columns={"latitude": "lat", "longitude": "lon"})

    out["in_gps_range"] = (query_ms >= t.min()) & (query_ms <= t.max())
    return out


# ──────────────────────────────────────────────
# SINGLE SESSION PIPELINE
# ──────────────────────────────────────────────

def process_session(session_dir: str, fps: float, img_size) -> pd.DataFrame | None:
    session_path = resolve_path(session_dir)

    if not session_path.exists():
        print(f"[SKIP] Directory not found: {session_path}")
        return None

    video_path = find_video(session_path)
    gps_csv    = find_csv(session_path, "Location.csv")
    meta_csv   = find_csv(session_path, "Metadata.csv")

    missing = (
        (["video (.mp4/.mov)"] if video_path is None else []) +
        (["Location.csv"]      if gps_csv    is None else []) +
        (["Metadata.csv"]      if meta_csv   is None else [])
    )
    if missing:
        print(f"[SKIP] {session_path.name} — missing: {', '.join(missing)}")
        return None

    csv_layout  = "gps/ subfolder" if gps_csv.parent.name == "gps" else "flat"
    out_session = output_path_for(session_path)
    frames_dir  = out_session / "frames"

    print(f"\n{'='*60}")
    print(f"  Session : {session_path.name}")
    print(f"  Batch   : {session_path.parent.name}")
    print(f"  Input   : {session_path}")
    print(f"  Output  : {out_session}")
    print(f"  FPS     : {fps}  |  Trim: {TRIM_START_S}s / {TRIM_END_S}s")
    print(f"{'='*60}")

    meta   = load_metadata(meta_csv)
    gps_df = load_gps(gps_csv)
    print(f"  Device  : {meta['device']}  |  {meta['recording_time']} ({meta['timezone']})")

    prefix     = make_session_prefix(session_path)
    frame_df   = extract_frames(video_path, frames_dir, fps, img_size, session_prefix=prefix)
    frame_df   = assign_timestamps(frame_df, meta["recording_epoch_ms"])
    gps_interp = interpolate_gps(gps_df, frame_df["frame_abs_ms"].values)
    result     = pd.concat(
        [frame_df.reset_index(drop=True), gps_interp.reset_index(drop=True)], axis=1
    )

    result.to_csv(out_session / "frame_gps_map.csv", index=False)
    print(f"  Saved : frame_gps_map.csv  ({len(result)} rows)")

    oor     = int((~result["in_gps_range"]).sum())
    summary = {
        "session":             session_path.name,
        "batch":               session_path.parent.name,
        "device":              meta["device"],
        "recording_time":      meta["recording_time"],
        "timezone":            meta["timezone"],
        "fps_extracted":       fps,
        "trim_start_s":        TRIM_START_S,
        "trim_end_s":          TRIM_END_S,
        "total_frames":        len(result),
        "frames_in_gps_range": int(result["in_gps_range"].sum()),
        "frames_out_of_range": oor,
        "lat_center":          float(gps_df["latitude"].mean()),
        "lon_center":          float(gps_df["longitude"].mean()),
        "csv_layout":          csv_layout,
    }
    with open(out_session / "session_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if oor > 0:
        print(f"  WARN  : {oor} frame(s) outside GPS window (clamped).")
        print(f"          Adjust SYNC_OFFSET_MS at top of script if this is large.")
    else:
        print(f"  GPS   : all frames matched OK")

    return result


# ──────────────────────────────────────────────
# BATCH MODE
# ──────────────────────────────────────────────

def process_batch(batch: str, fps: float, img_size) -> None:
    """Process all session subfolders inside data/raw/<batch>/."""
    batch_path = resolve_path(f"data/raw/{batch}")

    if not batch_path.exists():
        sys.exit(f"[ERROR] Batch folder not found: {batch_path}")

    sessions = sorted([d for d in batch_path.iterdir() if d.is_dir()])
    if not sessions:
        sys.exit(f"[ERROR] No session subfolders found in {batch_path}")

    print(f"\nBatch   : {batch}")
    print(f"Input   : {batch_path}")
    print(f"Output  : {resolve_path(f'data/processed/{batch}')}")
    print(f"Sessions: {len(sessions)}")

    statuses = {}
    for i, s in enumerate(sessions, 1):
        print(f"\n[{i}/{len(sessions)}] {s.name}")
        result = process_session(str(s), fps, img_size)
        statuses[s.name] = "OK" if result is not None else "SKIPPED"

    n_ok      = sum(v == "OK"      for v in statuses.values())
    n_skipped = sum(v == "SKIPPED" for v in statuses.values())
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE — {n_ok} processed  |  {n_skipped} skipped")
    print(f"{'='*60}")
    for name, status in statuses.items():
        print(f"  {'[+]' if status == 'OK' else '[-]'}  {name}")
    print()


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract GPS-synced frames from Sensor Logger sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
If run with no arguments (e.g. PyCharm green button), processes DEFAULT_BATCH = "{DEFAULT_BATCH}".
Change DEFAULT_BATCH at the top of the script to switch batches.

Examples (from terminal):
  python extract_frames.py                              # uses DEFAULT_BATCH
  python extract_frames.py --batch batch_2             # override batch
  python extract_frames.py --session 2622_Farwell_Ave-2026-03-14_16-40-32  # single session
        """
    )
    parser.add_argument(
        "--batch", "-b",
        default=None,
        help="Batch name to process, e.g. batch_1 (default: DEFAULT_BATCH in script)"
    )
    parser.add_argument(
        "--session", "-s",
        default=None,
        metavar="SESSION_NAME",
        help="Single session name inside DEFAULT_BATCH, e.g. 2622_Farwell_Ave-2026-03-14_16-40-32"
    )
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help=f"Frames per second to extract (default: {DEFAULT_FPS})"
    )
    parser.add_argument(
        "--width",  type=int, default=None, help="Resize width  in px (needs --height)")
    parser.add_argument(
        "--height", type=int, default=None, help="Resize height in px (needs --width)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    img_size = None
    if args.width and args.height:
        img_size = (args.width, args.height)
    elif args.width or args.height:
        print("[WARN] Both --width and --height required for resize. Ignoring.")

    batch = args.batch or DEFAULT_BATCH

    if args.session:
        # Single session inside the current batch
        session_path = resolve_path(f"data/raw/{batch}/{args.session}")
        process_session(str(session_path), fps=args.fps, img_size=img_size)
    else:
        # Whole batch
        process_batch(batch, fps=args.fps, img_size=img_size)
