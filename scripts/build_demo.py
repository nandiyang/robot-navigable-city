"""
build_demo.py
=============
Phase 1 — Interactive HTML Demo Page
Robot-Navigable City Project

Generates a self-contained HTML demo page showing:
  - Left panel : interactive map with all detections as colored dots
                 (orange = obstacle, red = path_discontinuity)
  - Right panel: when you click a dot, shows the annotated frame
                 with bounding boxes drawn

The HTML file is fully self-contained — images are embedded as base64,
so you can share the single file with anyone (Prof. Zhou, ASLA audience)
without needing a server or internet connection.

Input:
  data/processed/batch_1/inference/detections_clean.csv
  data/processed/batch_1/inference/annotated/*.jpg

Output:
  data/processed/batch_1/inference/demo.html

Usage:
  python scripts/build_demo.py
  python scripts/build_demo.py --batch batch_2
  python scripts/build_demo.py --max-images 100  # limit embedded images for file size
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────
# ✏️  EDIT THESE TO CONFIGURE
# ──────────────────────────────────────────────
PROJECT_ROOT   = Path("/Users/nandiyang/Documents/robot-navigable-city")
DEFAULT_BATCH  = "batch_1"
MAX_IMAGES     = 200    # max annotated frames to embed (keeps file size manageable)
                        # increase if you want all frames in the demo
# ──────────────────────────────────────────────

# Colors matching visualize_detections.py (CSS format)
CLASS_COLORS = {
    "obstacle":           "#FFA500",   # orange
    "path_discontinuity": "#DC0000",   # red
}
DEFAULT_COLOR = "#AAAAAA"


def resolve_path(p) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def img_to_base64(img_path: Path) -> str | None:
    """Convert image file to base64 data URI."""
    try:
        with open(img_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{data}"
    except Exception:
        return None


def build_demo(batch: str, max_images: int):
    print(f"\n{'='*60}")
    print(f"  Robot-Navigable City — HTML Demo Builder")
    print(f"{'='*60}")
    print(f"  Batch      : {batch}")
    print(f"  Max images : {max_images}")
    print(f"{'='*60}\n")

    # Load detections
    csv_path = resolve_path(
        f"data/processed/{batch}/inference/detections_clean.csv"
    )
    if not csv_path.exists():
        sys.exit(f"[ERROR] detections_clean.csv not found: {csv_path}\n"
                 f"        Run postprocess.py first.")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["lat", "lon"])
    print(f"  Detections : {len(df)} loaded")

    # Load annotated frames
    annotated_dir = resolve_path(
        f"data/processed/{batch}/inference/annotated"
    )
    if not annotated_dir.exists():
        sys.exit(f"[ERROR] Annotated frames not found: {annotated_dir}\n"
                 f"        Run visualize_detections.py first.")

    # Build image lookup — frame_file -> base64
    print(f"  Embedding annotated frames (max {max_images})...")
    frame_files = sorted(annotated_dir.glob("*.jpg"))[:max_images]
    image_data  = {}
    for fp in frame_files:
        b64 = img_to_base64(fp)
        if b64:
            image_data[fp.name] = b64
    print(f"  Embedded   : {len(image_data)} frames")

    # Build map center
    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())

    # Prepare GeoJSON features for map
    features = []
    for _, row in df.iterrows():
        color = CLASS_COLORS.get(row["class_name"], DEFAULT_COLOR)
        has_image = row["frame_file"] in image_data
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["lon"], row["lat"]]
            },
            "properties": {
                "frame_file":  row["frame_file"],
                "class_name":  row["class_name"],
                "confidence":  round(float(row["confidence"]), 3),
                "session":     str(row.get("session", "")),
                "color":       color,
                "has_image":   has_image,
            }
        })

    geojson_str  = json.dumps({"type": "FeatureCollection", "features": features})
    imagedata_str = json.dumps(image_data)

    # ── Build HTML ─────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Robot-Navigable City — Sidewalk Accessibility Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #111; color: #eee; height: 100vh; display: flex;
          flex-direction: column; }}

  header {{
    background: #1a1a1a; padding: 14px 24px;
    border-bottom: 1px solid #333;
    display: flex; align-items: center; gap: 16px;
  }}
  header h1 {{ font-size: 1.1rem; font-weight: 600; color: #fff; }}
  header p  {{ font-size: 0.8rem; color: #888; }}

  .legend {{
    display: flex; gap: 20px; margin-left: auto; align-items: center;
  }}
  .legend-item {{
    display: flex; align-items: center; gap: 6px;
    font-size: 0.8rem; color: #ccc;
  }}
  .legend-dot {{
    width: 12px; height: 12px; border-radius: 50%;
  }}

  .main {{
    display: flex; flex: 1; overflow: hidden;
  }}

  #map {{
    flex: 1.2; height: 100%;
  }}

  .sidebar {{
    width: 600px; background: #1a1a1a;
    border-left: 1px solid #333;
    display: flex; flex-direction: column;
    overflow: hidden;
  }}

  .sidebar-header {{
    padding: 16px 20px 12px;
    border-bottom: 1px solid #2a2a2a;
  }}
  .sidebar-header h2 {{
    font-size: 0.85rem; font-weight: 600;
    color: #aaa; text-transform: uppercase; letter-spacing: 0.05em;
  }}

  .placeholder {{
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    color: #555; font-size: 0.9rem; gap: 8px; padding: 24px; text-align: center;
  }}
  .placeholder svg {{ opacity: 0.3; }}

  .detection-card {{
    flex: 1; overflow-y: auto; padding: 20px;
    display: none; flex-direction: column; gap: 16px;
  }}

  .detection-card img {{
    width: 100%; border-radius: 8px;
    border: 1px solid #2a2a2a;
  }}

  .detection-meta {{
    display: flex; flex-direction: column; gap: 8px;
  }}

  .class-badge {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600;
    width: fit-content;
  }}

  .meta-row {{
    display: flex; justify-content: space-between;
    font-size: 0.8rem; color: #888; padding: 6px 0;
    border-bottom: 1px solid #222;
  }}
  .meta-row span:last-child {{ color: #ccc; }}

  .conf-bar-wrap {{
    background: #222; border-radius: 4px; height: 6px; margin-top: 4px;
  }}
  .conf-bar {{
    height: 6px; border-radius: 4px;
    transition: width 0.3s ease;
  }}

  .stats-bar {{
    padding: 12px 20px;
    border-top: 1px solid #2a2a2a;
    display: flex; gap: 20px;
    font-size: 0.75rem; color: #666;
  }}
  .stats-bar span {{ color: #aaa; font-weight: 600; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>Robot-Navigable City &nbsp;·&nbsp; Sidewalk Accessibility Map</h1>
    <p>Silver Lake / Atwater Village, Los Angeles &nbsp;·&nbsp; Batch 1</p>
  </div>
  <div class="legend">
    <div class="legend-item">
      <div class="legend-dot" style="background:#FFA500"></div>
      Obstacle
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#DC0000"></div>
      Path Discontinuity
    </div>
  </div>
</header>

<div class="main">
  <div id="map"></div>

  <div class="sidebar">
    <div class="sidebar-header">
      <h2>Detection Detail</h2>
    </div>

    <div class="placeholder" id="placeholder">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="1.5">
        <circle cx="12" cy="12" r="10"/>
        <path d="M12 8v4M12 16h.01"/>
      </svg>
      <p>Click any dot on the map<br>to see the detection detail</p>
    </div>

    <div class="detection-card" id="detectionCard">
      <img id="detectionImg" src="" alt="Detection frame"/>
      <div class="detection-meta">
        <div id="classBadge" class="class-badge"></div>
        <div class="conf-bar-wrap">
          <div class="conf-bar" id="confBar"></div>
        </div>
        <div id="metaRows"></div>
      </div>
    </div>

    <div class="stats-bar">
      <div>Total &nbsp;<span id="statTotal">—</span></div>
      <div>Obstacles &nbsp;<span id="statObstacle">—</span></div>
      <div>Path Discontinuity &nbsp;<span id="statPD">—</span></div>
    </div>
  </div>
</div>

<script>
const GEOJSON    = {geojson_str};
const IMAGE_DATA = {imagedata_str};

// ── Map setup ──────────────────────────────────────────
const map = L.map("map", {{
  center: [{center_lat}, {center_lon}],
  zoom: 16,
  zoomControl: true,
}});

L.tileLayer("https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
  attribution: "© OpenStreetMap © CARTO",
  subdomains: "abcd",
  maxZoom: 20,
}}).addTo(map);

// ── Stats ──────────────────────────────────────────────
const total    = GEOJSON.features.length;
const nObs     = GEOJSON.features.filter(f => f.properties.class_name === "obstacle").length;
const nPD      = GEOJSON.features.filter(f => f.properties.class_name === "path_discontinuity").length;
document.getElementById("statTotal").textContent   = total;
document.getElementById("statObstacle").textContent = nObs;
document.getElementById("statPD").textContent       = nPD;

// ── Plot detection dots ────────────────────────────────
GEOJSON.features.forEach(feature => {{
  const props = feature.properties;
  const [lon, lat] = feature.geometry.coordinates;
  const color = props.color;

  const marker = L.circleMarker([lat, lon], {{
    radius:      props.class_name === "path_discontinuity" ? 9 : 7,
    fillColor:   color,
    color:       "#fff",
    weight:      1.5,
    opacity:     0.9,
    fillOpacity: 0.85,
  }}).addTo(map);

  marker.on("click", () => showDetection(props, color));
}});

// ── Show detection detail ──────────────────────────────
function showDetection(props, color) {{
  document.getElementById("placeholder").style.display    = "none";
  const card = document.getElementById("detectionCard");
  card.style.display = "flex";

  // Image
  const img = document.getElementById("detectionImg");
  if (IMAGE_DATA[props.frame_file]) {{
    img.src   = IMAGE_DATA[props.frame_file];
    img.style.display = "block";
  }} else {{
    img.style.display = "none";
  }}

  // Class badge
  const badge = document.getElementById("classBadge");
  badge.textContent   = props.class_name.replace("_", " ").toUpperCase();
  badge.style.background = color + "22";
  badge.style.color      = color;
  badge.style.border     = `1px solid ${{color}}55`;

  // Confidence bar
  const pct = Math.round(props.confidence * 100);
  document.getElementById("confBar").style.width      = pct + "%";
  document.getElementById("confBar").style.background = color;

  // Meta rows
  const rows = [
    ["Confidence",  pct + "%"],
    ["Session",     props.session.split("-")[0] || "—"],
    ["Frame",       props.frame_file],
  ];
  document.getElementById("metaRows").innerHTML = rows.map(([k, v]) =>
    `<div class="meta-row"><span>${{k}}</span><span>${{v}}</span></div>`
  ).join("");
}}
</script>
</body>
</html>"""

    # Save
    out_path = resolve_path(
        f"data/processed/{batch}/inference/demo.html"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = out_path.stat().st_size / 1_000_000
    print(f"\n{'='*60}")
    print(f"  DEMO BUILT")
    print(f"{'='*60}")
    print(f"  File    : {out_path}")
    print(f"  Size    : {size_mb:.1f} MB")
    print(f"  Points  : {len(features)}")
    print(f"  Images  : {len(image_data)} embedded")
    print(f"\n  Open in browser:")
    print(f"  open \"{out_path}\"")
    print(f"{'='*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build interactive HTML demo page."
    )
    parser.add_argument("--batch",      default=DEFAULT_BATCH,
                        help=f"Batch name (default: {DEFAULT_BATCH})")
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES,
                        help=f"Max frames to embed (default: {MAX_IMAGES})")
    return parser.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    build_demo(batch=args.batch, max_images=args.max_images)
