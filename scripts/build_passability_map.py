"""
build_passability_map.py
========================
Build an interactive HTML map showing sidewalk passability scores
from the width analysis, geo-referenced using master_gps_map.csv.

Each frame becomes a colored dot on the map:
  Green  = comfortable (score >= 70)
  Yellow = constrained (score 40-69)
  Red    = failure (score < 40)

Clicking a dot shows the width analysis overlay image.

Just run:  python build_passability_map.py
"""

import os
import csv
import json

# ── PATHS ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser("~/Documents/robot-navigable-city")
GPS_CSV = os.path.join(PROJECT_ROOT, "data/processed/batch_2/master_gps_map.csv")
WIDTH_CSV = os.path.join(PROJECT_ROOT, "data/processed/round2_labeling/width_analysis/width_analysis_results.csv")
WIDTH_IMG_DIR = os.path.join(PROJECT_ROOT, "data/processed/round2_labeling/width_analysis")
SEG_IMG_DIR = os.path.join(PROJECT_ROOT, "data/processed/round2_labeling/segmentation")
OUTPUT_HTML = os.path.join(PROJECT_ROOT, "data/processed/round2_labeling/passability_map.html")

# Also copy to the demo location for GitHub Pages
DEMO_HTML = os.path.join(PROJECT_ROOT, "data/processed/batch_2/passability_map.html")

# ── IMAGE PATH PREFIX ─────────────────────────────────────────────────
# Relative path from the HTML file to the width analysis images
# The HTML lives at data/processed/round2_labeling/passability_map.html
# Images live at data/processed/round2_labeling/width_analysis/width_*.jpg
WIDTH_IMG_REL = "width_analysis"
SEG_IMG_REL = "segmentation"


def load_gps_data():
    """Load GPS coordinates for each frame."""
    gps = {}
    with open(GPS_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gps[row["frame_file"]] = {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "session": row["session"],
                "bearing": float(row["bearing"]),
                "speed": float(row["speed"]),
                "in_gps_range": row["in_gps_range"] == "True",
            }
    return gps


def load_width_results():
    """Load width analysis scores."""
    results = {}
    if not os.path.exists(WIDTH_CSV):
        print(f"WARNING: {WIDTH_CSV} not found. Run sidewalk_width_analysis.py first.")
        return results

    with open(WIDTH_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["frame"]] = {
                "score": int(row["score"]),
                "label": row["label"],
                "min_width_cm": float(row["min_width_cm"]),
                "median_width_cm": float(row["median_width_cm"]),
                "pinch_points": int(row["pinch_points"]),
                "full_blockages": int(row.get("full_blockages", 0)),
                "obstructions": row.get("obstructions", ""),
            }
    return results


def score_to_color(score):
    if score >= 70:
        return "#22c55e"  # green
    elif score >= 40:
        return "#eab308"  # yellow
    else:
        return "#ef4444"  # red


def build_map(gps, width_results):
    """Build interactive Leaflet map HTML."""

    # Merge GPS + width analysis
    points = []
    for frame_file, gps_info in gps.items():
        if not gps_info["in_gps_range"]:
            continue

        width_info = width_results.get(frame_file, None)
        if width_info is None:
            continue

        points.append({
            "lat": gps_info["lat"],
            "lon": gps_info["lon"],
            "frame": frame_file,
            "session": gps_info["session"],
            "score": width_info["score"],
            "label": width_info["label"],
            "min_width": width_info["min_width_cm"],
            "median_width": width_info["median_width_cm"],
            "pinch_points": width_info["pinch_points"],
            "full_blockages": width_info["full_blockages"],
            "obstructions": width_info["obstructions"],
            "color": score_to_color(width_info["score"]),
        })

    if not points:
        print("ERROR: No points with both GPS and width data found.")
        print(f"  GPS entries: {len(gps)}")
        print(f"  Width results: {len(width_results)}")
        return None

    # Compute map center
    avg_lat = sum(p["lat"] for p in points) / len(points)
    avg_lon = sum(p["lon"] for p in points) / len(points)

    # Stats
    n_comfortable = sum(1 for p in points if p["label"] == "comfortable")
    n_constrained = sum(1 for p in points if p["label"] == "constrained")
    n_failure = sum(1 for p in points if p["label"] == "failure")
    n_no_sw = sum(1 for p in points if p["label"] == "no sidewalk")
    avg_score = sum(p["score"] for p in points) / len(points)

    points_json = json.dumps(points)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Robot-Navigable City — Sidewalk Passability Map</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
        #map {{ width: 100%; height: 100vh; }}

        .info-panel {{
            position: absolute;
            top: 12px;
            right: 12px;
            z-index: 1000;
            background: rgba(255,255,255,0.95);
            border-radius: 10px;
            padding: 16px;
            max-width: 320px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.15);
            font-size: 13px;
            line-height: 1.5;
        }}
        .info-panel h2 {{
            font-size: 15px;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .info-panel .subtitle {{
            color: #666;
            font-size: 12px;
            margin-bottom: 12px;
        }}

        .stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            margin-bottom: 12px;
        }}
        .stat {{
            background: #f5f5f5;
            border-radius: 6px;
            padding: 8px;
            text-align: center;
        }}
        .stat .num {{
            font-size: 20px;
            font-weight: 600;
        }}
        .stat .lbl {{
            font-size: 11px;
            color: #888;
        }}

        .legend {{
            display: flex;
            gap: 12px;
            margin-bottom: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 12px;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}

        .filter-row {{
            display: flex;
            gap: 6px;
        }}
        .filter-btn {{
            padding: 4px 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background: white;
            cursor: pointer;
            font-size: 12px;
        }}
        .filter-btn.active {{
            background: #333;
            color: white;
            border-color: #333;
        }}

        .frame-popup {{
            min-width: 280px;
        }}
        .frame-popup img {{
            width: 100%;
            border-radius: 6px;
            margin-bottom: 6px;
        }}
        .frame-popup .meta {{
            font-size: 12px;
            color: #555;
            line-height: 1.6;
        }}
        .frame-popup .score-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            color: white;
            font-weight: 600;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>

    <div class="info-panel">
        <h2>Sidewalk Passability Map</h2>
        <div class="subtitle">
            Robot-Navigable City — Silver Lake / Atwater Village, LA<br>
            {len(points)} analyzed frames across {len(set(p['session'] for p in points))} walking sessions
        </div>

        <div class="stats">
            <div class="stat">
                <div class="num">{avg_score:.0f}</div>
                <div class="lbl">Avg score</div>
            </div>
            <div class="stat">
                <div class="num" style="color:#22c55e">{n_comfortable}</div>
                <div class="lbl">Comfortable</div>
            </div>
            <div class="stat">
                <div class="num" style="color:#eab308">{n_constrained}</div>
                <div class="lbl">Constrained</div>
            </div>
            <div class="stat">
                <div class="num" style="color:#ef4444">{n_failure + n_no_sw}</div>
                <div class="lbl">Failure</div>
            </div>
        </div>

        <div class="legend">
            <div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div> Comfortable (&gt;150cm)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#eab308"></div> Constrained (91-150cm)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div> Failure (&lt;91cm)</div>
        </div>

        <div class="filter-row">
            <button class="filter-btn active" onclick="filterPoints('all')">All</button>
            <button class="filter-btn" onclick="filterPoints('comfortable')">Comfortable</button>
            <button class="filter-btn" onclick="filterPoints('constrained')">Constrained</button>
            <button class="filter-btn" onclick="filterPoints('failure')">Failure</button>
        </div>
    </div>

    <script>
        const points = {points_json};

        const map = L.map('map').setView([{avg_lat}, {avg_lon}], 17);

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap &copy; CARTO',
            maxZoom: 20,
        }}).addTo(map);

        let markers = [];

        function createMarkers(filter) {{
            // Clear existing
            markers.forEach(m => map.removeLayer(m));
            markers = [];

            points.forEach(p => {{
                if (filter !== 'all' && p.label !== filter) return;

                const marker = L.circleMarker([p.lat, p.lon], {{
                    radius: 6,
                    fillColor: p.color,
                    color: 'white',
                    weight: 1.5,
                    fillOpacity: 0.85,
                }});

                const obsList = p.obstructions
                    ? p.obstructions.split('|').filter(o => o).join(', ')
                    : 'none';

                const scoreBg = p.score >= 70 ? '#22c55e' : p.score >= 40 ? '#eab308' : '#ef4444';

                marker.bindPopup(`
                    <div class="frame-popup">
                        <img src="{WIDTH_IMG_REL}/width_${{p.frame}}"
                             onerror="this.src='{SEG_IMG_REL}/seg_${{p.frame}}'"
                             alt="${{p.frame}}" />
                        <div class="meta">
                            <span class="score-badge" style="background:${{scoreBg}}">${{p.score}}/100</span>
                            <strong>${{p.label}}</strong><br>
                            Min width: ${{p.min_width.toFixed(0)}}cm &middot;
                            Median: ${{p.median_width.toFixed(0)}}cm<br>
                            Pinch points: ${{p.pinch_points}} &middot;
                            Blockages: ${{p.full_blockages}}<br>
                            Obstructions: ${{obsList}}<br>
                            <span style="color:#999">${{p.frame}}</span>
                        </div>
                    </div>
                `, {{ maxWidth: 320 }});

                marker.addTo(map);
                markers.push(marker);
            }});
        }}

        function filterPoints(filter) {{
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            createMarkers(filter);
        }}

        // Initial render
        createMarkers('all');
    </script>
</body>
</html>"""

    return html


def main():
    print("Loading GPS data...")
    gps = load_gps_data()
    print(f"  {len(gps)} GPS entries loaded")

    print("Loading width analysis results...")
    width_results = load_width_results()
    print(f"  {len(width_results)} width results loaded")

    print("\nBuilding map...")
    html = build_map(gps, width_results)

    if html is None:
        return

    # Save map
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
    print(f"Map saved → {OUTPUT_HTML}")

    # Also save to demo location
    os.makedirs(os.path.dirname(DEMO_HTML), exist_ok=True)
    with open(DEMO_HTML, "w") as f:
        f.write(html)
    print(f"Demo copy → {DEMO_HTML}")

    print(f"\nOpen in browser: file://{OUTPUT_HTML}")
    print("Done!")


if __name__ == "__main__":
    main()
