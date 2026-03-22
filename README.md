# Robot-Navigable City

**AI-powered sidewalk accessibility mapping using autonomous robot navigation data**

A computer vision pipeline that detects sidewalk accessibility barriers from street-level video, geo-tags each detection to GPS coordinates, and visualizes them as an interactive map. Built at the intersection of landscape architecture, machine learning, and urban robotics.

🗺️ **[Live Demo — Silver Lake / Atwater Village, Los Angeles](https://nandiyang.github.io/robot-navigable-city/data/processed/batch_1/inference/demo.html)**

---

## Motivation

Despite decades of ADA standards, sidewalk accessibility barriers persist across American cities. The same infrastructure failures that exclude wheelchair users also prevent autonomous delivery robots from navigating safely. This project leverages that parallel — using robot navigation data as a continuous, scalable diagnostic tool for accessibility mapping.

As robots join bikes and pedestrians as street users, they generate spatial data that reveals decades-old infrastructure failures. This pipeline transforms that data into actionable design tools for more inclusive streets.

---

## What It Detects

| Class | Description |
|---|---|
| `obstacle` | Objects blocking the sidewalk corridor — bins, poles, parked bikes, encroaching vegetation |
| `path_discontinuity` | Vertical drops, uplifted pavement, sudden grade changes that impair wheelchair and robot navigation |

---

## Pipeline Overview

```
Field Recording (iPhone + Sensor Logger)
          ↓
extract_frames.py     — extract GPS-synced frames at 1 FPS
          ↓
merge_gps.py          — merge per-session GPS into master CSV
          ↓
Roboflow              — manual annotation (obstacle / path_discontinuity)
          ↓
train.py              — YOLOv8s training on annotated dataset
          ↓
inference.py          — run model on all frames, join detections to GPS
          ↓
postprocess.py        — spatial deduplication (merge within 5m radius)
          ↓
visualize_detections.py — draw bounding boxes on detected frames
          ↓
build_demo.py         — generate interactive HTML map demo
```

---

## Results — Batch 1

- **Location:** Silver Lake / Atwater Village, Los Angeles
- **Sessions:** 10 walking routes (~15 min total)
- **Frames:** 947 extracted at 1 FPS
- **Labels:** 476 annotated frames (2 classes)
- **Model:** YOLOv8s, 100 epochs

| Metric | Value |
|---|---|
| mAP50 | 0.960 |
| mAP50-95 | 0.867 |
| Precision | 0.875 |
| Recall | 0.560 |

**Detections (conf ≥ 0.50, deduplicated at 5m):**
- Obstacles: ~150 unique locations
- Path discontinuities: ~30 unique locations

---

## Data Collection Protocol

- **Device:** iPhone 15 + Sensor Logger app (GPS + video synchronized)
- **Method:** Chest-height or stabilized handheld, landscape orientation, 1/3 sky framing
- **GPS:** Sensor Logger records Location.csv with nanosecond timestamps
- **Video:** Sensor Logger records to `Camera/` subfolder as `.mp4`
- **Trim:** First and last 2 seconds removed (camera handling artifacts)

---

## Project Structure

```
robot-navigable-city/
  scripts/
    extract_frames.py         — video → GPS-synced frames
    merge_gps.py              — combine session GPS into master CSV
    train.py                  — YOLOv8 training
    inference.py              — detection + GPS join
    postprocess.py            — spatial deduplication
    visualize_detections.py   — annotated frame images
    build_demo.py             — interactive HTML demo
    clean_roboflow_classes.py — Roboflow dataset cleanup utility
  data/
    raw/                      — field recordings (not tracked in git)
      batch_1/
        <session>/
          Camera/<video>.mp4
          Location.csv
          Metadata.csv
    processed/                — extracted frames + GPS maps (not tracked in git)
    datasets/                 — Roboflow YOLOv8 export (not tracked in git)
  runs/                       — YOLO training outputs (not tracked in git)
```

---

## Setup

```bash
# Clone
git clone https://github.com/nandiyang/robot-navigable-city
cd robot-navigable-city

# Create conda environment
conda create -n robot_yolo python=3.11
conda activate robot_yolo

# Install dependencies
pip install ultralytics opencv-python numpy pandas matplotlib \
            scikit-learn pyyaml tqdm pillow requests
```

---

## Usage

```bash
# 1. Extract frames from a batch of field recordings
python scripts/extract_frames.py

# 2. Merge GPS files into master CSV
python scripts/merge_gps.py

# 3. Train model (after labeling in Roboflow and downloading dataset)
python scripts/train.py

# 4. Run inference
python scripts/inference.py

# 5. Deduplicate detections spatially
python scripts/postprocess.py

# 6. Generate annotated frames
python scripts/visualize_detections.py

# 7. Build interactive demo
python scripts/build_demo.py

# 8. Open demo in browser
open data/processed/batch_1/inference/demo.html
```

---

## Roadmap

- [x] Phase 1 — Perception: frame extraction, GPS sync, YOLOv8 detection
- [ ] Phase 2 — Mapping: route-level safe passage analysis, accessibility scoring
- [ ] Phase 3 — Recovery: autonomy failure detection, robot-human mirror analysis
- [ ] Integration with autonomous delivery robot navigation data (Coco Robotics)
- [ ] Curb ramp detection as positive class
- [ ] Comparison with city ADA compliance records

---

## Research Context

This is an independent research project exploring the intersection of 
autonomous robot navigation and urban accessibility design. The work 
is informed by ongoing conversations with Prof. Bolei Zhou (UCLA) and 
his MetaUrban research group, whose work on embodied AI in urban 
environments is closely related.

A panel proposal connecting this research to landscape architecture 
practice has been submitted to the ASLA Annual Conference (result 
pending April 2026). This project continues regardless of that outcome.


**Related work:**
- [MetaUrban: A Simulation Platform for Embodied AI in Urban Spaces](https://metadriverse.github.io/metaurban/)
- [Urban Robotics Foundation — Public Area Mobile Robots Through a Planner's Lens](https://www.urbanroboticsfoundation.org/post/public-area-mobile-robots-through-a-planner-s-lens)
- [Streets for All](https://www.streetsforall.org/)

---

## Author

**Nandi Yang**
Landscape Architecture + Machine Learning
UCLA / Robot-Navigable City Project

---

## License

Data and trained models: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
Code: [MIT License](LICENSE)
