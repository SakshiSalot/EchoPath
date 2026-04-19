# EchoPath

**Real-Time Indoor Assistive Navigation System for Visually Impaired Users**

EchoPath is a computer vision pipeline that helps visually impaired students navigate college buildings independently. Using a smartphone-class camera, the system detects obstacles, estimates their proximity, determines their position relative to the user (left / ahead / right), and delivers concise spoken alerts such as *"person close ahead"* or *"chair medium left."*

The system is designed to work without any pre-installed building infrastructure — no beacons, no IR emitters, no pre-mapped floor plans. A camera and a mid-range edge device (NVIDIA Jetson Nano during development, smartphone as the final target) are all that's needed.

---

## Why this project exists

India has approximately 4.95 million blind persons and around 35 million with some form of vision impairment. A significant portion falls in the college-going age group, yet only a small fraction makes it to higher education. Research consistently identifies **mobility and orientation on campus** as one of the biggest barriers.

Existing systems either require expensive infrastructure (NavCog's Bluetooth beacons, IIT Delhi's Roshni IR emitters) or solve only part of the problem (Microsoft Seeing AI requires manual photo capture; Google Lookout identifies objects but gives no directional guidance). EchoPath aims to bridge this gap with a lightweight, infrastructure-free, real-time solution.

---

## How it works

The pipeline combines two vision models with a rule-based decision layer:

```
Camera frame
    │
    ├──► YOLO (fine-tuned on campus objects)
    │         └─► bounding boxes + class labels
    │
    ├──► MiDaS Small
    │         └─► relative depth map
    │
    └──► Data Fusion
              ├─► avg depth per bbox → proximity zone (close / medium / far)
              └─► bbox center x → direction (left / ahead / right)
                       │
                       ▼
              Decision Logic (rule-based alert filtering)
                       │
                       ▼
              Audio Output (pyttsx3, offline TTS)
```

Only close and medium-range obstacles in the user's path trigger alerts, and new alerts are spoken only when the scene meaningfully changes — avoiding the cognitive overload of constant narration.

---

## Models used

| Component | Model | Size | Purpose |
|-----------|-------|-----:|---------|
| Object detection | YOLOv8n (fine-tuned) | ~6 MB | Identify doors, chairs, stairs, people, common campus obstacles |
| Depth estimation | MiDaS Small (v2.1) | ~25 MB | Estimate relative depth per frame |
| Text-to-speech | pyttsx3 (espeak backend) | — | Offline voice alerts |

### About MiDaS depth output

MiDaS produces a **relative inverse depth map** — a grayscale image where higher pixel values indicate objects closer to the camera. The values are not calibrated to metric units; they are normalized per-frame and interpreted via threshold-based zones:

- Normalized depth > 0.70 → **close**
- 0.40 – 0.70 → **medium**
- < 0.40 → **far**

Thresholds can be calibrated per deployment environment using the notebook's calibration cell.

### Why MiDaS Small

- **Lightweight** — ~25 MB, runs comfortably alongside YOLOv8n on edge devices
- **Real-time capable** — proven deployment on Jetson Nano with TensorRT optimization
- **Smartphone-ready** — the small footprint makes eventual mobile deployment feasible
- **Well-documented** — mature community support and abundant deployment guides

A comparison notebook evaluating MiDaS Small against alternative depth models (such as Depth Anything V2) is included in the repository for benchmarking purposes.

---

## Repository structure

```
EchoPath/
├── main.py                    # CLI entry point
├── Yolo_train.py              # Script to fine-tune YOLO on campus data
├── echopath_test.ipynb        # Main pipeline notebook (YOLO + MiDaS)
├── yolov8n.pt                 # Pretrained YOLOv8 nano weights
├── yolo26n.pt                 # YOLO weights (checkpoint)
├── runs/detect/runs/echopath/ # Fine-tuned YOLO checkpoints + training logs
├── pyproject.toml             # uv project definition + dependencies
├── requirements.txt           # Pip-compatible mirror of dependencies
└── uv.lock                    # Lock file for reproducible installs
```

---

## Getting started

### Prerequisites

- Python 3.11 or 3.12
- CUDA 11.8 (if using GPU — strongly recommended)
- [uv](https://github.com/astral-sh/uv) package manager (or pip)
- ffmpeg (bundled via `imageio-ffmpeg`, no system install needed)

### Installation

Using uv (recommended):

```bash
git clone https://github.com/SakshiSalot/EchoPath.git
cd EchoPath
uv sync
```

Using pip:

```bash
git clone https://github.com/SakshiSalot/EchoPath.git
cd EchoPath
python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Verify GPU

```python
import torch
print(torch.cuda.is_available())   # should print True
print(torch.cuda.get_device_name(0))
```

---

## Usage

### Option 1 — Jupyter notebook (recommended for exploration)

Open `echopath_test.ipynb` and run cells sequentially. The notebook walks through:

1. GPU verification
2. Path configuration (point at your video folder + YOLO weights)
3. Model loading (YOLO + MiDaS Small via torch.hub)
4. Pipeline functions (rotation handling, depth fusion, alerts)
5. Calibration (verify depth output on a sample frame, tune thresholds)
6. Rotation diagnostic (detect phone-recorded video orientation)
7. Batch video processing
8. Alert summary

Each processed video produces an annotated `.mp4` with bounding boxes, class labels, proximity zone, direction, and normalized depth value overlaid on every detected object.

### Option 2 — Command line

```bash
uv run python main.py
```

Or fine-tune YOLO on your own dataset:

```bash
uv run python Yolo_train.py
```

---

## Output format

Each processed video shows overlays like:

```
person | close  | ahead | d=0.82
chair  | medium | left  | d=0.54
door   | far    | right | d=0.25
```

Where `d` is the normalized depth value (higher = closer, range 0–1 per frame).

The alert log captures changes across the video:

```
[ 2.50s | Frame   75]  person medium ahead
[ 4.12s | Frame  123]  person close ahead
[ 8.30s | Frame  249]  chair medium left
```

Box colors indicate urgency: red for close, orange for medium, green for far.

---

## Known limitations

- **Relative depth only** — MiDaS produces normalized per-frame depth, not real-world meters. Thresholds (0.70 / 0.40) are calibrated to corridor footage and may need retuning for other environments.
- **Bounding box averaging** — depth within a YOLO bbox is averaged, which can mix foreground and background pixels. Tall narrow objects (e.g., a person against a distant wall) may read differently than expected. Percentile-based sampling is used to mitigate this.
- **Per-frame scale drift** — because depth is normalized per frame, the same real-world distance can produce different values across frames. This is inherent to relative-depth models.
- **Textureless surfaces** — plain walls and featureless floors can produce noisy depth maps. This affects the reliability of proximity classification in sparse scenes.
- **Phone video rotation** — modern smartphones record landscape-oriented frames with a rotation metadata flag. The pipeline detects and corrects this, but the diagnostic cell should be run once per new dataset.

---

## Development status

This is an active student project with an 18-month development plan (see project proposal). Current status:

- ✅ YOLO fine-tuned on campus-specific objects
- ✅ MiDaS Small integrated with YOLO via bbox-level fusion
- ✅ Video batch processing with automatic orientation correction
- ✅ Depth model comparison notebook (MiDaS vs alternatives)
- ⏳ Audio output (pyttsx3) — integrated, smoothing under tuning
- ⏳ Jetson Nano deployment — planned for next milestone
- ⏳ Smartphone port — final milestone

---

## Contributors

- Harsh Raj
- Sakshi Salot
- Vedant Jadhav
- Pratima Chauhan

Project undertaken as part of the Pattern Recognition and Machine Learning course.

---

## References

Full academic references are documented in the project proposal. Key sources:

- Ahmetovic et al., *NavCog: A navigational cognitive assistant for the blind* (ACM ASSETS 2016)
- MiDaS (Intel ISL) — [https://github.com/isl-org/MiDaS](https://github.com/isl-org/MiDaS)
- Ultralytics YOLOv8 — [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- ASSISTECH IIT Delhi, *Roshni: Indoor wayfinding for the visually impaired*
- Messaoudi et al., *Review of navigation assistive tools and technologies for the visually impaired* (2022)
