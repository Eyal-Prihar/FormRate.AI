# FormRate.AI

> AI-powered gym form analyzer — upload a video, get a rep-by-rep biomechanics breakdown of your squat, deadlift, or bench press.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal?style=flat-square&logo=fastapi)
![Version](https://img.shields.io/badge/version-1.1.0-orange?style=flat-square)
![Status](https://img.shields.io/badge/status-active%20development-brightgreen?style=flat-square)

---

## What It Does

Most people train without a coach. FormRate.AI changes that.

Upload a video of your squat, deadlift, or bench press and get instant AI feedback on your form — scored rep by rep across six independent biomechanical categories. No wearables, no gym equipment, just a phone camera and your video.

The system automatically validates that your video contains a real exercise — random clips, group shots, or non-exercise videos are detected and rejected with helpful feedback. Uploaded videos are also compressed on the fly for faster processing.

**Live demo:** [formrate.ai](https://retain-conceptual-city-len.trycloudflare.com) *(tunnel-based, may require restart)*

---

## Demo

```
=======================================================
  FormRate.AI — SQUAT Analysis
=======================================================
  Video        : my_squat.mp4
  Reps Detected: 3
  Overall Score: 8.4 / 10
-------------------------------------------------------
  Rep  1  →  Score: 8.0/10

    Stability         ████████░░  4.0 / 5.0
    Range of Motion   ██████████  5.0 / 5.0
    Neutral Spine     █████████░  4.5 / 5.0
    Controlled Desc.  ██████████  5.0 / 5.0
    Leg Stance        ██████████  5.0 / 5.0
    Toe Flare         ████████░░  4.0 / 5.0

    ✓ squat_depth        [88.3°]
    ✗ knee_valgus        [1.31] — Mild knee cave. Cue 'knees out' on the way up.
    ✓ spine_angle        [138.2°]
    ✓ stance_width       [1.12×]
    ✓ toe_flare_angle    [24°]
    ✗ toe_flare_symmetry [14°] — Asymmetric flare (L:18° vs R:32°)
-------------------------------------------------------
  Top Coaching Cues:
    • Mild knee cave at depth. Actively cue 'knees out'.
    • Asymmetric toe flare — even out foot position.
=======================================================
```

---

## How It Works

```
Video Input  (.mp4 / .mov / .avi)
      │
      ▼
Video Compression  (ffmpeg)
  └─ Scales to 720p, re-encodes H.264 CRF 28
  └─ Strips audio — not needed for pose analysis
  └─ Reduces file size 50–85% for faster processing
      │
      ▼
MediaPipe Pose Estimation
  └─ 33 body landmarks tracked per frame
  └─ Joint angles computed each frame (hip, knee, spine, ankle, shoulder, wrist)
  └─ New: stance width ratio, toe flare angle, bar detection
      │
      ▼
Video Validation
  └─ Rejects non-exercise videos before scoring
  └─ Checks: landmark visibility, angle range sanity, standing posture,
     signal smoothness, spatial consistency (multi-person detection)
      │
      ▼
Rep Detection  (scipy signal processing)
  └─ Savitzky-Golay smoothing removes pose jitter
  └─ Prominence-based valley detection isolates individual reps
  └─ Noise-robust: minimum 45-frame distance between valleys
      │
      ▼
Biomechanics Scoring  (6-category system, research-backed)
  └─ Each category scored independently 0–5
  └─ Barbell auto-detected → 2 additional categories (bar position, bar path)
  └─ Overall score = mean of categories × 2  (reported on 0–10 scale)
      │
      ▼
Form Report
  └─ Per-rep category breakdown
  └─ Specific failed checks with coaching cues
  └─ JSON output via REST API or CLI summary
```

---

## Scoring System

Each rep is scored across **6 independent categories** (0–5 each). If a barbell is detected, two additional categories are included automatically.

### Squat

| Category | What's Measured | Research Basis |
|---|---|---|
| **Stability** | Knee valgus at depth + bilateral symmetry | Myer et al. 2008, Flanagan & Salem 2007 |
| **Range of Motion** | Hip depth (5th-percentile robust) + hip/knee balance ratio | Escamilla et al. 2001 |
| **Neutral Spine** | Torso angle at depth + posterior pelvic tilt (butt wink) | McGill 2010, Hartmann et al. 2013 |
| **Controlled Descent** | Eccentric tempo in seconds | Schoenfeld 2010 |
| **Leg Stance** | Ankle-to-hip width ratio | — |
| **Toe Flare** | Foot angle + left/right symmetry | — |
| **Bar Position** *(barbell)* | Bar height on traps + lateral tilt | Wretenberg et al. 1996 |
| **Bar Path** *(barbell)* | Horizontal wrist drift through full ROM | Wretenberg et al. 1996 |

### Deadlift

| Category | What's Measured | Research Basis |
|---|---|---|
| Spine at setup | Neutral spine before pull (>145°) | Cholewicki et al. 1991 |
| Lockout | Full hip extension (165–185°) | Escamilla et al. 2002 |
| Knee tracking | No valgus during pull | McGuigan & Wilson 1996 |
| Hip-shoulder sync | Hips and shoulders rising together | — |
| Hip shoot | Hips not rising before shoulders at liftoff | Hales 2010 |

### Bench Press

| Category | What's Measured | Research Basis |
|---|---|---|
| Elbow angle | 65–95° at bottom position | Fees et al. 1998 |
| Lockout | Full elbow extension at top (>160°) | Green & Comfort 2007 |
| Symmetry | Left/right elbow angle difference | — |
| Shoulder stability | Scapular retraction maintained | Lehman 2005 |

---

## Technical Architecture

```
formrate/
├── formrate.py          # Main analyzer + CLI entry point
│                          Includes video validation pipeline
├── pose_analyzer.py     # MediaPipe pose extraction + angle computation
│                          New: stance width, toe flare, bar detection
├── rep_detector.py      # Signal processing rep segmentation
│                          Savitzky-Golay smoothing + prominence detection
├── scoring_engine.py    # 6-category biomechanics scoring engine
│                          CategoryScore dataclass, barbell auto-detection
├── api.py               # FastAPI REST server + CORS + video compression
│                          ffmpeg compression, validation, serialization
├── index.html           # React frontend (single file, no build step)
├── requirements.txt
└── README.md
```

### Key Engineering Decisions

**Noise-robust depth measurement** — instead of taking the absolute minimum hip angle (which a single bad MediaPipe frame can corrupt), depth is measured at the 5th percentile across all frames. This requires depth to be sustained across multiple frames rather than flashing once.

**Category-based scoring over deduction-based** — the original system subtracted from 10. This caused bad performance in one area to bleed into unrelated areas. The new system scores each category independently so a lifter with a good spine but bad depth gets credit for the spine.

**Prominence-based rep detection** — rather than simple threshold crossing, rep boundaries are found using scipy's `find_peaks` with prominence filtering. A valley only counts as a rep if it stands out by at least 65% of the total range of motion — this eliminates wobble and pause artifacts at the bottom of a rep.

**Barbell auto-detection** — the system infers whether a barbell is present by checking if wrists are at trap height and wider than shoulder width across a majority of frames. No user input required.

**Pre-analysis video compression** — uploaded videos are compressed with ffmpeg (720p, H.264 CRF 28) before pose extraction. This reduces file sizes 50–85% and speeds up MediaPipe processing without meaningfully affecting landmark detection accuracy at 720p.

**Multi-layer video validation** — before scoring, uploaded videos pass through five checks: landmark visibility (are key joints actually visible, not hallucinated?), angle range sanity (ROM >130° = noise, not movement), standing posture detection (squats/deadlifts must show upright frames), signal smoothness (median frame-to-frame angle change >5° = unstable tracking), and spatial consistency (hip position std >0.15 = multiple people or camera cuts). This prevents non-exercise videos from receiving misleading scores.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/Eyal-Prihar/FormRate.AI.git
cd FormRate.AI

# Install system dependency (video compression)
brew install ffmpeg          # macOS
# sudo apt install ffmpeg    # Linux

# Install Python dependencies
pip install -r requirements.txt

# The MediaPipe model (~7MB) downloads automatically on first run
```

**Requirements:** Python 3.8+, ffmpeg, works on macOS, Linux, and Windows.

---

## Usage

### Web Interface

```bash
# Start the API server
uvicorn api:app --reload --port 8000

# Open index.html in your browser
# Upload a video, select your exercise, click Analyze
```

### Command Line

```bash
# Analyze a squat
python formrate.py --video my_squat.mp4 --exercise squat

# Analyze a deadlift and save JSON report
python formrate.py --video deadlift.mp4 --exercise deadlift --save-json report.json

# Save annotated video with skeleton overlay
python formrate.py --video bench.mp4 --exercise bench --annotate bench_annotated.mp4

# Higher accuracy (slower)
python formrate.py --video squat.mp4 --exercise squat --model-complexity 2
```

### Python API

```python
from formrate import FormRateAnalyzer

analyzer = FormRateAnalyzer()
report = analyzer.analyze("my_squat.mp4", exercise="squat")

print(report.overall_score)             # 8.4
print(report.rep_scores[0].score)       # 8.0
print(report.rep_scores[0].categories) # per-category breakdown
print(report.summary())                 # full human-readable report
print(report.to_json())                 # machine-readable JSON
```

### REST API

```bash
curl -X POST http://localhost:8000/analyze \
  -F "video=@my_squat.mp4" \
  -F "exercise=squat"
```

---

## Camera Setup Tips

| Exercise | Camera Position |
|---|---|
| Squat | Hip height, 45° side-front angle |
| Deadlift | Hip height, direct side view |
| Bench Press | Bench height, direct side view |

- Full body visible in frame at all times
- Good lighting, avoid backlighting
- Stable camera — tripod or leaned against something

---

## Roadmap

- [x] MediaPipe pose extraction (33 landmarks per frame)
- [x] Prominence-based rep detection with noise filtering
- [x] 6-category squat scoring system (v1.0.1)
- [x] Barbell auto-detection + bar position/path scoring
- [x] FastAPI backend + React frontend
- [x] Cloudflare tunnel for live sharing
- [x] Video validation — rejects non-exercise videos automatically
- [x] Video compression — ffmpeg pre-processing for faster analysis
- [ ] ML model trained on labeled video dataset (replaces rule engine)
- [ ] Real-time webcam mode
- [ ] Additional exercises (OHP, rows, lunges, hip thrust)
- [ ] Fatigue detection across sets
- [ ] Progress tracking over time
- [ ] Mobile app

---

## Contributing

Interested in collaborating? This project is actively looking for contributors, especially in:
- **ML / computer vision** — training a pose-based form classifier on labeled video data
- **Sports science** — validating and improving biomechanics scoring thresholds
- **Frontend / mobile** — building a proper React app or React Native mobile client

Open an issue or reach out directly.

---

## License

Copyright (c) 2025 Eyal Prihar. All rights reserved.

This source code is provided for viewing and evaluation purposes only. Unauthorized copying, modification, or commercial use without explicit written permission is prohibited.

---

*Built with [MediaPipe](https://mediapipe.dev/), [FastAPI](https://fastapi.tiangolo.com/), and [React](https://react.dev/).*