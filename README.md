# FormRate.ai — Prototype v0.1

AI-powered gym form analyzer for **Squat**, **Deadlift**, and **Bench Press**.

---

## How It Works

```
Video Input
    ↓
MediaPipe Pose Estimation  (33 body landmarks per frame)
    ↓
Rep Detection              (joint angle inflection points)
    ↓
Biomechanics Scoring       (rule-based, research-backed checks)
    ↓
Form Report (1–10 score + coaching cues per rep)
```

---

## Installation

```bash
pip install -r requirements.txt
```

> Requires Python 3.8+. MediaPipe works on macOS, Linux, and Windows.

---

## Usage

### Command Line

```bash
# Analyze a squat video
python formrate.py --video my_squat.mp4 --exercise squat

# Analyze a deadlift and save a JSON report
python formrate.py --video deadlift.mp4 --exercise deadlift --save-json report.json

# Analyze bench press and save annotated video with skeleton overlay
python formrate.py --video bench.mp4 --exercise bench --annotate bench_annotated.mp4

# Use high-accuracy model (slower but more precise)
python formrate.py --video squat.mp4 --exercise squat --model-complexity 2
```

### Python API

```python
from formrate import FormRateAnalyzer

analyzer = FormRateAnalyzer()
report = analyzer.analyze("my_squat.mp4", exercise="squat")

print(report.summary())          # human-readable output
print(report.to_json())          # machine-readable JSON
print(report.overall_score)      # float, e.g. 7.5
print(report.rep_scores[0].score)  # score for rep 1
```

---

## Sample Output

```
=======================================================
  FormRate.ai — SQUAT Analysis
=======================================================
  Video        : my_squat.mp4
  Reps Detected: 3
  Overall Score: 7.2 / 10
-------------------------------------------------------
  Rep  1  →  Score: 8.0/10
         ✓ squat_depth [88.3°]
         ✗ knee_valgus [1.24]
         ✓ spine_neutral [161.2°]
         ✓ knee_flexion [87.1°]
         💬 Mild knee cave detected. Actively push your knees out in line with your pinky toe.

  Rep  2  →  Score: 6.5/10
         ...
-------------------------------------------------------
  Top Coaching Cues:
    • Mild knee cave detected. Actively push your knees out.
    • Slightly above parallel. Aim to get your hip crease below your knee.
=======================================================
```

---

## Camera Setup Tips

For best results:
- **Squat**: Camera at hip height, 45° angle (side-front)
- **Deadlift**: Camera at hip height, direct side view
- **Bench Press**: Camera at bench height, direct side view
- Ensure full body is visible in frame
- Good lighting, no backlighting

---

## Scoring Criteria

### Squat
| Check | Ideal | Source |
|---|---|---|
| Depth | Hip crease ≤ parallel | Escamilla et al. 2001 |
| Knee valgus | Knee tracks over toes | Myer et al. 2008 |
| Spine | Neutral throughout | Hartmann et al. 2013 |
| Symmetry | <10° L/R difference | — |

### Deadlift
| Check | Ideal | Source |
|---|---|---|
| Spine at setup | Neutral (>145°) | Cholewicki et al. 1991 |
| Lockout | Full hip extension (165–185°) | Escamilla et al. 2002 |
| Knee tracking | No valgus on pull | McGuigan & Wilson 1996 |
| Hip-shoulder sync | Rise together | — |

### Bench Press
| Check | Ideal | Source |
|---|---|---|
| Elbow angle at bottom | 65–95° | Fees et al. 1998 |
| Lockout | Full extension (>160°) | Green & Comfort 2007 |
| Symmetry | <10° L/R difference | — |
| Shoulder stability | Scapulae retracted | Lehman 2005 |

---

## File Structure

```
formrate/
├── formrate.py          # Main analyzer + CLI entry point
├── pose_analyzer.py     # MediaPipe pose extraction
├── rep_detector.py      # Rep segmentation from pose sequences
├── scoring_engine.py    # Biomechanics rule checks + scoring
├── requirements.txt
└── README.md
```

---

## Roadmap (Post-Prototype)

- [ ] ML model trained on labeled form dataset (replaces rule engine)
- [ ] Real-time webcam mode
- [ ] Web UI / mobile app
- [ ] Additional exercises (OHP, rows, lunges)
- [ ] Fatigue detection across sets
- [ ] Progress tracking over time
