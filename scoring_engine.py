"""
FormRate.ai - Scoring Engine
Rule-based biomechanics scorer for Squat, Deadlift, and Bench Press.
Each check is derived from peer-reviewed biomechanics literature and
established coaching standards.

Scoring philosophy:
  - Each rep starts at 10.0
  - Deductions are applied per failed check
  - Final score = max(1.0, 10.0 - total_deductions)
  - Feedback cues are collected and returned alongside the score

Squat scoring philosophy (v0.2):
  - Depth and knee tracking are the PRIMARY indicators — highest weight
  - Spine angle is evaluated with powerlifting-aware thresholds (low-bar
    forward lean is correct technique, not a fault — McGill 2010)
  - Descent tempo deduction is halved for heavy/1RM loads where faster
    descent is intentional (stretch-reflex utilisation)
  - Symmetry tolerance widened: heavier loads naturally produce slightly
    more bilateral variance (Flanagan & Salem 2007)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from pose_analyzer import FramePose


# ─────────────────────────── Data Structures ──────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    deduction: float
    feedback: str
    value: Optional[float] = None


@dataclass
class RepScore:
    exercise: str
    rep_number: int
    score: float
    checks: List[CheckResult]
    feedback: List[str]
    raw_angles: Dict


# ─────────────────────────── Utility ──────────────────────────────────────

def avg_angle(frame_poses: List[FramePose], key: str) -> Optional[float]:
    vals = [fp.angles[key] for fp in frame_poses if key in fp.angles]
    return float(np.mean(vals)) if vals else None


def min_angle(frame_poses: List[FramePose], key: str) -> Optional[float]:
    vals = [fp.angles[key] for fp in frame_poses if key in fp.angles]
    return float(np.min(vals)) if vals else None


def max_angle(frame_poses: List[FramePose], key: str) -> Optional[float]:
    vals = [fp.angles[key] for fp in frame_poses if key in fp.angles]
    return float(np.max(vals)) if vals else None


def robust_min_angle(frame_poses: List[FramePose], key: str, pct: float = 5.0) -> Optional[float]:
    """
    Noise-robust minimum: return the Nth percentile instead of absolute min.
    A single bad MediaPipe frame can produce a spuriously low angle — using
    the 5th percentile means the reading must be sustained across several
    frames, not just one outlier.
    """
    vals = [fp.angles[key] for fp in frame_poses if key in fp.angles]
    return float(np.percentile(vals, pct)) if vals else None


def angle_at_bottom(frame_poses: List[FramePose], tracking_key: str, target_key: str) -> Optional[float]:
    """Return target_key angle at the frame where tracking_key is minimized."""
    min_val, best_fp = float('inf'), None
    for fp in frame_poses:
        v = fp.angles.get(tracking_key)
        if v is not None and v < min_val:
            min_val = v
            best_fp = fp
    if best_fp is None:
        return None
    return best_fp.angles.get(target_key)


def _bottom_frames(frame_poses: List[FramePose], key: str = "left_hip", window: int = 5) -> List[FramePose]:
    """
    Return the ~window frames closest to the bottom of the rep
    (where `key` angle is lowest). Used for checks that should
    be evaluated at depth rather than over the whole rep.
    """
    vals = [(i, fp.angles[key]) for i, fp in enumerate(frame_poses) if key in fp.angles]
    if not vals:
        return frame_poses
    bottom_i = min(vals, key=lambda x: x[1])[0]
    lo = max(0, bottom_i - window // 2)
    hi = min(len(frame_poses), lo + window)
    return frame_poses[lo:hi]


# ─────────────────────────── SQUAT SCORER ──────────────────────────────────

def score_squat_rep(frame_poses: List[FramePose], rep_number: int) -> RepScore:
    """
    Squat form checks based on:
    - Escamilla et al. (2001) knee biomechanics during squats
    - Myer et al. (2008) ACL/knee valgus risk during squatting
    - Hartmann et al. (2013) deep squat spine and hip biomechanics
    - Schoenfeld (2010) eccentric tempo and hypertrophic stimulus
    - McGill (2010) lumbar spine tolerance — low-bar forward lean is correct
    - Flanagan & Salem (2007) bilateral deficit and symmetry under load

    v0.3 changes:
    - Depth: uses robust_min_angle (5th percentile) to prevent one noisy
      MediaPipe frame from masking a genuinely shallow squat.
    - Knee/hip cross-validation: proper squat requires BOTH knee AND hip
      to flex together. Hip-dominant pattern (butt back, knees barely bending)
      is now detected and penalised separately.
    - Hip shift / good-morning pattern: new check comparing hip travel vs
      knee travel through the descent. Excessive rearward hip drive with
      minimal knee flexion = "sitting back" fault.
    - Butt wink: now uses robust spine values (not raw min) for noise immunity.
    - Knee valgus: evaluated only around bottom frames, not the whole rep,
      to avoid false positives during unloaded phases.
    """
    checks = []
    deductions = 0.0

    # ── Check 1: Depth (noise-robust) ── PRIMARY — weight: up to 3.0 pts ─
    # Use 5th-percentile angle so depth must be sustained across multiple
    # frames, not just one noisy outlier. Escamilla et al. (2001).
    min_hip = robust_min_angle(frame_poses, "left_hip") or robust_min_angle(frame_poses, "right_hip")
    if min_hip is not None:
        if min_hip <= 95:
            c = CheckResult("squat_depth", True, 0,
                "Good depth — hip crease at or below parallel.", min_hip)
        elif min_hip <= 110:
            c = CheckResult("squat_depth", False, 1.5,
                "Slightly above parallel. Drive your hips lower on the descent.", min_hip)
            deductions += 1.5
        else:
            c = CheckResult("squat_depth", False, 3.0,
                "Depth too shallow — hip crease well above the knee. Focus on sitting down, not back.", min_hip)
            deductions += 3.0
        checks.append(c)

    # ── Check 2: Hip-dominant pattern / "sitting back" ── PRIMARY — up to 2.5 pts ─
    # A proper squat has the knee angle change tracking the hip angle change
    # roughly 1:1. When someone sits back excessively (good-morning pattern),
    # hips hinge a lot but knees barely flex → hip ROM >> knee ROM.
    # We measure the full descent range for both joints.
    hip_rom = (max_angle(frame_poses, "left_hip") or 0) - (
        robust_min_angle(frame_poses, "left_hip") or 0)
    knee_rom = (max_angle(frame_poses, "left_knee") or 0) - (
        robust_min_angle(frame_poses, "left_knee") or 0)

    # Fallback to right side if left unavailable
    if hip_rom == 0:
        hip_rom = (max_angle(frame_poses, "right_hip") or 0) - (
            robust_min_angle(frame_poses, "right_hip") or 0)
    if knee_rom == 0:
        knee_rom = (max_angle(frame_poses, "right_knee") or 0) - (
            robust_min_angle(frame_poses, "right_knee") or 0)

    if hip_rom > 10 and knee_rom > 0:
        # Ratio > 1.8 means hips are doing most of the work (hip hinge bias)
        hip_knee_ratio = hip_rom / (knee_rom + 1e-6)
        if hip_knee_ratio <= 1.5:
            c = CheckResult("hip_knee_balance", True, 0,
                "Good balance of knee and hip flexion through the squat.", hip_knee_ratio)
        elif hip_knee_ratio <= 2.0:
            c = CheckResult("hip_knee_balance", False, 1.5,
                "Too much hip hinge, not enough knee bend. Think 'sit down' rather than 'sit back' — let your knees travel forward over your toes.", hip_knee_ratio)
            deductions += 1.5
        else:
            c = CheckResult("hip_knee_balance", False, 2.5,
                "Strong 'good morning' squat pattern — hips shooting back with minimal knee flexion. Drive your knees out and forward, keep your torso upright.", hip_knee_ratio)
            deductions += 2.5
        checks.append(c)

    # ── Check 3: Knee valgus ── PRIMARY — weight: up to 2.0 pts ─────────
    # Evaluate valgus only around the bottom of the rep where it matters most.
    # Using avg over bottom frames avoids flagging the unloaded top position.
    bottom_fps = _bottom_frames(frame_poses, key="left_hip", window=8)
    left_valgus = avg_angle(bottom_fps, "left_knee_valgus_ratio")
    right_valgus = avg_angle(bottom_fps, "right_knee_valgus_ratio")
    valgus_scores = [v for v in [left_valgus, right_valgus] if v is not None]
    if valgus_scores:
        max_valgus = max(valgus_scores)
        if max_valgus < 1.15:
            c = CheckResult("knee_valgus", True, 0,
                "Knees tracking well over toes.", max_valgus)
        elif max_valgus < 1.4:
            c = CheckResult("knee_valgus", False, 1.0,
                "Mild knee cave at the bottom. Cue 'knees out' actively on the way down and up.", max_valgus)
            deductions += 1.0
        else:
            c = CheckResult("knee_valgus", False, 2.0,
                "Sustained knee cave — cue 'spread the floor' and strengthen hip abductors.", max_valgus)
            deductions += 2.0
        checks.append(c)

    # ── Check 4: Spine angle ── SECONDARY — weight: up to 2.0 pts ───────
    # Low-bar powerlifting squat = 45–60° forward trunk lean (~120–140° in
    # our angle). High-bar / front squat = more upright (~140–160°). Both
    # are fine. Only penalise genuine collapse. McGill (2010), Hartmann (2013).
    min_spine = robust_min_angle(frame_poses, "spine", pct=10.0)
    if min_spine is not None:
        if min_spine >= 120:
            c = CheckResult("spine_neutral", True, 0,
                "Spine angle looks good — neutral or appropriate forward lean.", min_spine)
        elif min_spine >= 95:
            c = CheckResult("spine_neutral", False, 1.0,
                "More forward lean than ideal. Brace your core and drive your chest up out of the hole.", min_spine)
            deductions += 1.0
        else:
            c = CheckResult("spine_neutral", False, 2.0,
                "Excessive spinal flexion — chest collapsing toward knees. Work on thoracic mobility and upper-back bracing.", min_spine)
            deductions += 2.0
        checks.append(c)

    # ── Check 5: Butt wink ── SECONDARY — weight: up to 1.0 pt ──────────
    # Posterior pelvic tilt at bottom. Use robust values to reduce noise.
    # Hartmann et al. (2013).
    spine_at_mid = angle_at_bottom(frame_poses, "left_knee", "spine")
    spine_at_bottom = robust_min_angle(frame_poses, "spine", pct=10.0)
    if spine_at_mid is not None and spine_at_bottom is not None:
        pelvic_tilt_delta = spine_at_mid - spine_at_bottom
        if pelvic_tilt_delta <= 20:
            c = CheckResult("butt_wink", True, 0,
                "No significant posterior pelvic tilt detected.", pelvic_tilt_delta)
        else:
            c = CheckResult("butt_wink", False, 1.0,
                "Butt wink at the bottom — pelvis tucking under. Work on hip flexor and ankle mobility.",
                pelvic_tilt_delta)
            deductions += 1.0
        checks.append(c)

    # ── Check 6: Descent tempo ── INFORMATIONAL — weight: up to 1.0 pt ──
    bottom_frame = min(frame_poses, key=lambda fp: fp.angles.get("left_hip", 180))
    descent_time = bottom_frame.timestamp - frame_poses[0].timestamp
    if descent_time > 0:
        if descent_time >= 0.8:
            c = CheckResult("descent_tempo", True, 0,
                f"Controlled descent ({descent_time:.1f}s).", descent_time)
        elif descent_time >= 0.5:
            c = CheckResult("descent_tempo", False, 0.5,
                f"Descent slightly fast ({descent_time:.1f}s). Aim for a controlled 1–2s eccentric.",
                descent_time)
            deductions += 0.5
        else:
            c = CheckResult("descent_tempo", False, 1.0,
                f"Very fast descent ({descent_time:.1f}s) — ensure you're in control at the bottom.",
                descent_time)
            deductions += 1.0
        checks.append(c)

    # ── Check 7: Symmetry ── INFORMATIONAL — weight: up to 1.5 pts ──────
    left_knee_avg = avg_angle(frame_poses, "left_knee")
    right_knee_avg = avg_angle(frame_poses, "right_knee")
    if left_knee_avg and right_knee_avg:
        diff = abs(left_knee_avg - right_knee_avg)
        if diff <= 15:
            c = CheckResult("symmetry", True, 0,
                "Good bilateral symmetry.", diff)
        elif diff <= 25:
            c = CheckResult("symmetry", False, 0.75,
                f"Mild asymmetry ({diff:.0f}°). Check hip/ankle mobility side to side.", diff)
            deductions += 0.75
        else:
            c = CheckResult("symmetry", False, 1.5,
                f"Significant asymmetry ({diff:.0f}°). Assess for muscular imbalance or mobility restriction.", diff)
            deductions += 1.5
        checks.append(c)

    # ── Final score ──────────────────────────────────────────────────────
    score = max(1.0, round(10.0 - deductions, 1))
    feedback = [c.feedback for c in checks if not c.passed]
    raw_angles = {k: avg_angle(frame_poses, k) for k in
                  ["left_knee", "right_knee", "left_hip", "right_hip", "spine"]}

    return RepScore("squat", rep_number, score, checks, feedback, raw_angles)


# ─────────────────────────── DEADLIFT SCORER ───────────────────────────────

def score_deadlift_rep(frame_poses: List[FramePose], rep_number: int) -> RepScore:
    """
    Deadlift form checks based on:
    - Escamilla et al. (2002) lumbar and hip forces during deadlift
    - Cholewicki et al. (1991) lumbar spine stability during heavy lifting
    - McGuigan & Wilson (1996) biomechanical analysis of the deadlift
    """
    checks = []
    deductions = 0.0

    # ── Check 1: Hip hinge — spine at setup ──────────────────────────────
    min_spine = min_angle(frame_poses, "spine")
    if min_spine is not None:
        if min_spine >= 145:
            c = CheckResult("spine_at_setup", True, 0, "Good neutral spine at setup.", min_spine)
        elif min_spine >= 125:
            c = CheckResult("spine_at_setup", False, 2.0,
                "Rounded lower back at setup. Brace hard and 'proud chest' before the pull.", min_spine)
            deductions += 2.0
        else:
            c = CheckResult("spine_at_setup", False, 4.0,
                "Severe spinal flexion — high disc injury risk. Practice hip hinge with a dowel rod drill.", min_spine)
            deductions += 4.0
        checks.append(c)

    # ── Check 2: Lockout at top ──────────────────────────────────────────
    max_hip = max_angle(frame_poses, "left_hip") or max_angle(frame_poses, "right_hip")
    if max_hip is not None:
        if 165 <= max_hip <= 185:
            c = CheckResult("lockout", True, 0, "Full lockout achieved cleanly.", max_hip)
        elif max_hip < 165:
            c = CheckResult("lockout", False, 2.0,
                "Incomplete lockout — squeeze your glutes at the top to fully extend the hips.", max_hip)
            deductions += 2.0
        else:
            c = CheckResult("lockout", False, 1.5,
                "Hyperextension at lockout. Stand tall — don't lean back excessively.", max_hip)
            deductions += 1.5
        checks.append(c)

    # ── Check 3: Knee tracking ───────────────────────────────────────────
    left_valgus = avg_angle(frame_poses, "left_knee_valgus_ratio")
    right_valgus = avg_angle(frame_poses, "right_knee_valgus_ratio")
    valgus_vals = [v for v in [left_valgus, right_valgus] if v is not None]
    if valgus_vals:
        max_valgus = max(valgus_vals)
        if max_valgus < 1.15:
            c = CheckResult("knee_tracking", True, 0, "Knees tracking correctly during pull.", max_valgus)
        elif max_valgus < 1.3:
            c = CheckResult("knee_tracking", False, 1.5,
                "Knees slightly caving on the pull. Push your knees out as you drive through the floor.", max_valgus)
            deductions += 1.5
        else:
            c = CheckResult("knee_tracking", False, 2.5,
                "Knees collapsing during pull. Widen stance or address hip abductor weakness.", max_valgus)
            deductions += 2.5
        checks.append(c)

    # ── Check 4: Hip-to-shoulder rise synchrony ──────────────────────────
    hip_trajectory = [fp.angles.get("left_hip") for fp in frame_poses if fp.angles.get("left_hip")]
    spine_trajectory = [fp.angles.get("spine") for fp in frame_poses if fp.angles.get("spine")]
    if len(hip_trajectory) > 5 and len(spine_trajectory) > 5:
        hip_delta = hip_trajectory[-1] - hip_trajectory[0]
        spine_delta = spine_trajectory[-1] - spine_trajectory[0]
        if abs(hip_delta - spine_delta) < 30:
            c = CheckResult("hip_shoulder_sync", True, 0, "Hips and shoulders rising in sync.", None)
        else:
            c = CheckResult("hip_shoulder_sync", False, 1.5,
                "Hips rising faster than shoulders — avoid 'good morning' pattern. Drive chest up as you pull.", None)
            deductions += 1.5
        checks.append(c)

    # ── Check 5: Hip shoot at liftoff ────────────────────────────────────
    # Source: Hales (2010)
    first_10 = frame_poses[:10]
    hip_rise = (max_angle(first_10, "left_hip") or 0) - (min_angle(first_10, "left_hip") or 0)
    spine_change = (max_angle(first_10, "spine") or 0) - (min_angle(first_10, "spine") or 0)
    if hip_rise > 0 or spine_change > 0:
        if hip_rise <= spine_change + 20:
            c = CheckResult("hip_shoot", True, 0, "Hips and shoulders leaving the floor together.", None)
        else:
            c = CheckResult("hip_shoot", False, 1.5,
                "Hips shooting up before shoulders at liftoff. Think 'push the floor away' not 'pull the bar up'.", None)
            deductions += 1.5
        checks.append(c)

    score = max(1.0, round(10.0 - deductions, 1))
    feedback = [c.feedback for c in checks if not c.passed]
    raw_angles = {k: avg_angle(frame_poses, k) for k in
                  ["left_hip", "right_hip", "spine", "left_knee", "right_knee"]}

    return RepScore("deadlift", rep_number, score, checks, feedback, raw_angles)


# ─────────────────────────── BENCH PRESS SCORER ─────────────────────────────

def score_bench_rep(frame_poses: List[FramePose], rep_number: int) -> RepScore:
    """
    Bench press form checks based on:
    - Fees et al. (1998) upper extremity demands during bench press
    - Green & Comfort (2007) the effect of grip width on bench press performance
    - Lehman (2005) shoulder and chest activation in bench press variations
    """
    checks = []
    deductions = 0.0

    # ── Check 1: Elbow angle at bottom ──────────────────────────────────
    min_elbow = min_angle(frame_poses, "left_elbow") or min_angle(frame_poses, "right_elbow")
    if min_elbow is not None:
        if 65 <= min_elbow <= 95:
            c = CheckResult("elbow_angle_bottom", True, 0,
                "Good elbow angle at the bottom — safe shoulder position.", min_elbow)
        elif min_elbow > 95:
            c = CheckResult("elbow_angle_bottom", False, 2.0,
                "Elbows flaring too wide (>90°). Tuck elbows ~45–75° to protect the shoulder joint.", min_elbow)
            deductions += 2.0
        else:
            c = CheckResult("elbow_angle_bottom", False, 1.0,
                "Elbows very tucked. Slightly widen grip or let elbows flare to ~70°.", min_elbow)
            deductions += 1.0
        checks.append(c)

    # ── Check 2: Full extension at lockout ──────────────────────────────
    max_elbow = max_angle(frame_poses, "left_elbow") or max_angle(frame_poses, "right_elbow")
    if max_elbow is not None:
        if max_elbow >= 160:
            c = CheckResult("lockout", True, 0, "Full lockout at the top.", max_elbow)
        elif max_elbow >= 145:
            c = CheckResult("lockout", False, 1.5,
                "Partial lockout — extend your elbows fully at the top of each rep.", max_elbow)
            deductions += 1.5
        else:
            c = CheckResult("lockout", False, 2.5,
                "Significant incomplete lockout. Ensure full arm extension at the top.", max_elbow)
            deductions += 2.5
        checks.append(c)

    # ── Check 3: Symmetry ────────────────────────────────────────────────
    left_elbow_avg = avg_angle(frame_poses, "left_elbow")
    right_elbow_avg = avg_angle(frame_poses, "right_elbow")
    if left_elbow_avg and right_elbow_avg:
        diff = abs(left_elbow_avg - right_elbow_avg)
        if diff <= 10:
            c = CheckResult("symmetry", True, 0, "Good bilateral symmetry.", diff)
        elif diff <= 20:
            c = CheckResult("symmetry", False, 1.5,
                f"Arm asymmetry detected ({diff:.0f}° difference). Check for unilateral weakness or bar path deviation.", diff)
            deductions += 1.5
        else:
            c = CheckResult("symmetry", False, 3.0,
                f"Major asymmetry ({diff:.0f}° difference). One arm significantly weaker — use dumbbell variations to correct.", diff)
            deductions += 3.0
        checks.append(c)

    # ── Check 4: Shoulder stability ──────────────────────────────────────
    min_shoulder = min_angle(frame_poses, "left_shoulder") or min_angle(frame_poses, "right_shoulder")
    if min_shoulder is not None:
        if min_shoulder >= 50:
            c = CheckResult("shoulder_stability", True, 0, "Shoulder position looks stable.", min_shoulder)
        else:
            c = CheckResult("shoulder_stability", False, 1.5,
                "Shoulders may be coming off the bench or internally rotating. Retract and depress your scapulae.", min_shoulder)
            deductions += 1.5
        checks.append(c)

    score = max(1.0, round(10.0 - deductions, 1))
    feedback = [c.feedback for c in checks if not c.passed]
    raw_angles = {k: avg_angle(frame_poses, k) for k in
                  ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"]}

    return RepScore("bench", rep_number, score, checks, feedback, raw_angles)


# ─────────────────────────── Dispatcher ────────────────────────────────────

SCORER_MAP = {
    "squat": score_squat_rep,
    "deadlift": score_deadlift_rep,
    "bench": score_bench_rep,
}


def score_rep(exercise: str, frame_poses: List[FramePose], rep_number: int) -> RepScore:
    fn = SCORER_MAP.get(exercise.lower())
    if fn is None:
        raise ValueError(f"Unknown exercise: {exercise}")
    return fn(frame_poses, rep_number)
