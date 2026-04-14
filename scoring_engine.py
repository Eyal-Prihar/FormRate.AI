"""
FormRate.ai - Scoring Engine
Rule-based biomechanics scorer for Squat, Deadlift, and Bench Press.
Each check is derived from peer-reviewed biomechanics literature and
established coaching standards.

Squat scoring philosophy (v0.4):
  Each rep is scored across SIX independent categories, each 0–5:
    1. Stability          — balance, valgus, bilateral symmetry
    2. Range of Motion    — hip/knee depth, hip-knee balance
    3. Neutral Spine      — torso angle, butt wink
    4. Controlled Descent — eccentric tempo
    5. Leg Stance         — stance width relative to hips
    6. Toe Flare          — foot angle relative to forward axis

  If a barbell is detected (wrists at trap height, arms wide):
    + Bar Position        — bar height on traps, L/R level
    + Bar Path            — lateral drift of bar through ROM

  overall_score = mean of all category scores (0–5 scale)
  reported as rep.score on a 0–10 scale (multiply by 2) for
  API / UI consistency.
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
class CategoryScore:
    """One of the six (or eight for barbell) scoring categories."""
    name: str           # e.g. "Range of Motion"
    score: float        # 0.0 – 5.0
    max_score: float    # always 5.0
    checks: List[CheckResult] = field(default_factory=list)
    feedback: List[str] = field(default_factory=list)


@dataclass
class RepScore:
    exercise: str
    rep_number: int
    score: float                        # 0–10 (sum of category scores * 2 / n_categories)
    checks: List[CheckResult]           # flat list for API backward-compat
    feedback: List[str]
    raw_angles: Dict
    categories: List[CategoryScore] = field(default_factory=list)  # new structured breakdown


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
    Scores one squat rep across six independent categories (0–5 each).
    If a barbell is detected, two additional barbell categories are scored.

    References:
    - Escamilla et al. (2001) knee biomechanics during squats
    - Myer et al. (2008) knee valgus and ACL risk
    - Hartmann et al. (2013) deep squat spine / hip biomechanics
    - Schoenfeld (2010) eccentric tempo and hypertrophic stimulus
    - McGill (2010) lumbar spine — low-bar forward lean is correct technique
    - Flanagan & Salem (2007) bilateral symmetry under load
    - Wretenberg et al. (1996) high-bar vs low-bar bar path mechanics
    """

    categories: List[CategoryScore] = []

    # ════════════════════════════════════════════════════════════════════════
    # CATEGORY 1 — STABILITY  (knee valgus + bilateral symmetry)
    # ════════════════════════════════════════════════════════════════════════
    cat1_checks = []
    cat1_score  = 5.0

    # 1a. Knee valgus — evaluated only at bottom frames (loaded position)
    bottom_fps = _bottom_frames(frame_poses, key="left_hip", window=8)
    lv = avg_angle(bottom_fps, "left_knee_valgus_ratio")
    rv = avg_angle(bottom_fps, "right_knee_valgus_ratio")
    valgus_vals = [v for v in [lv, rv] if v is not None]
    if valgus_vals:
        mv = max(valgus_vals)
        if mv < 1.15:
            cat1_checks.append(CheckResult("knee_valgus", True, 0,
                "Knees tracking well over toes at depth.", mv))
        elif mv < 1.4:
            cat1_score -= 1.5
            cat1_checks.append(CheckResult("knee_valgus", False, 1.5,
                "Mild knee cave at depth. Actively cue 'knees out' on the descent and drive up.", mv))
        else:
            cat1_score -= 3.0
            cat1_checks.append(CheckResult("knee_valgus", False, 3.0,
                "Significant knee cave — cue 'spread the floor' and strengthen hip abductors.", mv))

    # 1b. Bilateral symmetry (left vs right knee average angle over full rep)
    lka = avg_angle(frame_poses, "left_knee")
    rka = avg_angle(frame_poses, "right_knee")
    if lka and rka:
        diff = abs(lka - rka)
        if diff <= 12:
            cat1_checks.append(CheckResult("symmetry", True, 0,
                "Good bilateral symmetry.", diff))
        elif diff <= 22:
            cat1_score -= 1.0
            cat1_checks.append(CheckResult("symmetry", False, 1.0,
                f"Mild left/right asymmetry ({diff:.0f}°). Check hip and ankle mobility side to side.", diff))
        else:
            cat1_score -= 2.0
            cat1_checks.append(CheckResult("symmetry", False, 2.0,
                f"Significant asymmetry ({diff:.0f}°). Assess for muscular imbalance or mobility restriction.", diff))

    cat1_score = max(0.0, cat1_score)
    categories.append(CategoryScore(
        name="Stability",
        score=cat1_score,
        max_score=5.0,
        checks=cat1_checks,
        feedback=[c.feedback for c in cat1_checks if not c.passed],
    ))

    # ════════════════════════════════════════════════════════════════════════
    # CATEGORY 2 — RANGE OF MOTION  (depth + hip-knee balance)
    # ════════════════════════════════════════════════════════════════════════
    cat2_checks = []
    cat2_score  = 5.0

    # 2a. Squat depth — robust min (5th percentile) so one noisy frame
    #     can't fake a deep squat. Escamilla et al. (2001).
    min_hip = robust_min_angle(frame_poses, "left_hip") or robust_min_angle(frame_poses, "right_hip")
    if min_hip is not None:
        if min_hip <= 90:
            cat2_checks.append(CheckResult("squat_depth", True, 0,
                "Excellent depth — hip crease clearly below parallel.", min_hip))
        elif min_hip <= 100:
            cat2_checks.append(CheckResult("squat_depth", True, 0,
                "Good depth — hip crease at or just below parallel.", min_hip))
        elif min_hip <= 112:
            cat2_score -= 1.5
            cat2_checks.append(CheckResult("squat_depth", False, 1.5,
                "Slightly above parallel. Drive your hips a few inches lower.", min_hip))
        else:
            cat2_score -= 3.0
            cat2_checks.append(CheckResult("squat_depth", False, 3.0,
                "Depth too shallow — hip crease well above the knee. Focus on sitting down, not back.", min_hip))

    # 2b. Hip-knee balance — detects "sitting back" / good-morning pattern.
    #     Proper squat: hip ROM ≈ knee ROM. Hip-dominant: hip >> knee.
    hip_rom  = (max_angle(frame_poses, "left_hip")  or 0) - (robust_min_angle(frame_poses, "left_hip")  or 0)
    knee_rom = (max_angle(frame_poses, "left_knee") or 0) - (robust_min_angle(frame_poses, "left_knee") or 0)
    if hip_rom == 0:
        hip_rom  = (max_angle(frame_poses, "right_hip")  or 0) - (robust_min_angle(frame_poses, "right_hip")  or 0)
        knee_rom = (max_angle(frame_poses, "right_knee") or 0) - (robust_min_angle(frame_poses, "right_knee") or 0)

    if hip_rom > 10 and knee_rom > 0:
        ratio = hip_rom / (knee_rom + 1e-6)
        if ratio <= 1.5:
            cat2_checks.append(CheckResult("hip_knee_balance", True, 0,
                "Good balance of hip and knee flexion.", ratio))
        elif ratio <= 2.1:
            cat2_score -= 1.0
            cat2_checks.append(CheckResult("hip_knee_balance", False, 1.0,
                "Slight hip-dominance — let your knees travel forward more as you descend.", ratio))
        else:
            cat2_score -= 2.0
            cat2_checks.append(CheckResult("hip_knee_balance", False, 2.0,
                "Strong 'sitting back' pattern — hips hinging far more than knees bending. Think 'sit down' not 'sit back'.", ratio))

    cat2_score = max(0.0, cat2_score)
    categories.append(CategoryScore(
        name="Range of Motion",
        score=cat2_score,
        max_score=5.0,
        checks=cat2_checks,
        feedback=[c.feedback for c in cat2_checks if not c.passed],
    ))

    # ════════════════════════════════════════════════════════════════════════
    # CATEGORY 3 — NEUTRAL SPINE  (torso angle + butt wink)
    # ════════════════════════════════════════════════════════════════════════
    cat3_checks = []
    cat3_score  = 5.0

    # 3a. Spine/torso angle at deepest point.
    #     High-bar: upright ~140–165°. Low-bar: forward lean ~120–140°.
    #     Both are correct — only penalise genuine collapse. McGill (2010).
    min_spine = robust_min_angle(frame_poses, "spine", pct=10.0)
    if min_spine is not None:
        if min_spine >= 120:
            cat3_checks.append(CheckResult("spine_angle", True, 0,
                "Spine angle looks good — neutral or appropriate forward lean.", min_spine))
        elif min_spine >= 95:
            cat3_score -= 1.5
            cat3_checks.append(CheckResult("spine_angle", False, 1.5,
                "More chest collapse than ideal. Brace hard and drive your chest up out of the hole.", min_spine))
        else:
            cat3_score -= 3.0
            cat3_checks.append(CheckResult("spine_angle", False, 3.0,
                "Excessive spinal flexion — chest collapsing toward knees. Work on thoracic mobility and upper-back bracing.", min_spine))

    # 3b. Butt wink — posterior pelvic tilt at bottom. Hartmann et al. (2013).
    spine_at_mid    = angle_at_bottom(frame_poses, "left_knee", "spine")
    spine_at_bottom = robust_min_angle(frame_poses, "spine", pct=10.0)
    if spine_at_mid is not None and spine_at_bottom is not None:
        delta = spine_at_mid - spine_at_bottom
        if delta <= 18:
            cat3_checks.append(CheckResult("butt_wink", True, 0,
                "No significant posterior pelvic tilt detected.", delta))
        elif delta <= 30:
            cat3_score -= 1.0
            cat3_checks.append(CheckResult("butt_wink", False, 1.0,
                "Mild butt wink — pelvis slightly tucking under at depth. Improve hip flexor and ankle mobility.", delta))
        else:
            cat3_score -= 2.0
            cat3_checks.append(CheckResult("butt_wink", False, 2.0,
                "Significant butt wink — pelvis rounding under heavily. Address hip flexor tightness and ankle dorsiflexion.", delta))

    cat3_score = max(0.0, cat3_score)
    categories.append(CategoryScore(
        name="Neutral Spine",
        score=cat3_score,
        max_score=5.0,
        checks=cat3_checks,
        feedback=[c.feedback for c in cat3_checks if not c.passed],
    ))

    # ════════════════════════════════════════════════════════════════════════
    # CATEGORY 4 — CONTROLLED DESCENT  (eccentric tempo)
    # ════════════════════════════════════════════════════════════════════════
    cat4_checks = []
    cat4_score  = 5.0

    bottom_frame  = min(frame_poses, key=lambda fp: fp.angles.get("left_hip", 180))
    descent_time  = bottom_frame.timestamp - frame_poses[0].timestamp

    if descent_time > 0:
        if descent_time >= 1.2:
            cat4_checks.append(CheckResult("descent_tempo", True, 0,
                f"Excellent controlled descent ({descent_time:.1f}s).", descent_time))
        elif descent_time >= 0.7:
            cat4_checks.append(CheckResult("descent_tempo", True, 0,
                f"Decent descent tempo ({descent_time:.1f}s) — a touch slower would increase time under tension.", descent_time))
        elif descent_time >= 0.4:
            cat4_score -= 2.0
            cat4_checks.append(CheckResult("descent_tempo", False, 2.0,
                f"Descent too fast ({descent_time:.1f}s). Aim for at least 1s on the way down.", descent_time))
        else:
            cat4_score -= 4.0
            cat4_checks.append(CheckResult("descent_tempo", False, 4.0,
                f"Very fast / uncontrolled descent ({descent_time:.1f}s). Slow down — control the weight, don't let it control you.", descent_time))

    cat4_score = max(0.0, cat4_score)
    categories.append(CategoryScore(
        name="Controlled Descent",
        score=cat4_score,
        max_score=5.0,
        checks=cat4_checks,
        feedback=[c.feedback for c in cat4_checks if not c.passed],
    ))

    # ════════════════════════════════════════════════════════════════════════
    # CATEGORY 5 — LEG STANCE  (ankle width relative to hip width)
    # ════════════════════════════════════════════════════════════════════════
    # Ideal: ankles roughly shoulder/hip width or slightly wider.
    # Ratio = ankle_width / hip_width.
    #   ~0.9–1.4  → hip-width or slightly wider (conventional) ✓
    #   ~1.4–1.8  → wide stance / sumo — valid but flagged for awareness
    #   < 0.7     → very narrow — limits depth and increases knee stress
    # ════════════════════════════════════════════════════════════════════════
    cat5_checks = []
    cat5_score  = 5.0

    # Average over standing frames (top 20% of hip angle = most upright)
    stance_vals = [fp.angles["stance_width_ratio"] for fp in frame_poses
                   if "stance_width_ratio" in fp.angles]
    if stance_vals:
        # Use median — stance shouldn't change during a rep; median is noise-robust
        stance_ratio = float(np.median(stance_vals))
        if 0.85 <= stance_ratio <= 1.45:
            cat5_checks.append(CheckResult("stance_width", True, 0,
                f"Stance width looks good — ankles at roughly hip width ({stance_ratio:.2f}×).", stance_ratio))
        elif stance_ratio < 0.85:
            if stance_ratio < 0.65:
                cat5_score -= 3.0
                cat5_checks.append(CheckResult("stance_width", False, 3.0,
                    f"Stance is very narrow ({stance_ratio:.2f}× hip width). Widen your feet to at least hip-width to allow depth and reduce knee stress.", stance_ratio))
            else:
                cat5_score -= 1.5
                cat5_checks.append(CheckResult("stance_width", False, 1.5,
                    f"Stance is slightly narrow ({stance_ratio:.2f}× hip width). Try widening a little to improve depth.", stance_ratio))
        else:  # > 1.45
            if stance_ratio > 1.9:
                cat5_score -= 2.0
                cat5_checks.append(CheckResult("stance_width", False, 2.0,
                    f"Very wide stance ({stance_ratio:.2f}× hip width). Make sure this is intentional (sumo style) — excessive width can strain the groin.", stance_ratio))
            else:
                cat5_checks.append(CheckResult("stance_width", True, 0,
                    f"Wide stance ({stance_ratio:.2f}× hip width) — valid sumo-style or powerlifting squat.", stance_ratio))

    cat5_score = max(0.0, cat5_score)
    categories.append(CategoryScore(
        name="Leg Stance",
        score=cat5_score,
        max_score=5.0,
        checks=cat5_checks,
        feedback=[c.feedback for c in cat5_checks if not c.passed],
    ))

    # ════════════════════════════════════════════════════════════════════════
    # CATEGORY 6 — TOE FLARE  (foot angle relative to forward axis)
    # ════════════════════════════════════════════════════════════════════════
    # Ideal: 15–35° flare each foot. Too straight (<10°) restricts depth
    # and loads the knee medially. Too wide (>50°) causes hip stress.
    # We also check symmetry between left/right flare.
    # ════════════════════════════════════════════════════════════════════════
    cat6_checks = []
    cat6_score  = 5.0

    left_flare_vals  = [fp.angles["left_toe_flare"]  for fp in frame_poses if "left_toe_flare"  in fp.angles]
    right_flare_vals = [fp.angles["right_toe_flare"] for fp in frame_poses if "right_toe_flare" in fp.angles]

    if left_flare_vals and right_flare_vals:
        lf = float(np.median(left_flare_vals))
        rf = float(np.median(right_flare_vals))
        avg_flare = (lf + rf) / 2
        flare_diff = abs(lf - rf)

        # Angle range check
        if 15 <= avg_flare <= 40:
            cat6_checks.append(CheckResult("toe_flare_angle", True, 0,
                f"Good toe flare ({avg_flare:.0f}°) — allows depth without stressing the knee.", avg_flare))
        elif avg_flare < 15:
            cat6_score -= 2.0
            cat6_checks.append(CheckResult("toe_flare_angle", False, 2.0,
                f"Toes too straight ({avg_flare:.0f}°). Turn your feet out 20–30° to allow your hips to open and reach depth.", avg_flare))
        elif avg_flare <= 50:
            cat6_checks.append(CheckResult("toe_flare_angle", True, 0,
                f"Slightly wide toe flare ({avg_flare:.0f}°) — acceptable if hips are mobile.", avg_flare))
        else:
            cat6_score -= 2.0
            cat6_checks.append(CheckResult("toe_flare_angle", False, 2.0,
                f"Toes flared very wide ({avg_flare:.0f}°). Reduce flare to 20–40° to protect hip joints.", avg_flare))

        # Symmetry check
        if flare_diff <= 10:
            cat6_checks.append(CheckResult("toe_flare_symmetry", True, 0,
                f"Toe flare is symmetric (L:{lf:.0f}° R:{rf:.0f}°).", flare_diff))
        else:
            cat6_score -= 1.5
            cat6_checks.append(CheckResult("toe_flare_symmetry", False, 1.5,
                f"Asymmetric toe flare (L:{lf:.0f}° vs R:{rf:.0f}°). Even out foot position to avoid rotational stress on knees and hips.", flare_diff))

    cat6_score = max(0.0, cat6_score)
    categories.append(CategoryScore(
        name="Toe Flare",
        score=cat6_score,
        max_score=5.0,
        checks=cat6_checks,
        feedback=[c.feedback for c in cat6_checks if not c.passed],
    ))

    # ════════════════════════════════════════════════════════════════════════
    # BARBELL CHECKS  (only if bar detected in majority of frames)
    # ════════════════════════════════════════════════════════════════════════
    bar_detected_vals = [fp.angles.get("bar_detected", 0.0) for fp in frame_poses]
    barbell_present   = (sum(bar_detected_vals) / max(len(bar_detected_vals), 1)) >= 0.5

    if barbell_present:

        # ── BARBELL CATEGORY A — Bar Position (height on traps + level) ──
        catB_checks = []
        catB_score  = 5.0

        # A1. Bar height — wrist should be near or slightly above shoulder
        #     height (bar resting on traps). bar_height_ratio ≈ 0.95–1.20.
        bar_h_vals = [fp.angles["bar_height_ratio"] for fp in frame_poses
                      if "bar_height_ratio" in fp.angles]
        # Sample from standing frames (rep start/end) not the bottom
        standing_fps = frame_poses[:min(10, len(frame_poses))]
        bar_h_standing = [fp.angles["bar_height_ratio"] for fp in standing_fps
                          if "bar_height_ratio" in fp.angles]
        if bar_h_standing:
            bh = float(np.median(bar_h_standing))
            if 0.90 <= bh <= 1.25:
                catB_checks.append(CheckResult("bar_height", True, 0,
                    f"Bar sitting at correct trap height (ratio {bh:.2f}).", bh))
            elif bh < 0.90:
                catB_score -= 2.0
                catB_checks.append(CheckResult("bar_height", False, 2.0,
                    f"Bar appears to be sitting too high on the neck (ratio {bh:.2f}). Let it rest on the upper traps, not the cervical spine.", bh))
            else:
                catB_score -= 2.0
                catB_checks.append(CheckResult("bar_height", False, 2.0,
                    f"Bar may be sitting too low (ratio {bh:.2f}). For high-bar: rest on upper traps. For low-bar: rest on rear deltoids.", bh))

        # A2. Bar level (lateral tilt) — wrist height asymmetry.
        bar_off_vals = [fp.angles["bar_lateral_offset"] for fp in standing_fps
                        if "bar_lateral_offset" in fp.angles]
        if bar_off_vals:
            bo = float(np.median(bar_off_vals))
            if bo <= 0.10:
                catB_checks.append(CheckResult("bar_level", True, 0,
                    "Bar appears level across both shoulders.", bo))
            elif bo <= 0.20:
                catB_score -= 1.5
                catB_checks.append(CheckResult("bar_level", False, 1.5,
                    f"Bar is slightly tilted (offset {bo:.2f}). Check that you're gripping evenly and your shoulder heights are balanced.", bo))
            else:
                catB_score -= 3.0
                catB_checks.append(CheckResult("bar_level", False, 3.0,
                    f"Bar is noticeably tilted (offset {bo:.2f}). This creates asymmetric loading — check grip position, shoulder mobility, and weight distribution.", bo))

        catB_score = max(0.0, catB_score)
        categories.append(CategoryScore(
            name="Bar Position",
            score=catB_score,
            max_score=5.0,
            checks=catB_checks,
            feedback=[c.feedback for c in catB_checks if not c.passed],
        ))

        # ── BARBELL CATEGORY B — Bar Path (should stay over mid-foot) ────
        # We track horizontal wrist midpoint x-coordinate through the rep.
        # A straight bar path = low horizontal drift relative to rep depth.
        # Wretenberg et al. (1996).
        catC_checks = []
        catC_score  = 5.0

        wrist_x_vals = []
        for fp in frame_poses:
            lw_lm = fp.landmarks.get("LEFT_WRIST")
            rw_lm = fp.landmarks.get("RIGHT_WRIST")
            if lw_lm and rw_lm and lw_lm[3] > 0.3 and rw_lm[3] > 0.3:
                wrist_x_vals.append((lw_lm[0] + rw_lm[0]) / 2)

        if len(wrist_x_vals) >= 10:
            x_arr   = np.array(wrist_x_vals)
            x_range = float(x_arr.max() - x_arr.min())
            # Normalize by hip width to make it scale-independent
            hip_w_vals = [fp.angles.get("stance_width_ratio", 1.0) for fp in frame_poses
                          if "stance_width_ratio" in fp.angles]
            hip_w = float(np.median(hip_w_vals)) if hip_w_vals else 1.0
            # x_range is in normalized image coords; ~0.05 = very straight
            if x_range <= 0.04:
                catC_checks.append(CheckResult("bar_path", True, 0,
                    f"Bar path is very straight — minimal horizontal drift ({x_range:.3f}).", x_range))
            elif x_range <= 0.08:
                catC_checks.append(CheckResult("bar_path", True, 0,
                    f"Bar path is mostly straight (drift {x_range:.3f}). Minor adjustments could improve efficiency.", x_range))
            elif x_range <= 0.14:
                catC_score -= 2.0
                catC_checks.append(CheckResult("bar_path", False, 2.0,
                    f"Bar path has noticeable lateral drift ({x_range:.3f}). Keep the bar over your mid-foot — avoid letting it drift forward on the descent.", x_range))
            else:
                catC_score -= 4.0
                catC_checks.append(CheckResult("bar_path", False, 4.0,
                    f"Bar path is very uneven ({x_range:.3f}). Focus on keeping the bar directly above your mid-foot throughout the entire rep.", x_range))

        catC_score = max(0.0, catC_score)
        categories.append(CategoryScore(
            name="Bar Path",
            score=catC_score,
            max_score=5.0,
            checks=catC_checks,
            feedback=[c.feedback for c in catC_checks if not c.passed],
        ))

    # ════════════════════════════════════════════════════════════════════════
    # AGGREGATE — overall score (mean of all categories, reported 0–10)
    # ════════════════════════════════════════════════════════════════════════
    all_checks = [c for cat in categories for c in cat.checks]
    all_feedback = [c.feedback for c in all_checks if not c.passed]

    category_mean = float(np.mean([cat.score for cat in categories]))  # 0–5
    overall_score = round(category_mean * 2, 1)                        # 0–10

    raw_angles = {k: avg_angle(frame_poses, k) for k in
                  ["left_knee", "right_knee", "left_hip", "right_hip", "spine"]}

    return RepScore(
        exercise="squat",
        rep_number=rep_number,
        score=overall_score,
        checks=all_checks,
        feedback=all_feedback,
        raw_angles=raw_angles,
        categories=categories,
    )


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
