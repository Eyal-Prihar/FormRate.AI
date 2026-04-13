"""
FormRate.ai - Rep Detector
Segments pose sequences into individual reps using scipy prominence-based
peak detection. Prominence filtering ignores shallow noise/wobble and only
counts movements with significant range of motion.
"""

import numpy as np
from typing import List, Tuple
from scipy.signal import find_peaks, savgol_filter
from pose_analyzer import FramePose


# ── Minimum range of motion (degrees) to count as a rep ──────────────────
MIN_ROM = {
    "squat":    25,
    "deadlift": 30,
    "bench":    20,
}

# Minimum frames between valley PEAKS (at ~30fps).
# A full squat/deadlift rep takes ~1.5–3s → 45–90 frames.
# Using 45 frames (~1.5s) as the floor prevents double-counting within one rep.
MIN_REP_FRAMES = 45


def _get_signal(frame_poses: List[FramePose], keys: List[str]) -> Tuple[List[int], np.ndarray]:
    """
    Extract a smoothed angle signal from a list of FramePoses.
    Averages across provided keys (e.g. left + right hip).
    Returns (frame_indices, smoothed_values).
    """
    points = []
    for fp in frame_poses:
        vals = [fp.angles[k] for k in keys if k in fp.angles]
        if vals:
            points.append((fp.frame_idx, float(np.mean(vals))))

    if len(points) < 10:
        return [], np.array([])

    indices = [p[0] for p in points]
    raw = np.array([p[1] for p in points], dtype=float)

    # Use a larger Savitzky-Golay window to smooth out mid-rep wobble and
    # pose jitter. Window must be odd and ≥ 5. Cap at 31 frames (~1s at 30fps).
    n = len(raw)
    window = min(31, n if n % 2 == 1 else n - 1)
    if window >= 5:
        smoothed = savgol_filter(raw, window_length=window, polyorder=2)
    else:
        smoothed = raw

    return indices, smoothed


def _valleys_to_reps(
    indices: List[int],
    values: np.ndarray,
    min_rom: float,
) -> List[Tuple[int, int]]:
    """
    Find valleys (bottoms of reps) using prominence-based detection,
    then expand each valley outward to find rep start/end boundaries.

    Prominence = how much a valley stands out from surrounding signal.
    This naturally ignores wobble and only counts real reps.
    """
    if len(values) < 10:
        return []

    val_range = values.max() - values.min()
    if val_range < min_rom:
        print(f"[FormRate] Range of motion ({val_range:.1f}°) below minimum ({min_rom}°) — no reps detected.")
        return []

    # Invert signal so valleys become peaks (find_peaks works on peaks)
    inverted = -values

    # FIX 1: Raise prominence threshold from 40% → 65% of total ROM.
    # A real rep bottom must be significantly lower than the standing position.
    # This prevents small wobbles at the bottom or mid-descent pauses from
    # being counted as separate reps.
    prominence_threshold = val_range * 0.65

    # FIX 2: MIN_REP_FRAMES is now 45 (raised from 20) — see constant above.
    valley_positions, props = find_peaks(
        inverted,
        prominence=prominence_threshold,
        distance=MIN_REP_FRAMES,
    )

    if len(valley_positions) == 0:
        return []

    print(f"[FormRate] Found {len(valley_positions)} valley(s) with prominence >= {prominence_threshold:.1f}°")

    reps = []
    last_end_i = 0  # track where the previous rep ended

    for vp in valley_positions:
        # FIX 3: Raise the boundary walk threshold from 85% → 90% of max.
        # This ensures we only mark a rep as "started" or "ended" when the
        # lifter is genuinely close to standing, not just halfway up.
        top_threshold = values.max() - val_range * 0.10

        # Walk left from valley — but never go before the end of the last rep
        start_i = vp
        for i in range(vp, last_end_i - 1, -1):
            if values[i] >= top_threshold:
                start_i = i
                break

        # Walk right from valley until signal rises back near the top
        end_i = vp
        for i in range(vp, len(values)):
            if values[i] >= top_threshold:
                end_i = i
                break

        start_frame = indices[start_i]
        end_frame = indices[end_i]

        if end_frame - start_frame >= MIN_REP_FRAMES:
            reps.append((start_frame, end_frame))
            last_end_i = end_i  # next rep cannot start before here

    return reps


# ── Per-exercise detectors ────────────────────────────────────────────────

def detect_reps_squat(frame_poses: List[FramePose]) -> List[Tuple[int, int]]:
    indices, values = _get_signal(frame_poses, ["left_hip", "right_hip"])
    return _valleys_to_reps(indices, values, MIN_ROM["squat"])


def detect_reps_deadlift(frame_poses: List[FramePose]) -> List[Tuple[int, int]]:
    indices, values = _get_signal(frame_poses, ["left_hip", "right_hip"])
    return _valleys_to_reps(indices, values, MIN_ROM["deadlift"])


def detect_reps_bench(frame_poses: List[FramePose]) -> List[Tuple[int, int]]:
    indices, values = _get_signal(frame_poses, ["left_elbow", "right_elbow"])
    return _valleys_to_reps(indices, values, MIN_ROM["bench"])


DETECTOR_MAP = {
    "squat":    detect_reps_squat,
    "deadlift": detect_reps_deadlift,
    "bench":    detect_reps_bench,
}


def detect_reps(exercise: str, frame_poses: List[FramePose]) -> List[Tuple[int, int]]:
    fn = DETECTOR_MAP.get(exercise.lower())
    if fn is None:
        raise ValueError(f"Unknown exercise: {exercise}. Choose from: {list(DETECTOR_MAP.keys())}")
    reps = fn(frame_poses)
    print(f"[FormRate] Rep boundaries: {reps}")
    return reps
