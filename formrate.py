"""
FormRate.ai - Main Analyzer
Entry point: analyze a video for a given exercise and return a full form report.

Usage:
    from formrate import FormRateAnalyzer
    analyzer = FormRateAnalyzer()
    report = analyzer.analyze("my_squat.mp4", exercise="squat")
    print(report.summary())

Or via CLI:
    python formrate.py --video my_squat.mp4 --exercise squat
    python formrate.py --url "https://youtube.com/..." --exercise squat
"""

import json
import argparse
import subprocess
import tempfile
import os
from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np
from pose_analyzer import PoseAnalyzer, FramePose
from rep_detector import detect_reps
from scoring_engine import score_rep, RepScore


# ─────────────────────────── URL Helpers ───────────────────────────────────

def is_url(path: str) -> bool:
    return any(x in path for x in ["http://", "https://", "youtube.com", "youtu.be"])


def fetch_url_to_temp(url: str) -> str:
    """
    Download a YouTube/web video to a temp directory using yt-dlp.
    Uses --no-check-certificate to bypass Mac SSL issues.
    Returns the actual downloaded file path.
    """
    tmp_dir = tempfile.mkdtemp()
    output_template = os.path.join(tmp_dir, "formrate_dl.%(ext)s")

    print("[FormRate] Fetching video from URL (this may take a moment)...")
    cmd = [
        "yt-dlp",
        "--no-check-certificate",
        "-f", "bestvideo[height<=720]+bestaudio/bestvideo[height<=720]/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed.\nError: {result.stderr.strip()}\n"
            f"Make sure yt-dlp is installed: pip3 install yt-dlp"
        )

    # Find the video file in the temp dir — skip audio-only files
    video_exts = (".mp4", ".mkv", ".webm", ".mov", ".avi")
    files = [
        os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
        if os.path.splitext(f)[1].lower() in video_exts
    ]
    if not files:
        # Last resort: grab whatever is there
        files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
    if not files:
        raise RuntimeError("yt-dlp ran but no output file was found.")

    actual_path = files[0]
    print(f"[FormRate] Video ready: {actual_path}")
    return actual_path


# ─────────────────────────── Report ────────────────────────────────────────

@dataclass
class FormReport:
    exercise: str
    video_path: str
    total_reps: int
    rep_scores: List[RepScore]
    overall_score: float
    overall_feedback: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 55,
            f"  FormRate.ai — {self.exercise.upper()} Analysis",
            "=" * 55,
            f"  Video        : {self.video_path}",
            f"  Reps Detected: {self.total_reps}",
            f"  Overall Score: {self.overall_score:.1f} / 10",
            "-" * 55,
        ]
        for rs in self.rep_scores:
            lines.append(f"  Rep {rs.rep_number:>2}  →  Score: {rs.score:.1f}/10")
            for check in rs.checks:
                status = "✓" if check.passed else "✗"
                val_str = f" [{check.value:.1f}°]" if check.value is not None else ""
                lines.append(f"         {status} {check.name}{val_str}")
            if rs.feedback:
                lines.append(f"         💬 {rs.feedback[0]}")
            lines.append("")

        lines.append("-" * 55)
        lines.append("  Top Coaching Cues:")
        for cue in self.overall_feedback[:3]:
            lines.append(f"    • {cue}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "exercise": self.exercise,
            "video_path": self.video_path,
            "total_reps": self.total_reps,
            "overall_score": self.overall_score,
            "overall_feedback": self.overall_feedback,
            "rep_scores": [asdict(r) for r in self.rep_scores],
        }, indent=2)


# ─────────────────────────── Video Validation ─────────────────────────────

def validate_exercise_video(
    frame_poses: List[FramePose],
    total_frames: int,
    exercise: str,
) -> Optional[str]:
    """
    Check whether the video actually contains a weightlifting exercise.
    Returns None if the video looks valid, or an error message string if not.

    Checks:
      1. Person detected in enough frames.
      2. Key landmarks have sufficient visibility (not hallucinated).
      3. Angle ranges are physically plausible (not noisy garbage).
      4. Standing posture detected at some point (squat/deadlift).
      5. Smooth, repetitive signal — not random noise.
    """
    if total_frames == 0:
        return "The video appears to be empty or unreadable."

    # ── Check 1: Person detection rate ────────────────────────────────────
    detection_rate = len(frame_poses) / total_frames
    if detection_rate < 0.15:
        return (
            "Could not detect a person in this video. "
            "Make sure your full body is visible and the lighting is adequate."
        )

    # ── Check 2: Key landmark visibility ──────────────────────────────────
    # Real exercise video = key landmarks consistently visible (>0.5).
    # Hallucinated poses from partial views have low visibility on the
    # joints the model is guessing at.
    REQUIRED_LANDMARKS = {
        "squat":    ["LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"],
        "deadlift": ["LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"],
        "bench":    ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"],
    }
    required = REQUIRED_LANDMARKS.get(exercise, [])
    low_visibility_count = 0
    for lm_name in required:
        vis = [fp.landmarks[lm_name][3] for fp in frame_poses if lm_name in fp.landmarks]
        if not vis or np.mean(vis) < 0.5:
            low_visibility_count += 1

    # If more than half of required landmarks have poor visibility, reject
    if low_visibility_count > len(required) // 2:
        return (
            "Your full body isn't clearly visible in this video. "
            f"For {exercise} analysis, the camera needs to capture your "
            "entire body — hips, knees, and ankles must all be in frame. "
            "Try filming from the side at hip height."
        )

    # ── Check 3: Angle range sanity ───────────────────────────────────────
    # Real exercises have a controlled ROM: squats ~50-90° hip ROM,
    # deadlifts ~40-80°, bench ~40-80° elbow ROM. If the range is
    # >130° it's noise from bad pose estimation, not a real movement.
    ANGLE_KEYS = {
        "squat":    ["left_hip", "right_hip"],
        "deadlift": ["left_hip", "right_hip"],
        "bench":    ["left_elbow", "right_elbow"],
    }
    for key in ANGLE_KEYS.get(exercise, []):
        vals = [fp.angles[key] for fp in frame_poses if key in fp.angles]
        if len(vals) >= 10:
            rom = max(vals) - min(vals)
            if rom > 130:
                return (
                    "This doesn't appear to be an exercise video — the pose "
                    "data is too noisy for analysis. Make sure the camera has "
                    "a clear, stable view of your full body throughout the movement."
                )

    # ── Check 4: Standing posture detection (squat/deadlift) ──────────────
    # At some point in a squat or deadlift, the person must be standing
    # upright: hip angle > 150°. If we never see this, it's not a squat.
    if exercise in ("squat", "deadlift"):
        hip_vals = [fp.angles.get("left_hip") or fp.angles.get("right_hip")
                    for fp in frame_poses]
        hip_vals = [v for v in hip_vals if v is not None]
        if hip_vals:
            # Use 90th percentile to be robust to outliers
            standing_angle = np.percentile(hip_vals, 90)
            if standing_angle < 140:
                return (
                    f"This doesn't look like a {exercise} — no standing posture "
                    "was detected. For squat or deadlift analysis, the video "
                    "should show you standing upright at some point during the movement."
                )

    # ── Check 5: Signal smoothness ────────────────────────────────────────
    # Real exercise = smooth angle curve. Random/bad pose = frame-to-frame
    # jumps. Check ALL relevant joints, not just primary ones. If most are
    # noisy, this isn't a clean single-person exercise video.
    ALL_JOINTS = {
        "squat":    ["left_hip", "right_hip", "left_knee", "right_knee"],
        "deadlift": ["left_hip", "right_hip", "left_knee", "right_knee"],
        "bench":    ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
    }
    noisy_joints = 0
    checked_joints = 0
    for key in ALL_JOINTS.get(exercise, []):
        vals = [fp.angles[key] for fp in frame_poses if key in fp.angles]
        if len(vals) >= 20:
            checked_joints += 1
            diffs = np.abs(np.diff(vals))
            median_diff = np.median(diffs)
            if median_diff > 5.0:
                noisy_joints += 1

    if checked_joints > 0 and noisy_joints * 2 >= checked_joints:
        return (
            "The pose tracking in this video is too unstable for "
            "accurate analysis. This usually means the video isn't "
            "showing a clear exercise movement, or the camera angle "
            "doesn't capture your full body well. Try filming from "
            "the side with your whole body in frame."
        )

    # ── Check 6: Spatial consistency ──────────────────────────────────────
    # In group shots or TikToks with cuts, MediaPipe jumps between different
    # people causing the tracked hip position (x,y) to teleport. In a real
    # exercise, the person stays roughly in the same location.
    hip_positions = []
    for fp in frame_poses:
        lh = fp.landmarks.get("LEFT_HIP")
        rh = fp.landmarks.get("RIGHT_HIP")
        if lh and rh:
            mid_x = (lh[0] + rh[0]) / 2
            mid_y = (lh[1] + rh[1]) / 2
            hip_positions.append((mid_x, mid_y))

    if len(hip_positions) >= 20:
        xs = [p[0] for p in hip_positions]
        ys = [p[1] for p in hip_positions]
        # Check 1: Large sudden jumps (>15% of frame) = person switching
        x_jumps = np.abs(np.diff(xs))
        y_jumps = np.abs(np.diff(ys))
        big_jumps = np.sum((x_jumps > 0.15) | (y_jumps > 0.15))
        jump_rate = big_jumps / len(x_jumps)
        if jump_rate > 0.10:
            return (
                "This video appears to contain multiple people or camera cuts, "
                "which makes exercise analysis unreliable. Please upload a video "
                "of a single person performing the exercise with a steady camera."
            )
        # Check 2: Position spread — in a real exercise, the person stays
        # roughly in the same spot. Std > 0.15 in x or y means the tracked
        # "person" is moving across the whole frame (multiple people / panning).
        if np.std(xs) > 0.15 or np.std(ys) > 0.15:
            return (
                "The tracked body position varies too much across the video. "
                "This usually means multiple people are in frame or the camera "
                "is moving too much. Film a single person from a fixed camera angle."
            )

    # All checks passed
    return None


# ─────────────────────────── Analyzer ──────────────────────────────────────

class FormRateAnalyzer:
    def __init__(self, model_complexity: int = 1):
        self.pose_analyzer = PoseAnalyzer(model_complexity=model_complexity)

    def analyze(
        self,
        video_path: str,
        exercise: str,
        annotate_output: Optional[str] = None,
    ) -> FormReport:
        print(f"[FormRate] Processing: {video_path}")
        print(f"[FormRate] Exercise:   {exercise}")

        # Handle URLs — download to temp file first
        tmp_file = None
        if is_url(video_path):
            tmp_file = fetch_url_to_temp(video_path)
            actual_path = tmp_file
        else:
            actual_path = video_path

        try:
            annotate = annotate_output is not None
            frame_poses, annotated_frames = self.pose_analyzer.process_video(actual_path, annotate=annotate)
            print(f"[FormRate] Extracted poses from {len(frame_poses)} frames")

            if annotate and annotated_frames and annotate_output:
                self._save_video(actual_path, annotated_frames, annotate_output)

            # Count total frames for validation
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(actual_path)
            total_frames = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            _cap.release()

            if not frame_poses:
                print("[FormRate] WARNING: No pose data detected.")
                return FormReport(exercise, video_path, 0, [], 0.0, ["No pose detected in video."])

            # ── Validate that this actually looks like an exercise ────────
            validation_error = validate_exercise_video(frame_poses, total_frames, exercise)
            if validation_error:
                print(f"[FormRate] Validation failed: {validation_error}")
                return FormReport(exercise, video_path, 0, [], 0.0, [validation_error])

            rep_boundaries = detect_reps(exercise, frame_poses)
            print(f"[FormRate] Detected {len(rep_boundaries)} rep(s)")

            if not rep_boundaries:
                print("[FormRate] Falling back to whole-video analysis")
                rep_boundaries = [(frame_poses[0].frame_idx, frame_poses[-1].frame_idx)]

            frame_lookup = {fp.frame_idx: fp for fp in frame_poses}

            rep_scores: List[RepScore] = []
            for i, (start, end) in enumerate(rep_boundaries):
                rep_frames = [frame_lookup[idx] for idx in range(start, end + 1) if idx in frame_lookup]
                if len(rep_frames) < 5:
                    continue
                rs = score_rep(exercise, rep_frames, rep_number=i + 1)
                rep_scores.append(rs)
                print(f"[FormRate] Rep {i+1}: {rs.score}/10")

            if not rep_scores:
                return FormReport(exercise, video_path, 0, [], 0.0, ["Could not score any reps."])

            overall_score = round(sum(r.score for r in rep_scores) / len(rep_scores), 1)
            cue_freq: dict = {}
            for rs in rep_scores:
                for cue in rs.feedback:
                    cue_freq[cue] = cue_freq.get(cue, 0) + 1
            overall_feedback = sorted(cue_freq, key=lambda k: -cue_freq[k])

            return FormReport(
                exercise=exercise,
                video_path=video_path,
                total_reps=len(rep_scores),
                rep_scores=rep_scores,
                overall_score=overall_score,
                overall_feedback=overall_feedback,
            )

        finally:
            # Always clean up temp file
            if tmp_file and os.path.exists(tmp_file):
                import shutil
                shutil.rmtree(os.path.dirname(tmp_file), ignore_errors=True)
                print("[FormRate] Temp file cleaned up.")

    def _save_video(self, original_path: str, frames, output_path: str):
        import cv2
        cap = cv2.VideoCapture(original_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        h, w = frames[0].shape[:2]
        cap.release()
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"[FormRate] Annotated video saved to: {output_path}")


# ─────────────────────────── CLI ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FormRate.ai — AI gym form analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python formrate.py --video squat.mp4 --exercise squat
  python formrate.py --url "https://youtube.com/shorts/abc123" --exercise squat
  python formrate.py --video deadlift.mp4 --exercise deadlift --save-json report.json
  python formrate.py --video bench.mp4 --exercise bench --annotate annotated.mp4
        """
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Path to a local video file")
    source.add_argument("--url", help="YouTube or direct video URL (no download needed)")

    parser.add_argument("--exercise", required=True, choices=["squat", "deadlift", "bench"])
    parser.add_argument("--save-json", metavar="FILE", help="Save JSON report to file")
    parser.add_argument("--annotate", metavar="FILE", help="Save pose-annotated video to file")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    args = parser.parse_args()

    video_input = args.url if args.url else args.video

    analyzer = FormRateAnalyzer(model_complexity=args.model_complexity)
    report = analyzer.analyze(
        video_path=video_input,
        exercise=args.exercise,
        annotate_output=args.annotate,
    )

    print(report.summary())

    if args.save_json:
        with open(args.save_json, "w") as f:
            f.write(report.to_json())
        print(f"\n[FormRate] JSON report saved to: {args.save_json}")


if __name__ == "__main__":
    main()