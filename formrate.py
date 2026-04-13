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

            if not frame_poses:
                print("[FormRate] WARNING: No pose data detected.")
                return FormReport(exercise, video_path, 0, [], 0.0, ["No pose detected in video."])

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
