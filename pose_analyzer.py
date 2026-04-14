"""
FormRate.ai - Pose Analyzer
Extracts joint angles and keypoints from video frames using MediaPipe.
Compatible with MediaPipe 0.10.30+ (Tasks API).
"""

import cv2
import numpy as np
import urllib.request
import os
from dataclasses import dataclass
from typing import Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import landmark as mp_landmark


# ── Model download ────────────────────────────────────────────────────────
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[FormRate] Downloading MediaPipe pose model (~7MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[FormRate] Model downloaded.")

# Landmark name -> index mapping (MediaPipe Pose 33 landmarks)
LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]
LANDMARK_INDEX = {name: i for i, name in enumerate(LANDMARK_NAMES)}


# ─────────────────────────── Data Structures ─────────────────────────────

@dataclass
class FramePose:
    landmarks: dict  # landmark_name -> (x, y, z, visibility)
    angles: dict     # angle_name -> degrees
    frame_idx: int
    timestamp: float


# ─────────────────────────── Geometry ────────────────────────────────────

def get_landmark_coords(landmarks: dict, name: str) -> Optional[np.ndarray]:
    lm = landmarks.get(name)
    if lm is None or lm[3] < 0.3:
        return None
    return np.array([lm[0], lm[1], lm[2]])


def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    a, b, c = a[:2], b[:2], c[:2]
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def extract_angles(lm: dict) -> dict:
    angles = {}

    def angle(name, a, b, c):
        pa = get_landmark_coords(lm, a)
        pb = get_landmark_coords(lm, b)
        pc = get_landmark_coords(lm, c)
        if pa is not None and pb is not None and pc is not None:
            angles[name] = calc_angle(pa, pb, pc)

    # Lower body
    angle("left_knee",  "LEFT_HIP",       "LEFT_KNEE",  "LEFT_ANKLE")
    angle("right_knee", "RIGHT_HIP",      "RIGHT_KNEE", "RIGHT_ANKLE")
    angle("left_hip",   "LEFT_SHOULDER",  "LEFT_HIP",   "LEFT_KNEE")
    angle("right_hip",  "RIGHT_SHOULDER", "RIGHT_HIP",  "RIGHT_KNEE")

    # Upper body
    angle("left_elbow",    "LEFT_SHOULDER",  "LEFT_ELBOW",  "LEFT_WRIST")
    angle("right_elbow",   "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST")
    angle("left_shoulder",  "LEFT_ELBOW",    "LEFT_SHOULDER", "LEFT_HIP")
    angle("right_shoulder", "RIGHT_ELBOW",   "RIGHT_SHOULDER", "RIGHT_HIP")

    # Spine
    ls = get_landmark_coords(lm, "LEFT_SHOULDER")
    rs = get_landmark_coords(lm, "RIGHT_SHOULDER")
    lh = get_landmark_coords(lm, "LEFT_HIP")
    rh = get_landmark_coords(lm, "RIGHT_HIP")
    lk = get_landmark_coords(lm, "LEFT_KNEE")
    rk = get_landmark_coords(lm, "RIGHT_KNEE")
    if all(v is not None for v in [ls, rs, lh, rh, lk, rk]):
        angles["spine"] = calc_angle((ls + rs) / 2, (lh + rh) / 2, (lk + rk) / 2)

    # Knee valgus ratio
    for side in ("LEFT", "RIGHT"):
        hip   = get_landmark_coords(lm, f"{side}_HIP")
        knee  = get_landmark_coords(lm, f"{side}_KNEE")
        ankle = get_landmark_coords(lm, f"{side}_ANKLE")
        if hip is not None and knee is not None and ankle is not None:
            hip_ankle_x = ankle[0] - hip[0]
            if abs(hip_ankle_x) > 1e-4:
                angles[f"{side.lower()}_knee_valgus_ratio"] = (knee[0] - hip[0]) / hip_ankle_x

    # ── Stance width ratio ────────────────────────────────────────────────
    # Ratio of hip-to-hip distance vs ankle-to-ankle distance (normalized
    # to image width). >1.0 = wider than hips (sumo-ish), ~1.0 = hip width,
    # <0.7 = narrow/close stance.
    lhip  = get_landmark_coords(lm, "LEFT_HIP")
    rhip  = get_landmark_coords(lm, "RIGHT_HIP")
    lank  = get_landmark_coords(lm, "LEFT_ANKLE")
    rank  = get_landmark_coords(lm, "RIGHT_ANKLE")
    if all(v is not None for v in [lhip, rhip, lank, rank]):
        hip_width   = abs(lhip[0] - rhip[0]) + 1e-6
        ankle_width = abs(lank[0] - rank[0])
        angles["stance_width_ratio"] = ankle_width / hip_width

    # ── Toe flare angle ───────────────────────────────────────────────────
    # Angle between heel→toe_index vector and the forward (vertical) axis.
    # 0° = toes pointing straight forward, 30–45° = typical squat flare.
    # We compute per foot and store both.
    for side, heel_name, toe_name in [
        ("left",  "LEFT_HEEL",  "LEFT_FOOT_INDEX"),
        ("right", "RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
    ]:
        heel = get_landmark_coords(lm, heel_name)
        toe  = get_landmark_coords(lm, toe_name)
        if heel is not None and toe is not None:
            dx = toe[0] - heel[0]
            dy = toe[1] - heel[1]   # positive = downward in image coords
            # Angle relative to the image vertical axis (pointing down = 0°)
            toe_angle = float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))
            angles[f"{side}_toe_flare"] = toe_angle

    # ── Bar position (wrist–shoulder–hip alignment) ───────────────────────
    # Proxy for barbell placement: if wrists are near shoulder height and
    # close together behind the neck → high-bar or low-bar squat.
    # We store:
    #   bar_detected       : 1.0 if wrists are near trap/shoulder region, else 0.0
    #   bar_height_ratio   : wrist_y / shoulder_y  (~1.0 = on traps, <0.8 = arms raised)
    #   bar_lateral_offset : abs difference in wrist x-coords (normalized by shoulder width)
    #                        small = bar level/even, large = bar tilted
    lw = get_landmark_coords(lm, "LEFT_WRIST")
    rw = get_landmark_coords(lm, "RIGHT_WRIST")
    ls = get_landmark_coords(lm, "LEFT_SHOULDER")
    rs = get_landmark_coords(lm, "RIGHT_SHOULDER")
    if all(v is not None for v in [lw, rw, ls, rs]):
        avg_wrist_y    = (lw[1] + rw[1]) / 2
        avg_shoulder_y = (ls[1] + rs[1]) / 2
        shoulder_width = abs(ls[0] - rs[0]) + 1e-6
        wrist_width    = abs(lw[0] - rw[0])
        bar_height_ratio   = avg_wrist_y / (avg_shoulder_y + 1e-6)
        bar_lateral_offset = abs(lw[1] - rw[1]) / shoulder_width  # vertical asymmetry

        # Barbell detected if wrists are roughly at or below shoulder height
        # and arms are wider than shoulder width (hands gripping bar out wide)
        bar_detected = 1.0 if (
            0.8 <= bar_height_ratio <= 1.35 and
            wrist_width >= shoulder_width * 0.8
        ) else 0.0

        angles["bar_detected"]        = bar_detected
        angles["bar_height_ratio"]    = bar_height_ratio
        angles["bar_lateral_offset"]  = bar_lateral_offset

    return angles


def landmarks_to_dict(pose_landmarks) -> dict:
    """Convert MediaPipe Tasks NormalizedLandmark list to plain dict."""
    result = {}
    for i, lm in enumerate(pose_landmarks):
        name = LANDMARK_NAMES[i]
        result[name] = (lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0)
    return result


# ─────────────────────────── Annotator ───────────────────────────────────

# Connections between landmarks for drawing skeleton
POSE_CONNECTIONS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
]

def draw_skeleton(frame: np.ndarray, lm_dict: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    for a, b in POSE_CONNECTIONS:
        pa = get_landmark_coords(lm_dict, a)
        pb = get_landmark_coords(lm_dict, b)
        if pa is not None and pb is not None:
            pt1 = (int(pa[0] * w), int(pa[1] * h))
            pt2 = (int(pb[0] * w), int(pb[1] * h))
            cv2.line(out, pt1, pt2, (0, 255, 0), 2)
    for name in LANDMARK_NAMES:
        lm = get_landmark_coords(lm_dict, name)
        if lm is not None:
            pt = (int(lm[0] * w), int(lm[1] * h))
            cv2.circle(out, pt, 4, (0, 0, 255), -1)
    return out


# ─────────────────────────── Main Analyzer ───────────────────────────────

class PoseAnalyzer:
    def __init__(self, model_complexity: int = 1):
        _ensure_model()
        self._options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _make_landmarker(self):
        return mp_vision.PoseLandmarker.create_from_options(self._options)

    def process_video(self, video_path: str, annotate: bool = False):
        # Create a fresh landmarker for each video — the landmarker cannot
        # be reused after close() so we instantiate a new one every call.
        landmarker = self._make_landmarker()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_poses = []
        annotated_frames = [] if annotate else None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lm_dict = landmarks_to_dict(result.pose_landmarks[0])
                angles = extract_angles(lm_dict)
                fp = FramePose(lm_dict, angles, frame_idx, frame_idx / fps)
                frame_poses.append(fp)
                if annotate:
                    annotated_frames.append(draw_skeleton(frame, lm_dict))
            elif annotate:
                annotated_frames.append(frame)

            frame_idx += 1

        cap.release()
        landmarker.close()
        return frame_poses, annotated_frames
