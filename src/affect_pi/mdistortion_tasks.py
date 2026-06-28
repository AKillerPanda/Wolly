"""Shared MediaPipe **Tasks** FaceLandmarker feature extractor.

This is the single source of truth for turning a frame/image into the three
Mdistortion matrices, so the **live demo** and the **trainer** compute features
identically (any drift between them makes a classifier meaningless).

It reuses the exact feature builder from ``train_mdistortion_de`` (``sample_feature``
-> three flattened pairwise matrices for mouth / left eye / right eye) but sources
landmarks from the Tasks ``FaceLandmarker`` API, which is the one available on
this Python 3.14 build (the legacy ``mediapipe.solutions`` is not).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .train_mdistortion_de import DetectedFace, sample_feature

DEFAULT_MODEL = Path("models/face_landmarker.task")

# Stable scale landmarks (cheek-to-cheek) for interpretable live metrics.
LEFT_CHEEK_ID = 234
RIGHT_CHEEK_ID = 454


def load_landmarker(model_path: Path = DEFAULT_MODEL, video: bool = False, num_faces: int = 1):
    """Create a FaceLandmarker in IMAGE (training) or VIDEO (live) running mode."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"FaceLandmarker model not found: {model_path}. "
            "Download face_landmarker.task into models/ first."
        )
    mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mode,
        num_faces=num_faces,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _pts_from_result(result, w: int, h: int) -> np.ndarray | None:
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)


def detect_image(landmarker, bgr: np.ndarray) -> np.ndarray | None:
    """Landmarks (478, 2) in pixel coords for a still image, or None."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return _pts_from_result(result, w, h)


def detect_video(landmarker, bgr: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
    """Landmarks (478, 2) for a video frame (needs increasing timestamps), or None."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
    )
    return _pts_from_result(result, w, h)


def face_from_pts(pts: np.ndarray, label: str = "?", image_path: Path = Path("live")) -> DetectedFace:
    """Wrap raw landmark points as a DetectedFace.

    The bounding box is recorded as ``face_width/height`` for reference, but the
    Mdistortion feature transform (``warp_group``) is centroid-relative and
    scale-invariant, so these values do not affect the features -- training and
    live share the identical normalization regardless.
    """
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    fw = float(max(mx[0] - mn[0], 1.0))
    fh = float(max(mx[1] - mn[1], 1.0))
    landmarks = {i: pts[i].astype(np.float32) for i in range(pts.shape[0])}
    return DetectedFace(image_path=image_path, label=label, landmarks=landmarks, face_width=fw, face_height=fh)


def feature_from_pts(pts: np.ndarray, offsets: np.ndarray) -> np.ndarray | None:
    """The exact 3-matrix Mdistortion feature vector used for training."""
    return sample_feature(face_from_pts(pts), offsets)
