from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FaceMeshDetector:
    max_num_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    refine_landmarks: bool = True

    def __post_init__(self) -> None:
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise RuntimeError("mediapipe is required for FaceMeshDetector") from exc

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def detect(self, bgr_frame: np.ndarray) -> np.ndarray | None:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        h, w = bgr_frame.shape[:2]
        face = result.multi_face_landmarks[0]
        coords = np.array(
            [[lm.x * w, lm.y * h, lm.z * w] for lm in face.landmark],
            dtype=np.float32,
        )
        return coords

    def close(self) -> None:
        self._face_mesh.close()


@dataclass
class PoseDetector:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_visibility: float = 0.5

    def __post_init__(self) -> None:
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise RuntimeError("mediapipe is required for PoseDetector") from exc

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmark_names = [lm.name for lm in self._mp_pose.PoseLandmark]

    def detect(self, bgr_frame: np.ndarray) -> dict[str, tuple[float, float, float, float]] | None:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None

        h, w = bgr_frame.shape[:2]
        visible: dict[str, tuple[float, float, float, float]] = {}
        for idx, lm in enumerate(result.pose_landmarks.landmark):
            if lm.visibility >= self.min_visibility:
                name = self._landmark_names[idx]
                visible[name] = (lm.x * w, lm.y * h, lm.z * w, lm.visibility)
        return visible or None

    def close(self) -> None:
        self._pose.close()
