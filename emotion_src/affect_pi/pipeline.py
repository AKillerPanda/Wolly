from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .detectors import FaceMeshDetector, PoseDetector
from .features import FacialDistortions, build_facial_distortions
from .model import EmotionModel, EmotionPrediction
from .status import VisionStatus
from .trend import GaussianTrendLayer, GaussianTrendState


@dataclass
class FrameResult:
    status: VisionStatus
    facial_distortions: FacialDistortions | None = None
    emotion: EmotionPrediction | None = None
    trend: GaussianTrendState | None = None
    body_parts: dict[str, tuple[float, float, float, float]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionPipeline:
    face_detector: FaceMeshDetector
    pose_detector: PoseDetector
    emotion_model: EmotionModel
    trend_layer: GaussianTrendLayer

    def process(self, bgr_frame: np.ndarray) -> FrameResult:
        face_landmarks = self.face_detector.detect(bgr_frame)
        if face_landmarks is not None:
            facial_distortions = build_facial_distortions(face_landmarks)
            feature_vector = facial_distortions.feature_vector()
            emotion = self.emotion_model.predict(feature_vector)
            trend = self.trend_layer.update(feature_vector)
            return FrameResult(
                status=VisionStatus.FACE_VISIBLE,
                facial_distortions=facial_distortions,
                emotion=emotion,
                trend=trend,
                meta={
                    "n_face_landmarks": int(face_landmarks.shape[0]),
                    "feature_dim": int(feature_vector.size),
                },
            )

        body_parts = self.pose_detector.detect(bgr_frame)
        if body_parts:
            return FrameResult(
                status=VisionStatus.BODY_VISIBLE,
                body_parts=body_parts,
                meta={"n_body_parts": len(body_parts)},
            )

        return FrameResult(status=VisionStatus.CANT_SEE)

    def close(self) -> None:
        self.face_detector.close()
        self.pose_detector.close()
