from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LandmarkCluster:
    name: str
    indices: tuple[int, ...]

    def extract(self, landmarks: np.ndarray) -> np.ndarray:
        max_idx = landmarks.shape[0] - 1
        usable = [i for i in self.indices if i <= max_idx]
        if not usable:
            return np.zeros((0, 3), dtype=np.float32)
        return landmarks[np.array(usable, dtype=np.int32)]


# MediaPipe FaceMesh indices. These are deliberately easy to edit.
# FaceMesh has 468 landmarks; with refine_landmarks=True it adds iris landmarks.
FOREHEAD = LandmarkCluster(
    "forehead",
    (10, 109, 67, 103, 54, 21, 162, 127, 338, 297, 332, 284, 251, 389, 356),
)

LEFT_EYE_SKIN = LandmarkCluster(
    "left_eye_skin",
    (33, 133, 160, 159, 158, 144, 145, 153, 70, 63, 105, 66, 107),
)

RIGHT_EYE_SKIN = LandmarkCluster(
    "right_eye_skin",
    (362, 263, 387, 386, 385, 373, 374, 380, 336, 296, 334, 293, 300),
)

# Iris landmarks are available only when MediaPipe FaceMesh refine_landmarks=True.
LEFT_PUPIL = LandmarkCluster("left_pupil", (474, 475, 476, 477))
RIGHT_PUPIL = LandmarkCluster("right_pupil", (469, 470, 471, 472))

MOUTH = LandmarkCluster(
    "mouth",
    (
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        185, 40, 39, 37, 0, 267, 269, 270, 409,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ),
)

DEFAULT_FACE_CLUSTERS: tuple[LandmarkCluster, ...] = (
    FOREHEAD,
    LEFT_EYE_SKIN,
    RIGHT_EYE_SKIN,
    LEFT_PUPIL,
    RIGHT_PUPIL,
    MOUTH,
)


def extract_clusters(
    landmarks: np.ndarray,
    clusters: Iterable[LandmarkCluster] = DEFAULT_FACE_CLUSTERS,
) -> dict[str, np.ndarray]:
    return {cluster.name: cluster.extract(landmarks) for cluster in clusters}
