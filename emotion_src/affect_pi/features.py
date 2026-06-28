from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clusters import extract_clusters
from .geometry import (
    cluster_scale,
    eye_aspect_ratio,
    flatten_matrices,
    iris_diameter_ratio,
    mouth_open_ratio,
    pairwise_angle_matrix,
    pairwise_distance_matrix,
)


@dataclass
class FacialDistortions:
    """Container for clustered facial geometry features.

    `matrices` contains entries like:
      - forehead.distance
      - forehead.angle
      - mouth.distance
      - mouth.angle

    `scalars` contains compact metrics that are useful for the model/trend layer.
    """

    matrices: dict[str, np.ndarray] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)

    def feature_vector(self) -> np.ndarray:
        matrix_vec = flatten_matrices(self.matrices)
        scalar_vec = np.array(
            [self.scalars[k] for k in sorted(self.scalars)], dtype=np.float32
        )
        if scalar_vec.size == 0:
            return matrix_vec
        if matrix_vec.size == 0:
            return scalar_vec
        return np.concatenate([matrix_vec, scalar_vec]).astype(np.float32)

    def compact_matrix(self) -> np.ndarray:
        """Small matrix for logging/trends: one row per scalar metric."""
        if not self.scalars:
            return np.zeros((0, 1), dtype=np.float32)
        return np.array([[self.scalars[k]] for k in sorted(self.scalars)], dtype=np.float32)


def build_facial_distortions(landmarks: np.ndarray) -> FacialDistortions:
    clusters = extract_clusters(landmarks)
    matrices: dict[str, np.ndarray] = {}
    scalars: dict[str, float] = {}

    for name, points in clusters.items():
        if len(points) < 2:
            continue
        matrices[f"{name}.distance"] = pairwise_distance_matrix(points, normalize=True)
        matrices[f"{name}.angle"] = pairwise_angle_matrix(points)
        scalars[f"{name}.scale"] = cluster_scale(points)

    left_eye = clusters.get("left_eye_skin")
    right_eye = clusters.get("right_eye_skin")
    left_pupil = clusters.get("left_pupil")
    right_pupil = clusters.get("right_pupil")
    mouth = clusters.get("mouth")

    if left_eye is not None and len(left_eye) >= 8:
        scalars["left_eye.aspect_ratio"] = eye_aspect_ratio(left_eye)
    if right_eye is not None and len(right_eye) >= 8:
        scalars["right_eye.aspect_ratio"] = eye_aspect_ratio(right_eye)
    if left_pupil is not None and left_eye is not None and len(left_pupil) >= 4:
        scalars["left_pupil.iris_ratio"] = iris_diameter_ratio(left_pupil, left_eye)
    if right_pupil is not None and right_eye is not None and len(right_pupil) >= 4:
        scalars["right_pupil.iris_ratio"] = iris_diameter_ratio(right_pupil, right_eye)
    if mouth is not None and len(mouth) >= 13:
        scalars["mouth.open_ratio"] = mouth_open_ratio(mouth)

    return FacialDistortions(matrices=matrices, scalars=scalars)
