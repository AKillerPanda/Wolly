from __future__ import annotations

import numpy as np

EPS = 1e-8


def pairwise_distance_matrix(points: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Return an n x n Euclidean distance matrix for 2D/3D points.

    Normalization uses the maximum pairwise distance in the cluster, making the
    matrix less sensitive to camera distance and resolution.
    """
    pts = _as_points(points)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=-1).astype(np.float32)
    if normalize:
        max_dist = float(np.max(dist))
        if max_dist > EPS:
            dist = dist / max_dist
    return dist


def pairwise_angle_matrix(points: np.ndarray) -> np.ndarray:
    """Return an n x n bearing/orientation matrix in radians, scaled to [-1, 1].

    Entry [i, j] is the 2D angle of the vector from node i to node j divided by pi.
    The diagonal is zero.
    """
    pts = _as_points(points)
    xy = pts[:, :2]
    diff = xy[None, :, :] - xy[:, None, :]
    ang = np.arctan2(diff[..., 1], diff[..., 0]).astype(np.float32) / np.pi
    np.fill_diagonal(ang, 0.0)
    return ang


def centroid(points: np.ndarray) -> np.ndarray:
    pts = _as_points(points)
    return np.mean(pts, axis=0).astype(np.float32)


def cluster_scale(points: np.ndarray) -> float:
    """Robust-ish scale estimate: median distance from centroid."""
    pts = _as_points(points)
    c = centroid(pts)
    d = np.linalg.norm(pts - c, axis=1)
    return float(np.median(d) + EPS)


def eye_aspect_ratio(points: np.ndarray) -> float:
    """Approximate EAR for an 8-point MediaPipe eye ring.

    Expected order: [outer, inner, top-ish, top-ish, top-ish, bottom-ish, bottom-ish, bottom-ish]
    This is intentionally approximate because cluster landmarks can be customized.
    """
    pts = _as_points(points)
    if len(pts) < 8:
        return 0.0
    horizontal = np.linalg.norm(pts[0, :2] - pts[1, :2]) + EPS
    vertical = (
        np.linalg.norm(pts[2, :2] - pts[5, :2])
        + np.linalg.norm(pts[3, :2] - pts[6, :2])
        + np.linalg.norm(pts[4, :2] - pts[7, :2])
    ) / 3.0
    return float(vertical / horizontal)


def mouth_open_ratio(points: np.ndarray) -> float:
    """Approximate mouth opening using selected inner/outer lip points."""
    pts = _as_points(points)
    if len(pts) < 10:
        return 0.0
    width = np.linalg.norm(pts[0, :2] - pts[8, :2]) + EPS
    vertical = np.linalg.norm(pts[4, :2] - pts[12, :2]) if len(pts) > 12 else 0.0
    return float(vertical / width)


def iris_diameter_ratio(iris_points: np.ndarray, eye_points: np.ndarray) -> float:
    """Approximate pupil/iris dilation proxy normalized by visible eye width.

    RGB cameras usually cannot isolate the black pupil reliably under all lighting;
    MediaPipe iris landmarks give an iris-region proxy. Treat this as a proxy, not a
    clinical measurement.
    """
    iris = _as_points(iris_points)
    eye = _as_points(eye_points)
    if len(iris) < 4 or len(eye) < 2:
        return 0.0
    iris_diameter = (
        np.linalg.norm(iris[0, :2] - iris[2, :2])
        + np.linalg.norm(iris[1, :2] - iris[3, :2])
    ) / 2.0
    eye_width = np.linalg.norm(eye[0, :2] - eye[1, :2]) + EPS
    return float(iris_diameter / eye_width)


def flatten_matrices(matrices: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten a dictionary of matrices in stable key order."""
    if not matrices:
        return np.zeros((0,), dtype=np.float32)
    parts = [np.asarray(matrices[k], dtype=np.float32).ravel() for k in sorted(matrices)]
    return np.concatenate(parts).astype(np.float32)


def _as_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"points must be shaped [n, >=2], got {pts.shape}")
    return pts
