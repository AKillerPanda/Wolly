import numpy as np
import pytest

from affect_pi.geometry import (
    centroid,
    cluster_scale,
    eye_aspect_ratio,
    flatten_matrices,
    iris_diameter_ratio,
    mouth_open_ratio,
    pairwise_angle_matrix,
    pairwise_distance_matrix,
)


def test_distance_matrix_shape_diagonal_and_symmetry():
    points = np.array([[0, 0, 0], [3, 4, 0], [6, 8, 0]], dtype=np.float32)
    d = pairwise_distance_matrix(points)
    assert d.shape == (3, 3)
    assert np.allclose(np.diag(d), 0.0)
    assert np.allclose(d, d.T)
    # normalized by the max pairwise distance -> max entry is exactly 1.
    assert np.isclose(d.max(), 1.0)
    assert np.isclose(d[0, 2], 1.0)


def test_distance_matrix_unnormalized_keeps_raw_distance():
    points = np.array([[0, 0], [3, 4]], dtype=np.float32)
    d = pairwise_distance_matrix(points, normalize=False)
    assert np.isclose(d[0, 1], 5.0)


def test_angle_matrix_range_and_diagonal():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    a = pairwise_angle_matrix(points)
    assert a.shape == (3, 3)
    assert np.allclose(np.diag(a), 0.0)
    assert a.min() >= -1.0 and a.max() <= 1.0
    # vector 0->1 points along +x => angle 0.
    assert np.isclose(a[0, 1], 0.0)


def test_centroid_and_scale():
    points = np.array([[0, 0], [2, 0], [0, 2], [2, 2]], dtype=np.float32)
    assert np.allclose(centroid(points), [1.0, 1.0])
    assert cluster_scale(points) > 0.0


def test_eye_aspect_ratio_needs_eight_points():
    assert eye_aspect_ratio(np.zeros((4, 2), dtype=np.float32)) == 0.0
    pts = np.array([[0, 0], [10, 0], [1, 2], [5, 2], [9, 2],
                    [1, -2], [5, -2], [9, -2]], dtype=np.float32)
    assert eye_aspect_ratio(pts) > 0.0


def test_mouth_open_ratio_guard_and_value():
    assert mouth_open_ratio(np.zeros((5, 2), dtype=np.float32)) == 0.0
    pts = np.zeros((13, 2), dtype=np.float32)
    pts[0] = [0, 0]; pts[8] = [10, 0]   # width 10
    pts[4] = [5, 5]; pts[12] = [5, -5]  # vertical 10
    assert np.isclose(mouth_open_ratio(pts), 1.0)


def test_iris_diameter_ratio_guard_and_value():
    assert iris_diameter_ratio(np.zeros((2, 2)), np.zeros((2, 2))) == 0.0
    iris = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)  # diameter ~2
    eye = np.array([[-5, 0], [5, 0]], dtype=np.float32)                    # width 10
    assert np.isclose(iris_diameter_ratio(iris, eye), 0.2, atol=1e-3)


def test_flatten_matrices_stable_order_and_empty():
    assert flatten_matrices({}).shape == (0,)
    mats = {"b": np.array([[1.0, 2.0]]), "a": np.array([[9.0]])}
    flat = flatten_matrices(mats)
    # sorted by key -> 'a' first.
    assert np.allclose(flat, [9.0, 1.0, 2.0])


def test_as_points_rejects_bad_shape():
    with pytest.raises(ValueError):
        pairwise_distance_matrix(np.zeros((5,), dtype=np.float32))
