"""Tests for the pure metric helpers in live_tasks_demo.py."""
import numpy as np

from affect_pi.live_tasks_demo import compute_metrics, open_ratio, points_for


def test_points_for_selects_indices():
    pts = np.arange(20, dtype=np.float32).reshape(10, 2)
    out = points_for([0, 3, 9], pts)
    assert out.shape == (3, 2)
    assert np.allclose(out[1], pts[3])


def test_open_ratio_geometry():
    top = np.array([0.0, 0.0], dtype=np.float32)
    bottom = np.array([0.0, 4.0], dtype=np.float32)   # vertical 4
    left = np.array([-4.0, 0.0], dtype=np.float32)
    right = np.array([4.0, 0.0], dtype=np.float32)    # width 8
    assert np.isclose(open_ratio(top, bottom, left, right), 0.5)


def test_open_ratio_zero_width_guard():
    p = np.zeros(2, dtype=np.float32)
    assert open_ratio(p, p, p, p) == 0.0


def test_compute_metrics_keys_and_finiteness(face_pts2d):
    m = compute_metrics(face_pts2d)
    expected = {"mouth_md", "left_eye_md", "right_eye_md",
                "mouth_open", "left_eye_open", "right_eye_open"}
    assert set(m) == expected
    assert all(np.isfinite(v) for v in m.values())
    assert all(v >= 0.0 for v in m.values())
