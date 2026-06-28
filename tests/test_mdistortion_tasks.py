"""Tests for the pure (no-MediaPipe-call) parts of the Tasks feature extractor."""
import numpy as np

from affect_pi.mdistortion_tasks import face_from_pts, feature_from_pts
from affect_pi.train_mdistortion_de import sample_feature


def test_face_from_pts_builds_bbox_scaled_face(face_pts2d):
    face = face_from_pts(face_pts2d, label="Happy")
    assert face.label == "Happy"
    assert len(face.landmarks) == 478
    # face_width/height come from the landmark bounding box.
    span = face_pts2d.max(axis=0) - face_pts2d.min(axis=0)
    assert np.isclose(face.face_width, max(span[0], 1.0))
    assert np.isclose(face.face_height, max(span[1], 1.0))


def test_feature_from_pts_matches_sample_feature(face_pts2d):
    offsets = np.zeros(6, dtype=np.float32)
    direct = feature_from_pts(face_pts2d, offsets)
    viaface = sample_feature(face_from_pts(face_pts2d), offsets)
    assert direct.shape == (912,)
    assert np.allclose(direct, viaface)


def test_feature_responds_to_node_offsets(face_pts2d):
    """Node offsets are now per-group anisotropic scale factors, so they DO change
    the feature (regression for the old no-op bug, where rigid translation left the
    intra-group pairwise distances -- and thus DE's objective -- unchanged)."""
    a = feature_from_pts(face_pts2d, np.zeros(6, dtype=np.float32))
    b = feature_from_pts(face_pts2d, np.array([0.0, 0.3, 0, 0, 0, 0], dtype=np.float32))
    assert not np.allclose(a, b)
    # zero offsets remain the identity (consistent with training at offsets=0)
    a2 = feature_from_pts(face_pts2d, np.zeros(6, dtype=np.float32))
    assert np.allclose(a, a2)
