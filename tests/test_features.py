import numpy as np

from affect_pi.features import FacialDistortions, build_facial_distortions


def test_build_facial_distortions_full_face(face_pts3d):
    fd = build_facial_distortions(face_pts3d)
    # Every cluster with >=2 points contributes a .distance and .angle matrix.
    assert "forehead.distance" in fd.matrices
    assert "forehead.angle" in fd.matrices
    assert "mouth.distance" in fd.matrices
    # Scalars include per-cluster scale plus the expression ratios.
    assert "mouth.open_ratio" in fd.scalars
    assert "left_eye.aspect_ratio" in fd.scalars
    assert "left_pupil.iris_ratio" in fd.scalars


def test_feature_vector_is_finite_and_nonempty(face_pts3d):
    vec = build_facial_distortions(face_pts3d).feature_vector()
    assert vec.ndim == 1
    assert vec.size > 0
    assert np.all(np.isfinite(vec))
    assert vec.dtype == np.float32


def test_compact_matrix_one_row_per_scalar(face_pts3d):
    fd = build_facial_distortions(face_pts3d)
    cm = fd.compact_matrix()
    assert cm.shape == (len(fd.scalars), 1)


def test_empty_landmarks_degrade_gracefully():
    fd = build_facial_distortions(np.zeros((0, 3), dtype=np.float32))
    assert fd.matrices == {}
    assert fd.scalars == {}
    assert fd.feature_vector().size == 0
    assert fd.compact_matrix().shape == (0, 1)


def test_feature_vector_handles_scalars_only():
    fd = FacialDistortions(matrices={}, scalars={"a": 1.0, "b": 2.0})
    # sorted keys -> [a, b]
    assert np.allclose(fd.feature_vector(), [1.0, 2.0])
