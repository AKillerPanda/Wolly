from pathlib import Path

import numpy as np
import pytest

from affect_pi.train_mdistortion_de import (
    LEFT_EYE_SKIN_IDS,
    OUTER_MOUTH_IDS,
    RIGHT_EYE_SKIN_IDS,
    DetectedFace,
    NearestCentroidModel,
    compute_mdistortion_ranges,
    fisher_score,
    group_mdistortion_upper,
    iou_xyxy,
    pairwise_mdistortion,
    range_band_accuracy,
    ranges_summary,
    ranges_to_jsonable,
    sample_feature,
    train_model,
    warp_group,
)

ALL_IDS = set(OUTER_MOUTH_IDS + LEFT_EYE_SKIN_IDS + RIGHT_EYE_SKIN_IDS)


def make_face(label, rng, shift=0.0):
    landmarks = {i: (rng.uniform(0, 200, size=2) + shift).astype(np.float32)
                 for i in range(478)}
    return DetectedFace(Path("x"), label, landmarks, 100.0, 120.0)


def test_iou_identical_disjoint_and_partial():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # half-overlap along x: intersection 50, union 150 -> 1/3.
    assert np.isclose(iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3, atol=1e-6)


def test_pairwise_mdistortion_normalized():
    pts = np.array([[0, 0], [3, 4], [6, 8]], dtype=np.float32)
    m = pairwise_mdistortion(pts)
    assert m.shape == (3, 3)
    assert np.allclose(np.diag(m), 0.0)
    assert np.isclose(m.max(), 1.0)
    assert np.allclose(m, m.T)


def test_warp_group_scales_anisotropically_about_centroid():
    # distinct points so the scaling is observable
    landmarks = {idx: np.array([float(k), float(2 * k)], dtype=np.float32)
                 for k, idx in enumerate(OUTER_MOUTH_IDS)}
    base = warp_group(landmarks, OUTER_MOUTH_IDS, 0.0, 0.0)
    scaled = warp_group(landmarks, OUTER_MOUTH_IDS, 0.5, 0.0)   # x expands 1.5x
    cx = base[:, 0].mean()
    assert np.allclose(scaled[:, 0] - cx, (base[:, 0] - cx) * 1.5, atol=1e-4)
    assert np.allclose(scaled[:, 1], base[:, 1], atol=1e-4)     # y untouched
    # zero offsets are the identity transform
    assert np.allclose(base, np.stack([landmarks[i] for i in OUTER_MOUTH_IDS]))


def test_de_objective_responds_to_offsets(rng):
    """Regression for the old no-op bug: the Fisher objective DE maximizes must
    actually change with the node offsets (it didn't with rigid translation)."""
    faces = []
    for _ in range(20):
        lm = {i: rng.uniform(0, 200, 2).astype(np.float32) for i in range(478)}
        for i in OUTER_MOUTH_IDS:
            lm[i][1] *= 1.6                      # class A: tall mouths
        faces.append(make_face("A", rng))
        faces[-1] = DetectedFace(faces[-1].image_path, "A", lm, 100.0, 120.0)
        faces.append(make_face("B", rng))        # class B: random
    def fisher_at(off):
        X = np.stack([sample_feature(f, off) for f in faces])
        y = np.array([f.label for f in faces])
        return fisher_score(X, y)
    base = fisher_at(np.zeros(6, dtype=np.float32))
    warped = fisher_at(np.array([0.3, -0.3, 0, 0, 0, 0], dtype=np.float32))
    assert not np.isclose(base, warped)          # objective is sensitive -> DE useful


def test_sample_feature_dimension(rng):
    face = make_face("Happy", rng)
    feat = sample_feature(face, np.zeros(6, dtype=np.float32))
    expected = len(OUTER_MOUTH_IDS) ** 2 + len(LEFT_EYE_SKIN_IDS) ** 2 + len(RIGHT_EYE_SKIN_IDS) ** 2
    assert feat.size == expected == 912


def test_sample_feature_returns_none_when_group_missing():
    face = DetectedFace(Path("x"), "Happy", {0: np.zeros(2, np.float32)}, 100.0, 100.0)
    assert sample_feature(face, np.zeros(6, dtype=np.float32)) is None


def test_fisher_score_separates_classes():
    a = np.zeros((10, 3))
    b = np.ones((10, 3)) * 5
    feats = np.vstack([a, b])
    labels = np.array(["a"] * 10 + ["b"] * 10)
    assert fisher_score(feats, labels) > 0.0
    assert fisher_score(a, np.array(["a"] * 10)) == 0.0   # single class


def test_nearest_centroid_model():
    centroids = {"a": np.zeros(3), "b": np.ones(3) * 10}
    m = NearestCentroidModel(centroids)
    preds = m.predict(np.array([[0.1, 0.0, 0.0], [9.0, 9.0, 9.0]]))
    assert list(preds) == ["a", "b"]
    probs = m.predict_proba(np.array([[0.0, 0.0, 0.0]]))
    assert np.isclose(probs.sum(), 1.0)


def test_train_model_randomforest_path(rng):
    X = np.vstack([rng.normal(0, 1, (10, 5)), rng.normal(8, 1, (10, 5))]).astype(np.float32)
    y = np.array(["calm"] * 10 + ["excited"] * 10)
    model, acc, name = train_model(X, y, seed=1)
    assert name == "RandomForestClassifier"
    assert 0.0 <= acc <= 1.0
    assert hasattr(model, "predict")


def test_mdistortion_ranges_structure_and_jsonable(rng):
    faces = ([make_face("Happy", rng, shift=0.0) for _ in range(6)]
             + [make_face("Sad", rng, shift=30.0) for _ in range(6)])
    offsets = np.zeros(6, dtype=np.float32)
    ranges = compute_mdistortion_ranges(faces, offsets)
    assert set(ranges.keys()) == {"mouth", "left_eye", "right_eye"}
    n_pairs = len(OUTER_MOUTH_IDS) * (len(OUTER_MOUTH_IDS) - 1) // 2
    assert len(ranges["mouth"]["pairs"]) == n_pairs
    band = ranges["mouth"]["by_emotion"]["Happy"]
    assert band["min"].shape == (n_pairs,)
    assert band["count"] == 6

    js = ranges_to_jsonable(ranges)
    assert isinstance(js["mouth"]["by_emotion"]["Happy"]["min"], list)
    summary = ranges_summary(ranges)
    assert "mouth" in summary and "Happy" in summary["mouth"]["emotions"]

    acc = range_band_accuracy(faces, offsets, ranges)
    assert 0.0 <= acc <= 1.0


def test_group_mdistortion_upper_length(rng):
    face = make_face("Happy", rng)
    vec = group_mdistortion_upper(face, "mouth", np.zeros(6, dtype=np.float32))
    assert vec.shape == (len(OUTER_MOUTH_IDS) * (len(OUTER_MOUTH_IDS) - 1) // 2,)
