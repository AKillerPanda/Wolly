"""Equivalence tests for the vectorized hot paths.

Each test reconstructs the original Python-loop logic as a reference and asserts
the new broadcasting implementation produces identical results.
"""
import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_yolo_live_module():
    """Load the scripts/ module by path. It defines dataclasses, so it must be
    registered in sys.modules before exec or dataclasses can't resolve __module__."""
    pytest.importorskip("ultralytics")
    pytest.importorskip("mediapipe")
    if "ml_yolo_live" in sys.modules:
        return sys.modules["ml_yolo_live"]
    path = ROOT / "scripts" / "mdistortion_live_yolo_face.py"
    spec = importlib.util.spec_from_file_location("ml_yolo_live", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ml_yolo_live"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_normalized_mdistortion_energy_matches_loop(rng):
    from affect_pi.live_tasks_demo import normalized_mdistortion_energy

    pts = rng.uniform(0, 200, size=(20, 2)).astype(np.float32)
    scale = 137.0
    ref = float(np.mean([np.linalg.norm(a - b) for a, b in combinations(pts, 2)]) / scale)
    assert normalized_mdistortion_energy(pts, scale) == pytest.approx(ref, abs=1e-4)


def test_normalized_mdistortion_energy_guards():
    from affect_pi.live_tasks_demo import normalized_mdistortion_energy
    assert normalized_mdistortion_energy(np.zeros((1, 2), dtype=np.float32), 10.0) == 0.0
    assert normalized_mdistortion_energy(np.zeros((5, 2), dtype=np.float32), 0.0) == 0.0


def test_nearest_centroid_matches_naive_loop(rng):
    from affect_pi.train_mdistortion_de import NearestCentroidModel

    centroids = {c: rng.normal(0, 1, 6).astype(np.float32) for c in ["a", "b", "c", "d"]}
    model = NearestCentroidModel(centroids)
    X = rng.normal(0, 1, (15, 6)).astype(np.float32)

    classes = model.classes_
    # naive reference (the original per-row logic)
    ref_pred, ref_proba = [], []
    for row in X:
        d = np.array([np.linalg.norm(row - centroids[c]) for c in classes], dtype=np.float32)
        ref_pred.append(classes[int(np.argmin(d))])
        inv = 1.0 / (d + 1e-8)
        ref_proba.append(inv / inv.sum())

    assert list(model.predict(X)) == ref_pred
    assert np.allclose(model.predict_proba(X), np.stack(ref_proba), atol=1e-5)
    assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0, atol=1e-5)


def test_yolo_live_pairwise_features_match_loop(rng):
    mod = _load_yolo_live_module()

    names = [f"n{i:02d}" for i in range(8)]
    nodes = {n: tuple(rng.uniform(0, 300, size=2).astype(float)) for n in names}
    scale = 88.0

    out = mod.MdistortionEngine._pairwise_group_features("mouthMd", nodes, scale)
    ref = {f"mouthMd:{a}__to__{b}": mod.euclidean(nodes[a], nodes[b]) / scale
           for a, b in combinations(sorted(names), 2)}
    assert list(out.keys()) == list(ref.keys())     # same keys, same order
    for k in ref:
        assert out[k] == pytest.approx(ref[k], abs=1e-4)


def test_yolo_live_pairwise_features_empty():
    mod = _load_yolo_live_module()
    assert mod.MdistortionEngine._pairwise_group_features("g", {"only": (1.0, 2.0)}, 1.0) == {}


# --- batched training math equals the per-face implementation -------------- #

def _mk_faces(rng, n_per=8):
    from affect_pi.train_mdistortion_de import DetectedFace
    faces = []
    for label in ("A", "B"):
        for _ in range(n_per):
            lm = {i: rng.uniform(0, 200, 2).astype(np.float32) for i in range(478)}
            faces.append(DetectedFace(Path("x"), label, lm, 100.0, 120.0))
    return faces


def test_batch_features_match_per_face(rng):
    from affect_pi.train_mdistortion_de import build_feature_matrix, sample_feature
    faces = _mk_faces(rng)
    offsets = np.array([0.1, -0.2, 0.0, 0.3, -0.4, 0.1], dtype=np.float32)
    ref = np.stack([sample_feature(f, offsets) for f in faces])
    batch, labels, kept = build_feature_matrix(faces, offsets)
    assert batch.shape == ref.shape == (16, 912)
    assert np.allclose(batch, ref, atol=1e-5)
    assert list(labels) == [f.label for f in faces]


def test_fisher_score_matches_loop(rng):
    from affect_pi.train_mdistortion_de import fisher_score
    X = np.vstack([rng.normal(0, 1, (12, 6)), rng.normal(4, 1, (10, 6))]).astype(np.float32)
    y = np.array(["a"] * 12 + ["b"] * 10)
    classes = sorted(set(y))
    mu = X.mean(0)
    sb = sw = 0.0
    for c in classes:
        xk = X[y == c]
        muk = xk.mean(0)
        sb += len(xk) * np.sum((muk - mu) ** 2)
        sw += np.sum((xk - muk) ** 2)
    assert np.isclose(fisher_score(X, y), sb / (sw + 1e-8), rtol=1e-4)


def test_ranges_and_band_accuracy_match_per_face(rng):
    from affect_pi.train_mdistortion_de import (
        MDISTORTION_GROUPS,
        compute_mdistortion_ranges,
        group_mdistortion_upper,
        range_band_accuracy,
    )
    faces = _mk_faces(rng)
    offsets = np.array([0.1, 0.0, -0.2, 0.3, 0.0, 0.1], dtype=np.float32)
    ranges = compute_mdistortion_ranges(faces, offsets)
    # ranges stats equal the per-face stacked reference
    refA = np.stack([group_mdistortion_upper(f, "mouth", offsets) for f in faces if f.label == "A"])
    band = ranges["mouth"]["by_emotion"]["A"]
    assert band["count"] == 8
    assert np.allclose(band["min"], refA.min(0), atol=1e-5)
    assert np.allclose(band["mean"], refA.mean(0), atol=1e-5)

    # band accuracy equals the old triple-nested implementation
    def ref_acc():
        classes = sorted({l for g in ranges for l in ranges[g]["by_emotion"]})
        cor = tot = 0
        for f in faces:
            s = {g: group_mdistortion_upper(f, g, offsets) for g in MDISTORTION_GROUPS}
            best, bs = None, -1.0
            for c in classes:
                ib = npr = 0
                for g in MDISTORTION_GROUPS:
                    b = ranges[g]["by_emotion"].get(c)
                    if b is None:
                        continue
                    v = s[g]
                    ib += int(np.sum((v >= b["min"]) & (v <= b["max"]))); npr += v.size
                sc = ib / npr if npr else 0.0
                if sc > bs:
                    bs, best = sc, c
            tot += 1; cor += (best == f.label)
        return cor / tot
    assert np.isclose(range_band_accuracy(faces, offsets, ranges), ref_acc())
