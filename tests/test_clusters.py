import numpy as np

from affect_pi.clusters import (
    DEFAULT_FACE_CLUSTERS,
    MOUTH,
    LandmarkCluster,
    extract_clusters,
)


def test_cluster_extract_selects_indices():
    cluster = LandmarkCluster("x", (0, 2, 4))
    lm = np.arange(15, dtype=np.float32).reshape(5, 3)
    out = cluster.extract(lm)
    assert out.shape == (3, 3)
    assert np.allclose(out[0], lm[0])
    assert np.allclose(out[2], lm[4])


def test_cluster_extract_drops_out_of_range_indices():
    cluster = LandmarkCluster("x", (0, 999))
    lm = np.zeros((3, 3), dtype=np.float32)
    out = cluster.extract(lm)
    assert out.shape == (1, 3)   # only index 0 is usable


def test_cluster_extract_all_out_of_range_returns_empty():
    cluster = LandmarkCluster("x", (50, 51))
    out = cluster.extract(np.zeros((3, 3), dtype=np.float32))
    assert out.shape == (0, 3)


def test_extract_clusters_returns_all_default_names():
    lm = np.zeros((478, 3), dtype=np.float32)
    clusters = extract_clusters(lm)
    names = {c.name for c in DEFAULT_FACE_CLUSTERS}
    assert set(clusters.keys()) == names
    # MOUTH has 31 indices, all < 478.
    assert clusters["mouth"].shape == (len(MOUTH.indices), 3)
