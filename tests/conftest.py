"""Shared pytest fixtures + import-path setup for the affect-pi test suite.

Both in-repo packages (``affect_pi``, ``robot_eyes``) live directly under ``src``;
add that root so tests can import them regardless of how the project is (or isn't)
installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_src = str(ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def face_pts2d(rng) -> np.ndarray:
    """A synthetic (478, 2) set of face landmark pixel coordinates."""
    return rng.uniform([100, 80], [540, 400], size=(478, 2)).astype(np.float32)


@pytest.fixture
def face_pts3d(rng) -> np.ndarray:
    """A synthetic (478, 3) set of face landmarks (x, y, z)."""
    return rng.uniform([100, 80, -50], [540, 400, 50], size=(478, 3)).astype(np.float32)
