import numpy as np
import pytest

from affect_pi.DE_nodes import de_optimize
from affect_pi.DFO_image import dfo_optimize


def sphere(x):
    return float(np.sum(x ** 2))


def shifted_sphere(shift):
    return lambda x: float(np.sum((x - shift) ** 2))


def test_de_converges_to_shift():
    shift = np.array([1.5, -2.0, 0.75])
    res = de_optimize(shifted_sphere(shift), [(-5.0, 5.0)] * 3,
                      population_size=24, max_iterations=120, seed=7)
    assert res.best_fitness < 1e-2
    assert np.allclose(res.best_position, shift, atol=0.15)
    # history is recorded and non-increasing overall.
    assert res.history[-1] <= res.history[0]


def test_de_validates_inputs():
    with pytest.raises(ValueError):
        de_optimize(sphere, [(-1.0, 1.0)], population_size=3)   # need >= 4
    with pytest.raises(ValueError):
        de_optimize(sphere, [], population_size=10)             # empty bounds
    with pytest.raises(ValueError):
        de_optimize(sphere, [(1.0, 1.0)], population_size=10)   # upper not > lower
    with pytest.raises(ValueError):
        de_optimize(sphere, [(-1.0, 1.0)], population_size=10, cr=2.0)


def test_de_respects_bounds():
    res = de_optimize(sphere, [(2.0, 3.0), (2.0, 3.0)],
                      population_size=12, max_iterations=20, seed=1)
    assert np.all(res.best_position >= 2.0 - 1e-9)
    assert np.all(res.best_position <= 3.0 + 1e-9)


def test_dfo_converges_to_zero():
    res = dfo_optimize(sphere, [(-10.0, 10.0)] * 2,
                       population_size=30, max_iterations=100, seed=7)
    assert res.best_fitness < 1e-1
    assert np.allclose(res.best_position, 0.0, atol=0.5)


def test_dfo_validates_inputs():
    with pytest.raises(ValueError):
        dfo_optimize(sphere, [(-1.0, 1.0)], population_size=2)
    with pytest.raises(ValueError):
        dfo_optimize(sphere, [], population_size=10)


def test_optimizers_are_deterministic_with_seed():
    a = de_optimize(sphere, [(-5.0, 5.0)] * 2, population_size=10, max_iterations=15, seed=42)
    b = de_optimize(sphere, [(-5.0, 5.0)] * 2, population_size=10, max_iterations=15, seed=42)
    assert np.allclose(a.best_position, b.best_position)
