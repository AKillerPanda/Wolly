from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


Objective = Callable[[np.ndarray], float]


@dataclass
class DEResult:
    best_position: np.ndarray
    best_fitness: float
    history: list[float]


def de_optimize(
    objective: Objective,
    bounds: list[tuple[float, float]],
    population_size: int = 30,
    max_iterations: int = 120,
    f: float = 0.6,
    cr: float = 0.9,
    seed: int | None = None,
) -> DEResult:
    """Minimize an objective with classic Differential Evolution (DE/rand/1/bin).

    This is the node-group optimizer referenced by the training pipeline.
    DFO (``DFO_image.dfo_optimize``) tunes the YOLO detector; DE tunes per-group
    anisotropic scaling of the facial node groups so the three Mdistortion
    matrices separate the emotion classes as well as possible.

    The signature mirrors ``DFO_image.dfo_optimize`` (objective/bounds/
    population_size/max_iterations/seed) so it is a drop-in for that call site,
    with two extra DE knobs:

    - ``f``  differential weight (mutation factor), typically in [0.4, 1.0].
    - ``cr`` crossover probability, typically in [0.1, 1.0].
    """
    if population_size < 4:
        # DE/rand/1 needs the target plus three distinct donors.
        raise ValueError("population_size must be >= 4 for DE/rand/1")
    if not bounds:
        raise ValueError("bounds cannot be empty")
    if not 0.0 <= cr <= 1.0:
        raise ValueError("cr must be in [0, 1]")

    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lower = np.array([b[0] for b in bounds], dtype=np.float64)
    upper = np.array([b[1] for b in bounds], dtype=np.float64)
    if np.any(upper <= lower):
        raise ValueError("each bound must satisfy upper > lower")

    pop = rng.uniform(lower, upper, size=(population_size, dim)).astype(np.float64)
    fitness = np.array([float(objective(ind)) for ind in pop], dtype=np.float64)

    best_idx = int(np.argmin(fitness))
    history: list[float] = [float(fitness[best_idx])]

    all_indices = np.arange(population_size)
    for _ in range(max_iterations):
        for i in range(population_size):
            # Pick three distinct donors, all different from the target i.
            choices = all_indices[all_indices != i]
            a, b, c = rng.choice(choices, size=3, replace=False)

            mutant = pop[a] + f * (pop[b] - pop[c])
            mutant = np.clip(mutant, lower, upper)

            # Binomial crossover; jrand guarantees at least one mutant gene.
            cross = rng.random(dim) < cr
            jrand = rng.integers(dim)
            cross[jrand] = True
            trial = np.where(cross, mutant, pop[i])

            trial_fitness = float(objective(trial))
            if trial_fitness <= fitness[i]:
                pop[i] = trial
                fitness[i] = trial_fitness

        best_idx = int(np.argmin(fitness))
        history.append(float(fitness[best_idx]))

    best_idx = int(np.argmin(fitness))
    return DEResult(
        best_position=pop[best_idx].copy(),
        best_fitness=float(fitness[best_idx]),
        history=history,
    )


if __name__ == "__main__":
    # Tiny demo: minimize a shifted sphere; optimum is the shift vector.
    shift = np.array([1.5, -2.0, 0.75], dtype=np.float64)

    def shifted_sphere(x: np.ndarray) -> float:
        return float(np.sum((x - shift) ** 2))

    result = de_optimize(
        objective=shifted_sphere,
        bounds=[(-5.0, 5.0)] * 3,
        population_size=24,
        max_iterations=120,
        seed=7,
    )
    print("Best fitness:", result.best_fitness)
    print("Best position:", result.best_position)
