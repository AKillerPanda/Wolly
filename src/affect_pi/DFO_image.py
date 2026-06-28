from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


Objective = Callable[[np.ndarray], float]


@dataclass
class DFOResult:
	best_position: np.ndarray
	best_fitness: float
	history: list[float]


def dfo_optimize(
	objective: Objective,
	bounds: list[tuple[float, float]],
	population_size: int = 40,
	delta: float = 0.005,
	max_iterations: int = 120,
	seed: int | None = None,
) -> DFOResult:
	"""Minimize an objective with Dispersive Fly Optimization.

	This is a reusable version of the original DFO image script so it can be
	plugged into training pipelines (for example tuning node offsets).
	"""
	if population_size < 3:
		raise ValueError("population_size must be >= 3")
	if not bounds:
		raise ValueError("bounds cannot be empty")

	rng = np.random.default_rng(seed)
	dim = len(bounds)
	lower = np.array([b[0] for b in bounds], dtype=np.float32)
	upper = np.array([b[1] for b in bounds], dtype=np.float32)
	if np.any(upper <= lower):
		raise ValueError("each bound must satisfy upper > lower")

	flies = rng.uniform(lower, upper, size=(population_size, dim)).astype(np.float32)
	fitness = np.zeros((population_size,), dtype=np.float32)
	history: list[float] = []

	for _ in range(max_iterations):
		for i in range(population_size):
			fitness[i] = float(objective(flies[i]))

		best_idx = int(np.argmin(fitness))
		history.append(float(fitness[best_idx]))

		for i in range(population_size):
			if i == best_idx:
				continue

			left = (i - 1) % population_size
			right = (i + 1) % population_size
			best_neighbor = right if fitness[right] < fitness[left] else left

			u = rng.random(dim, dtype=np.float32)
			flies[i] = flies[best_neighbor] + u * (flies[best_idx] - flies[i])

			# Disturbance step keeps diversity and avoids early local lock-in.
			disturb_mask = rng.random(dim) < delta
			if np.any(disturb_mask):
				flies[i, disturb_mask] = rng.uniform(
					lower[disturb_mask], upper[disturb_mask]
				).astype(np.float32)

			flies[i] = np.clip(flies[i], lower, upper)

	for i in range(population_size):
		fitness[i] = float(objective(flies[i]))

	best_idx = int(np.argmin(fitness))
	return DFOResult(
		best_position=flies[best_idx].copy(),
		best_fitness=float(fitness[best_idx]),
		history=history,
	)


if __name__ == "__main__":
	# Tiny demo: minimize sphere function around zero.
	def sphere(x: np.ndarray) -> float:
		return float(np.sum(x ** 2))

	result = dfo_optimize(
		objective=sphere,
		bounds=[(-10.0, 10.0), (-10.0, 10.0)],
		population_size=30,
		max_iterations=100,
		seed=7,
	)
	print("Best fitness:", result.best_fitness)
	print("Best position:", result.best_position)

