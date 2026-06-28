from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np


@dataclass
class EmotionPrediction:
    label: str
    probabilities: dict[str, float] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)


class EmotionModel(Protocol):
    def predict(self, feature_vector: np.ndarray) -> EmotionPrediction: ...


@dataclass
class NeutralPlaceholderModel:
    """Safe placeholder until your trained model is plugged in."""

    label: str = "neutral"

    def predict(self, feature_vector: np.ndarray) -> EmotionPrediction:
        return EmotionPrediction(label=self.label, probabilities={self.label: 1.0})


@dataclass
class JoblibEmotionModel:
    """Adapter for sklearn-like models saved with joblib."""

    model_path: str | Path

    def __post_init__(self) -> None:
        self._model = joblib.load(self.model_path)

    def predict(self, feature_vector: np.ndarray) -> EmotionPrediction:
        x = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)

        if hasattr(self._model, "predict_proba"):
            probs = np.asarray(self._model.predict_proba(x)[0], dtype=np.float32)
            classes = [str(c) for c in getattr(self._model, "classes_", range(len(probs)))]
            probability_map = {cls: float(prob) for cls, prob in zip(classes, probs)}
            label = max(probability_map, key=probability_map.get)
            return EmotionPrediction(label=label, probabilities=probability_map)

        if hasattr(self._model, "predict"):
            label = str(self._model.predict(x)[0])
            return EmotionPrediction(label=label)

        raise TypeError("Loaded model must expose predict_proba(X) or predict(X)")


def make_model(model_path: str | None) -> EmotionModel:
    if model_path:
        return JoblibEmotionModel(model_path=model_path)
    return NeutralPlaceholderModel()
