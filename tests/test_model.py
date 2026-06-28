import numpy as np
import pytest

from affect_pi.model import (
    EmotionPrediction,
    JoblibEmotionModel,
    NeutralPlaceholderModel,
    make_model,
)


def test_neutral_placeholder_always_neutral():
    m = NeutralPlaceholderModel()
    pred = m.predict(np.zeros(10, dtype=np.float32))
    assert isinstance(pred, EmotionPrediction)
    assert pred.label == "neutral"
    assert pred.probabilities == {"neutral": 1.0}


def test_make_model_without_path_is_placeholder():
    assert isinstance(make_model(None), NeutralPlaceholderModel)


def test_joblib_model_predicts_label_and_probs(tmp_path):
    joblib = pytest.importorskip("joblib")
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (12, 4)), rng.normal(6, 1, (12, 4))])
    y = np.array(["calm"] * 12 + ["excited"] * 12)
    clf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)

    path = tmp_path / "m.joblib"
    joblib.dump(clf, path)

    model = make_model(str(path))
    assert isinstance(model, JoblibEmotionModel)
    pred = model.predict(np.full(4, 6.0, dtype=np.float32))
    assert pred.label == "excited"
    assert set(pred.probabilities) == {"calm", "excited"}
    assert np.isclose(sum(pred.probabilities.values()), 1.0, atol=1e-5)


class PredictOnly:
    """Module-level so joblib can pickle it; exposes predict() but no predict_proba."""

    def predict(self, x):
        return ["angry"] * len(x)


def test_joblib_model_predict_only_fallback(tmp_path):
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "po.joblib"
    joblib.dump(PredictOnly(), path)
    pred = JoblibEmotionModel(model_path=path).predict(np.zeros(3, dtype=np.float32))
    assert pred.label == "angry"
    assert pred.probabilities == {}
