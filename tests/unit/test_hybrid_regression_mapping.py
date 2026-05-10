import numpy as np
import pytest


@pytest.mark.skip(reason="thesis.stage_4_training.walk_forward.hybrid module does not exist")
@pytest.mark.unit
def test_hybrid_regression_to_direction_mapping() -> None:
    """Regression objective must map negative returns to SHORT (-1)."""
    from thesis.stage_4_training.walk_forward.hybrid import _wf_format_predictions

    class DummyRegressor:
        def __init__(self, preds: np.ndarray) -> None:
            self._preds = preds

        def predict(self, X):  # noqa: N803 - matches LightGBM API
            assert len(X) == len(self._preds)
            return self._preds

    raw = np.array([-0.2, 0.0, 0.3], dtype=np.float64)
    model = DummyRegressor(raw)
    X_test = np.zeros((len(raw), 2), dtype=np.float64)

    preds, aligned_proba, proba, raw_preds = _wf_format_predictions(
        model,
        X_test,
        ["f1", "f2"],
        is_regression=True,
    )

    assert proba is None
    assert raw_preds is not None
    assert raw_preds.dtype == np.float64
    assert preds.tolist() == [-1, 0, 1]

    # One-hot columns correspond to [-1, 0, 1] in that order.
    assert aligned_proba.shape == (3, 3)
    assert aligned_proba[0].tolist() == [1.0, 0.0, 0.0]
    assert aligned_proba[1].tolist() == [0.0, 1.0, 0.0]
    assert aligned_proba[2].tolist() == [0.0, 0.0, 1.0]

