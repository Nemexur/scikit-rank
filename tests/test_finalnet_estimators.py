"""Integration tests for sklearn-compatible FINAL estimators."""

import numpy as np
import polars as pl

from scikit_rank import FinalNetClassifier, FinalNetRanker, FinalNetRegressor


def _frame(n: int = 96) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    features = pl.DataFrame(
        {
            "signal": rng.normal(size=n),
            "noise": rng.normal(size=n),
            "category": rng.choice(["a", "b", "c"], size=n),
        },
    )
    return features, (features["signal"].to_numpy() > 0.0).astype(int)


def _options() -> dict:
    return {
        "block_type": "1B",
        "block1_hidden_units": [8],
        "batch_size": 32,
        "epochs": 1,
        "accelerator_config": {"cpu": True},
        "random_state": 0,
    }


def test_finalnet_classifier_fits_and_predicts_probabilities() -> None:
    features, target = _frame()
    classifier = FinalNetClassifier(**_options()).fit(features, target)

    probabilities = classifier.predict_proba(features.head(4))
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_finalnet_classifier_supports_multiclass_targets() -> None:
    features, _ = _frame()
    target = np.arange(len(features)) % 3
    classifier = FinalNetClassifier(**_options()).fit(features, target)

    assert classifier.predict_proba(features.head(4)).shape == (4, 3)


def test_finalnet_classifier_averages_two_blocks() -> None:
    features, target = _frame()
    classifier = FinalNetClassifier(
        **(_options() | {"block_type": "2B", "block2_hidden_units": [8]}),
    ).fit(features, target)

    assert classifier.predict_proba(features.head(4)).shape == (4, 2)


def test_finalnet_classifier_predicts_from_a_lazy_frame() -> None:
    features, target = _frame()
    classifier = FinalNetClassifier(**_options()).fit(features, target)

    lazy_features = features.lazy()
    assert classifier.predict_proba(lazy_features).shape == (len(features), 2)


def test_finalnet_regressor_fits_and_predicts() -> None:
    features, _ = _frame()
    target = features["signal"].to_numpy() * 2.0
    regressor = FinalNetRegressor(**_options()).fit(features, target)

    assert regressor.predict(features.head(4)).shape == (4,)


def test_finalnet_ranker_fits_with_grouped_loss() -> None:
    features, target = _frame()
    ranker = FinalNetRanker(loss="bpr", **_options()).fit(
        features,
        target,
        group=np.arange(len(target)) // 8,
    )

    assert ranker.predict(features.head(4)).shape == (4,)
