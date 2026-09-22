"""Lock in sklearn estimator-compliance checks for DESTINEClassifier / DESTINERegressor.

Runs ``sklearn.utils.estimator_checks.estimator_checks_generator`` against both
estimators. SkipTest results are treated as passes (sklearn auto-skips checks
that aren't applicable to non-deterministic estimators or that require optional
environment variables like ``SCIPY_ARRAY_API``).
"""

from __future__ import annotations

import pytest
from sklearn.utils.estimator_checks import estimator_checks_generator

from scikit_rank import (
    DESTINEClassifier,
    DESTINERegressor,
)


def _conformance_cases():
    for est in [
        DESTINEClassifier(
            embedding_dim=16,
            attention_dim=16,
            dnn_hidden_units=(32, 16),
            use_wide=True,
            lr=1e-2,
            epochs=50,
            batch_size=128,
            accelerator_config={"cpu": True},
        ),
        DESTINERegressor(
            embedding_dim=16,
            attention_dim=16,
            dnn_hidden_units=(32, 16),
            use_wide=True,
            lr=1e-2,
            epochs=50,
            batch_size=128,
            accelerator_config={"cpu": True},
        ),
    ]:
        for estimator, check in estimator_checks_generator(est):
            check_name = (
                getattr(check, "func", check).__name__ if hasattr(check, "func") else check.__name__
            )
            yield pytest.param(estimator, check, id=f"{type(est).__name__}-{check_name}")


@pytest.mark.parametrize(("estimator", "check"), list(_conformance_cases()))
def test_sklearn_estimator_check(estimator, check) -> None:
    """Every sklearn ``estimator_checks_generator`` check must pass or skip."""
    try:
        check(estimator)
    except Exception as exc:
        # sklearn raises SkipTest for non-applicable checks; treat as pass.
        if type(exc).__name__ == "SkipTest":
            pytest.skip(str(exc))
        raise
