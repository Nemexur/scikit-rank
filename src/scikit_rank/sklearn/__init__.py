"""sklearn-compatible estimators for scikit_rank.

Public API:

- :class:`DCNClassifier`
- :class:`DCNRegressor`
- :class:`DCNRanker`
- :class:`DCNBase` (shared base class for custom estimators)
- :class:`FinalNetClassifier`
- :class:`FinalNetRegressor`
- :class:`FinalNetRanker`
- :class:`FinalNetBase` (shared base class for custom estimators)

Input-validation helpers live in :mod:`scikit_rank.sklearn.input_validation`.
"""

from scikit_rank.sklearn.dcn import (
    DCNBase,
    DCNClassifier,
    DCNRanker,
    DCNRegressor,
)
from scikit_rank.sklearn.finalnet import (
    FinalNetBase,
    FinalNetClassifier,
    FinalNetRanker,
    FinalNetRegressor,
)

__all__ = [
    "DCNBase",
    "DCNClassifier",
    "DCNRanker",
    "DCNRegressor",
    "FinalNetBase",
    "FinalNetClassifier",
    "FinalNetRanker",
    "FinalNetRegressor",
]
