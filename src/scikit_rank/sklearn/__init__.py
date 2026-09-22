"""sklearn-compatible estimators for scikit_rank."""

from scikit_rank.sklearn.dcn import (
    DCNBase,
    DCNClassifier,
    DCNRanker,
    DCNRegressor,
)
from scikit_rank.sklearn.destine import (
    DESTINEBase,
    DESTINEClassifier,
    DESTINERanker,
    DESTINERegressor,
)
from scikit_rank.sklearn.final_mlp import (
    FinalMLPBase,
    FinalMLPClassifier,
    FinalMLPRanker,
    FinalMLPRegressor,
)
from scikit_rank.sklearn.finalnet import (
    FinalNetBase,
    FinalNetClassifier,
    FinalNetRanker,
    FinalNetRegressor,
)
from scikit_rank.sklearn.tabm import (
    TabMBase,
    TabMClassifier,
    TabMRanker,
    TabMRegressor,
)

__all__ = [
    "DCNBase",
    "DCNClassifier",
    "DCNRanker",
    "DCNRegressor",
    "DESTINEBase",
    "DESTINEClassifier",
    "DESTINERanker",
    "DESTINERegressor",
    "FinalMLPBase",
    "FinalMLPClassifier",
    "FinalMLPRanker",
    "FinalMLPRegressor",
    "FinalNetBase",
    "FinalNetClassifier",
    "FinalNetRanker",
    "FinalNetRegressor",
    "TabMBase",
    "TabMClassifier",
    "TabMRanker",
    "TabMRegressor",
]
