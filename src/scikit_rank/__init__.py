"""scikit_rank — sklearn-compatible neural estimators for tabular data.

Neural drop-in alternatives to LightGBM/CatBoost/XGBoost for tabular ranking
and classification.
"""

from scikit_rank.factories import (
    build_categorical_encoder,
    build_dcnv2,
    build_destine,
    build_embedding_encoder,
    build_final_mlp,
    build_finalnet,
    build_multihash_encoder,
    build_numeric_encoder,
    build_tabm,
)
from scikit_rank.modules.dcn import DCNv2, EmbeddingTower, MultiHashEmbeddings, UnifiedEmbeddings
from scikit_rank.modules.destine import DESTINE, DisentangledSelfAttention
from scikit_rank.modules.final_mlp import FinalMLP
from scikit_rank.modules.losses import LOSSES, Loss, make_loss
from scikit_rank.modules.reducers import Concat, PassThrough, Reduce
from scikit_rank.modules.tabm import TabM
from scikit_rank.preprocessing import TabularPreprocessor
from scikit_rank.run import RunOutput, TrainingModule, TrainingRun
from scikit_rank.sklearn import (
    DCNClassifier,
    DCNRanker,
    DCNRegressor,
    DESTINEClassifier,
    DESTINERanker,
    DESTINERegressor,
    FinalMLPClassifier,
    FinalMLPRanker,
    FinalMLPRegressor,
    FinalNetClassifier,
    FinalNetRanker,
    FinalNetRegressor,
    TabMClassifier,
    TabMRanker,
    TabMRegressor,
)
from scikit_rank.train import OptimizerConfig, Trainer, build_optimizer
from scikit_rank.utils import ModuleParserSpec

__all__ = [
    "DESTINE",
    "LOSSES",
    "Concat",
    "DCNClassifier",
    "DCNRanker",
    "DCNRegressor",
    "DCNv2",
    "DESTINEClassifier",
    "DESTINERanker",
    "DESTINERegressor",
    "DisentangledSelfAttention",
    "EmbeddingTower",
    "FinalMLP",
    "FinalMLPClassifier",
    "FinalMLPRanker",
    "FinalMLPRegressor",
    "FinalNetClassifier",
    "FinalNetRanker",
    "FinalNetRegressor",
    "Loss",
    "ModuleParserSpec",
    "MultiHashEmbeddings",
    "OptimizerConfig",
    "PassThrough",
    "Reduce",
    "RunOutput",
    "TabM",
    "TabMClassifier",
    "TabMRanker",
    "TabMRegressor",
    "TabularPreprocessor",
    "Trainer",
    "TrainingModule",
    "TrainingRun",
    "UnifiedEmbeddings",
    "build_categorical_encoder",
    "build_dcnv2",
    "build_destine",
    "build_embedding_encoder",
    "build_final_mlp",
    "build_finalnet",
    "build_multihash_encoder",
    "build_numeric_encoder",
    "build_optimizer",
    "build_tabm",
    "make_loss",
]
