"""Shared classification target and probability semantics for estimators."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy.special import expit, softmax
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets

from scikit_rank.data import to_numpy_1d

if TYPE_CHECKING:
    from scikit_rank.sklearn._types import YLike


@dataclass(frozen=True)
class ClassificationTarget:
    """Fitted label encoding and its matching eager/lazy transformations."""

    label_encoder: LabelEncoder
    classes: np.ndarray

    @classmethod
    def fit(
        cls,
        frame: pl.DataFrame | pl.LazyFrame,
        y: YLike,
        target_col: str | None,
    ) -> ClassificationTarget:
        """Fit label metadata from an in-memory target or a Polars column."""
        if target_col is not None:
            values = (
                frame.lazy()
                .select(pl.col(target_col).unique().sort())
                .collect()
                .to_series()
                .to_numpy()
            )
        else:
            values = to_numpy_1d(y)
            check_classification_targets(values)
            values = np.unique(values)
        label_encoder = LabelEncoder().fit(values)
        return cls(label_encoder=label_encoder, classes=label_encoder.classes_)

    @property
    def n_classes(self) -> int:
        """Number of fitted classes."""
        return len(self.classes)

    def encode(self, y: np.ndarray) -> np.ndarray:
        """Encode an eager target as the float32 values expected by losses."""
        return self.label_encoder.transform(y).astype(np.float32)

    def expression(self, target_col: str) -> pl.Expr:
        """Encode a lazy Polars target column as float32."""
        classes = pl.Series(self.classes).cast(pl.String).to_list()
        codes = [float(index) for index in range(self.n_classes)]
        return (
            pl.col(target_col)
            .cast(pl.String)
            .replace_strict(classes, codes, default=None, return_dtype=pl.Float32)
        )


def class_probabilities(scores: np.ndarray) -> np.ndarray:
    """Convert binary or multiclass logits into sklearn-style probabilities."""
    if scores.ndim == 1:
        positive = expit(scores)
        return np.stack([1.0 - positive, positive], axis=1)
    return softmax(scores, axis=1)
