"""Sklearn-compatible DESTINE estimators."""

from __future__ import annotations
import contextlib
import copy
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import polars as pl
import torch
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from scikit_rank.data import to_polars
from scikit_rank.factories import (
    build_destine,
    build_lr_scheduler_config,
    multihash_encoder_config,
)
from scikit_rank.modules.dcn import CoralLayer
from scikit_rank.modules.losses import (
    LOSSES,
    CORALLayerLoss,
    Loss,
    make_loss,
)
from scikit_rank.preprocessing import TabularPreprocessor
from scikit_rank.run import TrainingRun
from scikit_rank.sklearn._classification import ClassificationTarget, class_probabilities
from scikit_rank.sklearn._data_router import DataRouter
from scikit_rank.sklearn._inference import score_tabular_model
from scikit_rank.sklearn._input_validation import validate_X, validate_y
from scikit_rank.train.optimizers import OptimizerConfig
from scikit_rank.utils import ModuleParserSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sklearn.utils import Tags

    from scikit_rank.sklearn._types import EvalSet, GroupLike, XLike, YLike


class DESTINEBase(BaseEstimator):
    """Shared DESTINE architecture parameters over the common estimator pipeline."""

    def __init__(  # noqa: PLR0913 -- sklearn requires explicit constructor parameters
        self,
        *,
        embedding_dim: int = 16,
        attention_dim: int = 16,
        num_heads: int = 2,
        attention_layers: int = 2,
        dnn_hidden_units: list[int] | tuple[int, ...] = (),
        net_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
        relu_before_attention: bool = False,
        scale_attention: bool = True,
        unary_mode: Literal["paper", "static"] = "paper",
        residual_mode: Literal["each_layer", "last_layer", "none"] | None = "each_layer",
        attention_activation: bool = True,
        use_wide: bool = False,
        cat_encoder: str | torch.nn.Module = "per_feature",
        loss: str | Loss | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        optimizer_kwargs: dict[str, Any] | None = None,
        epochs: int = 10,
        batch_size: int = 1024,
        early_stopping_rounds: int | None = None,
        eval_metric: Callable[..., float] | None = None,
        eval_metric_name: str = "metric",
        eval_metric_direction: str = "max",
        eval_metric_group_aware: bool = False,
        num_features: Sequence[str] | None = None,
        cat_features: Sequence[str] | None = None,
        multihash_features: Sequence[str] | None = None,
        multihash_encoder: str | torch.nn.Module = "multihash",
        embedding_features: dict[str, str] | None = None,
        embedding_encoders: dict[str, str | torch.nn.Module] | None = None,
        normalize_numeric: bool | str | None = True,
        n_quantiles: int = 1000,
        numeric_nan_fill: Literal["median", "zero"] = "median",
        lr_scheduler: str | None = None,
        grad_clip_norm: float | None = None,
        embedding_regularizer: float = 0.0,
        ema_decay: float | None = None,
        chunk_rows: int = 100_000,
        random_state: int | None = None,
        accelerator_config: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.attention_layers = attention_layers
        self.dnn_hidden_units = dnn_hidden_units
        self.net_dropout = net_dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.relu_before_attention = relu_before_attention
        self.scale_attention = scale_attention
        self.unary_mode = unary_mode
        self.residual_mode = residual_mode
        self.attention_activation = attention_activation
        self.use_wide = use_wide
        self.cat_encoder = cat_encoder
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.eval_metric_name = eval_metric_name
        self.eval_metric_direction = eval_metric_direction
        self.eval_metric_group_aware = eval_metric_group_aware
        self.num_features = num_features
        self.cat_features = cat_features
        self.multihash_features = multihash_features
        self.multihash_encoder = multihash_encoder
        self.embedding_features = embedding_features
        self.embedding_encoders = embedding_encoders
        self.normalize_numeric = normalize_numeric
        self.n_quantiles = n_quantiles
        self.numeric_nan_fill = numeric_nan_fill
        self.lr_scheduler = lr_scheduler
        self.grad_clip_norm = grad_clip_norm
        self.embedding_regularizer = embedding_regularizer
        self.ema_decay = ema_decay
        self.chunk_rows = chunk_rows
        self.random_state = random_state
        self.accelerator_config = accelerator_config
        self.verbose = verbose

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        tags.input_tags.allow_nan = True
        tags.input_tags.categorical = True
        tags.input_tags.string = True
        tags.input_tags.sparse = False
        return tags

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        """Encode the raw target into the float32 array seen by the loss."""
        return y.astype(np.float32)

    def _y_expr(self, target_col: str) -> pl.Expr:
        """Lazy counterpart of :meth:`_prepare_y` as a polars expression."""
        return pl.col(target_col).cast(pl.Float32)

    def _resolve_n_outputs(self) -> int:
        return 1

    def _make_loss(self) -> Loss:
        if isinstance(self.loss, Loss):
            return copy.deepcopy(self.loss)
        loss_spec = ModuleParserSpec(self.loss or self._default_loss, allowed=LOSSES)
        return make_loss(loss_spec.module_name(), **loss_spec.kwargs())

    def _fit_target_meta(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        y: YLike,
        target_col: str | None,
    ) -> None:
        """Fit task-specific target metadata, such as classifier classes."""

    def _build_model(
        self,
        *,
        cards: list[int],
        multihash_n_inputs: int,
        embedding_encoders: dict[str, str | torch.nn.Module] | None,
        embedding_input_dims: dict[str, int],
        n_outputs: int,
        use_coral_head: bool,
    ) -> torch.nn.Module:
        return build_destine(
            n_num_features=len(self.preprocessor_.num_cols_),
            cardinalities=cards,
            embedding_dim=int(self.embedding_dim),
            attention_dim=self.attention_dim,
            num_heads=self.num_heads,
            attention_layers=self.attention_layers,
            dnn_hidden_units=self.dnn_hidden_units,
            net_dropout=self.net_dropout,
            attention_dropout=self.attention_dropout,
            activation=self.activation,
            batch_norm=self.batch_norm,
            relu_before_attention=self.relu_before_attention,
            scale_attention=self.scale_attention,
            unary_mode=self.unary_mode,
            residual_mode=self.residual_mode,
            attention_activation=self.attention_activation,
            use_wide=self.use_wide,
            cat_encoder=self.cat_encoder,
            multihash_encoder=self.multihash_encoder,
            multihash_n_inputs=multihash_n_inputs,
            embedding_encoders=embedding_encoders,
            embedding_input_dims=embedding_input_dims,
            n_outputs=n_outputs,
            use_coral_head=use_coral_head,
        )

    def fit(
        self,
        X: XLike,
        y: YLike = None,
        group: GroupLike = None,
        eval_set: EvalSet | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the estimator on tabular features and a task-specific target."""
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.fit() got unexpected keyword "
                f"argument(s): {sorted(kwargs)!r}",
            )
        if self.eval_metric is not None and not callable(self.eval_metric):
            raise TypeError(
                "eval_metric must be a callable metric_fn(y_true, y_pred) -> float "
                "(e.g. sklearn.metrics.roc_auc_score) or None; "
                f"got {self.eval_metric!r}",
            )
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self._rng = np.random.default_rng(self.random_state)

        frame = to_polars(X)
        is_lazy = isinstance(frame, pl.LazyFrame)
        target_col = y if isinstance(y, str) else None
        group_col = group if isinstance(group, str) else None
        if is_lazy and target_col is None:
            raise ValueError("With a polars LazyFrame, `y` must be a column name in X.")
        if is_lazy and group is not None and group_col is None:
            raise ValueError("With a polars LazyFrame, `group` must be a column name in X.")
        if not is_lazy:
            validate_X(X if isinstance(X, np.ndarray) else frame)
        if not isinstance(y, str):
            y = validate_y(y)

        exclude = tuple(c for c in (target_col, group_col) if c is not None)
        multihash_config = multihash_encoder_config(self.multihash_encoder)
        self.preprocessor_ = TabularPreprocessor(
            num_features=list(self.num_features) if self.num_features is not None else None,
            cat_features=list(self.cat_features) if self.cat_features is not None else None,
            normalize=self.normalize_numeric,
            n_quantiles=self.n_quantiles,
            multihash_features=(
                list(self.multihash_features) if self.multihash_features is not None else None
            ),
            multihash_cardinality=multihash_config["cardinality"],
            multihash_n_hashes=multihash_config["n_hashes"],
            embedding_features=(
                dict(self.embedding_features) if self.embedding_features is not None else None
            ),
            numeric_nan_fill=self.numeric_nan_fill,
        ).fit(frame, exclude=exclude)

        self._fit_target_meta(frame, y, target_col)

        cards = self.preprocessor_.cardinalities_
        loss_fn = self._make_loss()
        is_coral = isinstance(loss_fn, CORALLayerLoss)
        if is_coral and not hasattr(self, "n_classes_"):
            estimator_name = type(self).__name__
            model_name = estimator_name.removesuffix("Regressor").removesuffix("Ranker")
            raise ValueError(
                f"loss='coral_layer' is only supported by {model_name}Classifier",
            )
        n_outputs = int(self.n_classes_) if is_coral else self._resolve_n_outputs()
        embedding_input_dims = self.preprocessor_.embedding_input_dims_
        if self.embedding_encoders and not embedding_input_dims:
            raise ValueError("embedding_encoders requires embedding_features")
        embedding_encoders = None
        if embedding_input_dims:
            embedding_encoders = dict.fromkeys(embedding_input_dims, "tower") | dict(
                self.embedding_encoders or {},
            )

        model = self._build_model(
            cards=cards,
            multihash_n_inputs=self.preprocessor_.multihash_n_inputs_,
            embedding_encoders=embedding_encoders,
            embedding_input_dims=embedding_input_dims,
            n_outputs=n_outputs,
            use_coral_head=is_coral,
        )

        with contextlib.ExitStack() as cleanup:
            router = DataRouter(
                self.preprocessor_,
                batch_size=self.batch_size,
                chunk_rows=self.chunk_rows,
                rng=self._rng,
                encode_target=self._prepare_y,
                target_expr=self._y_expr,
            )
            train_source = router.build_train_source(frame, y, group, cleanup=cleanup)
            val_source = router.build_eval_source(
                eval_set,
                group_col=group_col,
                cleanup=cleanup,
                require_group=self.eval_metric is not None and self.eval_metric_group_aware,
            )
            if self.early_stopping_rounds is not None and val_source is None:
                raise ValueError("early_stopping_rounds requires eval_set")

            train_out = TrainingRun(
                model,
                loss_fn,
                train_source,
                val_source,
                lr=self.lr,
                weight_decay=self.weight_decay,
                optimizer=OptimizerConfig(
                    optimizer_type=self.optimizer,
                    **(self.optimizer_kwargs or {}),
                ),
                epochs=self.epochs,
                accelerator_config=self.accelerator_config,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=self.verbose,
                eval_metric_fn=self.eval_metric,
                eval_metric_name=self.eval_metric_name,
                eval_metric_direction=self.eval_metric_direction,
                eval_metric_group_aware=self.eval_metric_group_aware,
                lr_scheduler=build_lr_scheduler_config(self.lr_scheduler),
                grad_clip_norm=self.grad_clip_norm,
                embedding_regularizer=self.embedding_regularizer,
                ema_decay=self.ema_decay,
            ).run()
            module, history = train_out.module(), train_out.metrics()["history"]

        self.model_ = module.model().cpu()
        self.loss_ = module.loss_fn().cpu()
        self.history_ = history
        self.n_features_in_ = (
            len(self.preprocessor_.num_cols_)
            + len(self.preprocessor_.cat_cols_)
            + len(self.preprocessor_.multihash_cols_)
            + len(self.preprocessor_.embedding_cols_)
        )
        self.feature_names_in_ = np.asarray(
            self.preprocessor_.num_cols_
            + self.preprocessor_.cat_cols_
            + self.preprocessor_.multihash_cols_
            + list(self.preprocessor_.embedding_cols_.values()),
        )
        return self

    def save(self, path: str | Path) -> None:
        """Pickle the fitted estimator to ``path``."""
        with Path(path).open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load an estimator saved with :meth:`save`."""
        with Path(path).open("rb") as file:
            obj = pickle.load(file)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected saved {cls.__name__}, got {type(obj).__name__}")
        return obj

    def _decision_scores(self, X: XLike) -> np.ndarray:
        check_is_fitted(self, "model_")
        if not isinstance(X, pl.LazyFrame):
            validate_X(
                X,
                expected_features=self.n_features_in_,
                estimator_name=type(self).__name__,
            )
        return score_tabular_model(
            model=self.model_,
            preprocessor=self.preprocessor_,
            frame=to_polars(X),
            batch_size=self.batch_size,
            chunk_rows=self.chunk_rows,
            accelerator_config=self.accelerator_config,
        )


class DESTINEClassifier(ClassifierMixin, DESTINEBase):
    """DESTINE classifier for binary and multiclass targets."""

    _default_loss = "bce"

    def _fit_target_meta(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        y: YLike,
        target_col: str | None,
    ) -> None:
        self._classification_target_ = ClassificationTarget.fit(frame, y, target_col)
        self._label_encoder_ = self._classification_target_.label_encoder
        self.classes_ = self._classification_target_.classes
        self.n_classes_ = self._classification_target_.n_classes

    def _resolve_n_outputs(self) -> int:
        return 1 if self.n_classes_ <= 2 else self.n_classes_

    def fit(
        self,
        X: XLike,
        y: YLike = None,
        group: GroupLike = None,
        eval_set: EvalSet | None = None,
        **kwargs: Any,
    ) -> DESTINEClassifier:
        """Fit the classifier after inferring and encoding its classes."""
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.fit() got unexpected keyword "
                f"argument(s): {sorted(kwargs)!r}",
            )
        is_lazy_X = isinstance(X, pl.LazyFrame)
        if not is_lazy_X:
            validate_X(X if isinstance(X, np.ndarray) else to_polars(X))
        if not isinstance(y, str):
            y = validate_y(y)
        frame = to_polars(X)
        target_col = y if isinstance(y, str) else None
        self._fit_target_meta(frame, y, target_col)
        self._default_loss = "bce" if self.n_classes_ <= 2 else "cross_entropy"
        return super().fit(X, y=y, group=group, eval_set=eval_set)

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        return self._classification_target_.encode(y)

    def _y_expr(self, target_col: str) -> pl.Expr:
        return self._classification_target_.expression(target_col)

    def predict_proba(self, X: XLike) -> np.ndarray:
        """Predict class probabilities for ``X``."""
        scores = self._decision_scores(X)
        if isinstance(self.model_.head(), CoralLayer) and scores.ndim > 1:
            # Convert cumulative CORAL probabilities to class probabilities.
            p_ge = np.minimum.accumulate(expit(scores), axis=1)
            return np.concatenate(
                [
                    1.0 - p_ge[:, :1],
                    p_ge[:, :-1] - p_ge[:, 1:],
                    p_ge[:, -1:],
                ],
                axis=1,
            )
        return class_probabilities(scores)

    def predict(self, X: XLike) -> np.ndarray:
        """Predict class labels for ``X``."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class DESTINERegressor(RegressorMixin, DESTINEBase):
    """DESTINE regressor."""

    _default_loss = "mse"

    def predict(self, X: XLike) -> np.ndarray:
        """Predict continuous targets for ``X``."""
        return self._decision_scores(X)


class DESTINERanker(DESTINEBase):
    """DESTINE learning-to-rank estimator."""

    _default_loss = "lambdarank"

    def fit(
        self,
        X: XLike,
        y: YLike = None,
        group: GroupLike = None,
        eval_set: EvalSet | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the ranker, requiring groups for group-aware losses."""
        loss_fn = self._make_loss()
        if group is None and loss_fn.requires_group:
            raise ValueError(
                "DESTINERanker.fit requires `group` (per-row query ids or a column name).",
            )
        return super().fit(X, y=y, group=group, eval_set=eval_set, **kwargs)

    def predict(self, X: XLike) -> np.ndarray:
        """Predict ranking scores for ``X``."""
        scores = self._decision_scores(X)
        return scores.sum(axis=1) if scores.ndim == 2 else scores
