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
    """Shared fit/predict machinery for the DESTINE estimators.

    Do not instantiate this class directly. Use :class:`DESTINEClassifier`,
    :class:`DESTINERegressor`, or :class:`DESTINERanker`. The explicit constructor
    parameters work with sklearn ``clone``, ``get_params``/``set_params``, and
    ``GridSearchCV``.

    Parameters
    ----------
    embedding_dim : int, default=16
        Common width of categorical embeddings and projected numeric fields.
    attention_dim : int, default=16
        Width of the disentangled-attention query and key projections.
    num_heads : int, default=2
        Number of attention heads.
    attention_layers : int, default=2
        Number of stacked disentangled-attention layers.
    dnn_hidden_units : sequence of int, default=()
        Hidden widths for the optional deep branch. An empty sequence selects
        attention-only DESTINE; non-empty units create the DESTINE+ branch.
    net_dropout, attention_dropout : float, default=0.0
        Dropout probabilities for the deep and attention branches.
    activation : str, default="relu"
        Activation used by the optional deep branch.
    batch_norm : bool, default=False
        Apply batch normalization in the optional deep branch.
    relu_before_attention : bool, default=False
        Apply ReLU to input fields before attention.
    scale_attention : bool, default=True
        Scale pairwise attention scores by the attention dimension.
    unary_mode : {"paper", "static"}, default="paper"
        Unary-attention formulation. ``"paper"`` follows the paper; ``"static"``
        is the FuxiCTR-compatible key-only scorer.
    residual_mode : {"each_layer", "last_layer", "none"} or None,
        default="each_layer"
        Attention residual-connection strategy.
    attention_activation : bool, default=True
        Apply an activation after the attention projection.
    use_wide : bool, default=False
        Add a linear wide component over raw features.
    cat_encoder : str or torch.nn.Module, default="per_feature"
        Categorical encoder specification.
    loss : str or Loss or None, default=None
        Loss specification or instance. ``None`` selects the task default:
        ``bce``/``cross_entropy``, ``mse``, or ``lambdarank``.
    lr, weight_decay, optimizer, optimizer_kwargs
        Optimizer configuration. DESTINE defaults to ``optimizer="adam"``.
    epochs : int, default=10
        Maximum training epochs.
    batch_size : int, default=1024
        Mini-batch size. Ranking batches preserve query boundaries.
    early_stopping_rounds, eval_metric, eval_metric_name,
    eval_metric_direction, eval_metric_group_aware
        Validation and early-stopping configuration. Early stopping requires an
        ``eval_set`` passed to :meth:`fit`.
    num_features, cat_features : sequence of str or None, default=None
        Explicit feature columns. ``None`` infers columns from input dtypes.
    multihash_features, multihash_encoder
        High-cardinality columns and their shared hashed encoder.
    embedding_features, embedding_encoders
        Named columns containing external embedding vectors and their encoders.
    normalize_numeric, n_quantiles, numeric_nan_fill
        Numeric preprocessing configuration.
    lr_scheduler, grad_clip_norm, embedding_regularizer, ema_decay
        Optional scheduler, optimization, and weight-averaging controls.
    chunk_rows : int, default=100_000
        Streaming chunk size for lazy training and inference.
    random_state : int or None, default=None
        Torch and NumPy random seed.
    accelerator_config : dict or None, default=None
        Keyword arguments for :class:`accelerate.Accelerator`.
    verbose : bool, default=False
        Print training progress and epoch logs.

    Attributes
    ----------
    model_ : torch.nn.Module
        Fitted DESTINE network, retained on CPU for stable pickling.
    loss_ : Loss
        Instantiated training loss.
    history_ : list[dict[str, float]]
        Per-epoch train and validation metrics.
    preprocessor_ : TabularPreprocessor
        Fitted feature preprocessor.
    n_features_in_ : int
        Number of fitted input features.
    feature_names_in_ : numpy.ndarray
        Input feature names in preprocessing order.

    """

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
        """Fit the estimator on ``X`` and ``y``.

        Parameters
        ----------
        X : numpy.ndarray, pandas.DataFrame, polars.DataFrame, or polars.LazyFrame
            Training features. String categoricals and NaNs are handled natively.
            With a lazy frame, the data is preprocessed and streamed through a
            temporary Arrow IPC file.
        y : array-like or str, default=None
            Target values, or a target-column name when ``X`` is a polars frame.
        group : array-like or str or None, default=None
            Per-row query ids for ranking losses. With a lazy frame, pass the
            group-column name. Group-aware losses require this argument.
        eval_set : tuple or None, default=None
            Validation data as ``(X_val, y_val)`` or ``(X_val, y_val, group_val)``.
            Required when ``early_stopping_rounds`` is set.
        **kwargs
            Unsupported. Passing fit parameters such as ``sample_weight`` raises
            :class:`TypeError`.

        Returns
        -------
        self
            The fitted estimator.

        """
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
    """DESTINE classifier for binary, multiclass, and ordinal targets.

    Binary targets use ``bce`` and multiclass targets use ``cross_entropy`` by
    default. ``loss="coral_layer"`` enables an ordinal output head. Fitted labels
    are stored in ``classes_``. See :class:`DESTINEBase` for parameters.

    Examples
    --------
    >>> from scikit_rank import DESTINEClassifier
    >>> classifier = DESTINEClassifier(epochs=5, num_features=["num"])
    >>> classifier.fit(X, y)  # doctest: +SKIP
    >>> classifier.predict_proba(X).shape  # doctest: +SKIP
    (n_samples, n_classes)

    """

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
    """DESTINE regressor with an ``mse`` loss by default.

    See :class:`DESTINEBase` for parameters.

    Examples
    --------
    >>> from scikit_rank import DESTINERegressor
    >>> regressor = DESTINERegressor(epochs=5, num_features=["num"])
    >>> predictions = regressor.fit(X, y_reg).predict(X)  # doctest: +SKIP

    """

    _default_loss = "mse"

    def predict(self, X: XLike) -> np.ndarray:
        """Predict continuous targets for ``X``."""
        return self._decision_scores(X)


class DESTINERanker(DESTINEBase):
    """DESTINE learning-to-rank estimator.

    The default ``lambdarank`` loss, and other group-aware losses, require one
    query id per row through ``fit(X, y, group=...)``. Ranking batches preserve
    complete queries and :meth:`predict` returns scores where larger is better.
    See :class:`DESTINEBase` for parameters.

    Examples
    --------
    >>> from scikit_rank import DESTINERanker
    >>> ranker = DESTINERanker(loss="listwise", epochs=5)
    >>> scores = ranker.fit(df, y="click", group="impression_id").predict(df)  # doctest: +SKIP

    """

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
