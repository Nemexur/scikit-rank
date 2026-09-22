"""Tests for ordered estimator-level TrainingRun configuration callbacks."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import pytest
from accelerate import Accelerator
from ignite.engine import Events
from sklearn.base import clone

from scikit_rank import (
    DCNClassifier,
    DESTINEClassifier,
    FinalMLPClassifier,
    FinalNetClassifier,
    TabMClassifier,
)
from scikit_rank.train.trainer import Trainer


def _features() -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    features = pl.DataFrame(
        {
            "num": rng.normal(size=24),
            "cat": rng.choice(["a", "b", "c"], size=24),
        },
    )
    return features, (features["num"].to_numpy() > 0).astype(int)


@dataclass
class TrainingOptionRecorder:
    name: str
    calls: list[str]
    trainer: Trainer | None = None
    accelerator: Accelerator | None = None
    iteration_count: int = 0

    def __call__(self, trainer: Trainer, accelerator: Accelerator) -> None:
        self.calls.append(self.name)
        self.trainer = trainer
        self.accelerator = accelerator

        def count_iteration(_: Any) -> None:
            self.iteration_count += 1

        trainer.add_event("train", Events.ITERATION_COMPLETED, count_iteration)


def _estimator(kind: str, training_options: list[TrainingOptionRecorder]):
    common = {
        "epochs": 1,
        "batch_size": 8,
        "num_features": ["num"],
        "cat_features": ["cat"],
        "random_state": 0,
        "accelerator_config": {"cpu": True},
        "training_options": training_options,
    }
    if kind == "dcn":
        return DCNClassifier(hidden_units=[8], cross_layers=1, **common)
    if kind == "finalnet":
        return FinalNetClassifier(block1_hidden_units=[8], **common)
    if kind == "finalmlp":
        return FinalMLPClassifier(mlp1_hidden_units=[8], mlp2_hidden_units=[8], **common)
    if kind == "tabm":
        return TabMClassifier(n_blocks=1, d_block=8, k=2, **common)
    if kind == "destine":
        return DESTINEClassifier(embedding_dim=8, attention_dim=4, num_heads=2, **common)
    raise AssertionError(f"Unknown estimator kind: {kind}")


@pytest.mark.parametrize("kind", ["dcn", "finalnet", "finalmlp", "tabm", "destine"])
def test_estimator_training_options_are_ordered_and_attach_handlers(kind: str) -> None:
    calls: list[str] = []
    first = TrainingOptionRecorder("first", calls)
    second = TrainingOptionRecorder("second", calls)
    estimator = _estimator(kind, [first, second])
    features, target = _features()

    estimator.fit(features, target)

    assert calls == ["first", "second"]
    for recorder in (first, second):
        assert isinstance(recorder.trainer, Trainer)
        assert isinstance(recorder.accelerator, Accelerator)
        assert recorder.iteration_count > 0


def _no_op_option(trainer: Trainer, accelerator: Accelerator) -> None:
    del trainer, accelerator


def test_training_options_are_sklearn_clone_parameters() -> None:
    options = [_no_op_option]
    estimator = DCNClassifier(training_options=options)

    cloned = clone(estimator)

    assert estimator.get_params()["training_options"] is options
    assert cloned.training_options == options
