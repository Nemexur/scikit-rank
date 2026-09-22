from __future__ import annotations
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import torch.nn.functional as F
import yaml
from sklearn.base import clone

from scikit_rank import (
    DESTINE,
    DCNClassifier,
    DESTINEClassifier,
    DESTINERanker,
    DisentangledSelfAttention,
    build_destine,
)
from scikit_rank.modules.losses import BPRLoss, ListwiseSoftmaxLoss


def _reference_attention(layer, query, key, value):
    batch_size = query.size(0)
    q = layer._query(query)
    k = layer._key(key)
    v = layer._value(value)
    if layer._relu_before_attention:
        q, k, v = q.relu(), k.relu(), v.relu()

    q = torch.cat(q.split(layer._head_dim, dim=2), dim=0)
    k = torch.cat(k.split(layer._head_dim, dim=2), dim=0)
    v = torch.cat(v.split(layer._head_dim, dim=2), dim=0)
    raw_k = k
    q = q - q.mean(dim=1, keepdim=True)
    k = k - k.mean(dim=1, keepdim=True)
    pairwise = torch.bmm(q, k.transpose(1, 2))
    if layer._use_scale:
        pairwise = pairwise / layer._head_dim**0.5
    pairwise = F.softmax(pairwise, dim=2)

    if layer._unary_mode == "paper":
        unary_query = layer._unary_query(query)
        if layer._relu_before_attention:
            unary_query = unary_query.relu()
        unary_query = torch.cat(unary_query.split(layer._head_dim, dim=2), dim=0)
        unary = torch.bmm(
            unary_query.mean(dim=1, keepdim=True),
            raw_k.transpose(1, 2),
        ).softmax(dim=2)
    else:
        unary = F.softmax(layer._unary(key), dim=1)
        unary = torch.cat(unary.split(1, dim=2), dim=0).transpose(1, 2)
    output = torch.bmm(pairwise + unary, v)
    output = torch.cat(output.split(batch_size, dim=0), dim=2)
    if layer._residual is not None:
        output = output + layer._residual(query)
    return output


@pytest.mark.parametrize("unary_mode", ["paper", "static"])
def test_disentangled_attention_matches_reference(unary_mode: str) -> None:
    torch.manual_seed(7)
    layer = DisentangledSelfAttention(
        embedding_dim=5,
        attention_dim=8,
        num_heads=2,
        use_scale=True,
        dropout=0.0,
        unary_mode=unary_mode,
    ).eval()
    x = torch.randn(3, 4, 5)

    actual = layer(x, x, x)
    expected = _reference_attention(layer, x, x, x)

    torch.testing.assert_close(actual, expected)
    pairwise, unary = layer.attention_weights(x, x)
    torch.testing.assert_close(pairwise.sum(dim=-1), torch.ones_like(pairwise[..., 0]))
    torch.testing.assert_close(unary.sum(dim=-1), torch.ones_like(unary[..., 0]))
    assert unary.shape == (6, 1, 4)


def test_paper_unary_uses_sample_dependent_query_context() -> None:
    torch.manual_seed(13)
    layer = DisentangledSelfAttention(
        embedding_dim=4,
        attention_dim=8,
        num_heads=2,
        unary_mode="paper",
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 5, 4)

    _, original = layer.attention_weights(x, x)
    changed_query = x.clone()
    changed_query[0] += 2.0
    _, changed = layer.attention_weights(changed_query, x)

    assert not torch.allclose(original[[0, 2]], changed[[0, 2]])
    torch.testing.assert_close(original[[1, 3]], changed[[1, 3]])


def test_destine_rejects_unknown_unary_mode() -> None:
    with pytest.raises(ValueError, match="unary_mode"):
        DisentangledSelfAttention(4, 4, unary_mode="unknown")


@pytest.mark.parametrize("residual_mode", ["each_layer", "last_layer", "none", None])
def test_build_destine_mixed_fields_forward(residual_mode: str | None) -> None:
    model = build_destine(
        n_num_features=2,
        cardinalities=[4, 5],
        embedding_dim=6,
        attention_dim=8,
        num_heads=2,
        attention_layers=2,
        dnn_hidden_units=[12, 6],
        use_wide=True,
        residual_mode=residual_mode,
        n_outputs=3,
    )
    output = model(
        {
            "num": torch.randn(7, 2),
            "cat": torch.stack(
                [torch.randint(0, 4, (7,)), torch.randint(0, 5, (7,))],
                dim=1,
            ),
        },
    )
    assert isinstance(model, DESTINE)
    assert model.n_fields() == 4
    assert output.shape == (7, 3)


def test_destine_factory_rejects_invalid_attention_width() -> None:
    with pytest.raises(ValueError, match="attention_dim must be divisible"):
        build_destine(
            n_num_features=1,
            cardinalities=[],
            attention_dim=7,
            num_heads=2,
        )


def test_destine_numeric_fields_have_finite_reproducible_initialization() -> None:
    def initialized_parameters() -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(19)
        model = build_destine(
            n_num_features=13,
            cardinalities=[3, 4],
            embedding_dim=8,
            attention_dim=8,
        )
        numeric = model.layers()["num"]
        return numeric._weight.detach().clone(), numeric._bias.detach().clone()

    first_weight, first_bias = initialized_parameters()
    # Disturb the allocator so the assertion would expose an uninitialized
    # ``torch.empty`` parameter rather than accidentally comparing fresh pages.
    _ = [torch.empty(10_000).normal_() for _ in range(4)]
    second_weight, second_bias = initialized_parameters()

    assert torch.isfinite(first_weight).all()
    assert torch.count_nonzero(first_weight) > 0
    torch.testing.assert_close(first_bias, torch.zeros_like(first_bias))
    torch.testing.assert_close(first_weight, second_weight)
    torch.testing.assert_close(first_bias, second_bias)


def test_destine_factory_supports_multihash_and_dense_embedding_fields() -> None:
    model = build_destine(
        n_num_features=1,
        cardinalities=[3],
        embedding_dim=4,
        attention_dim=4,
        multihash_encoder="multihash:cardinality=11;n_hashes=2;embedding_dim=4",
        multihash_n_inputs=4,
        embedding_encoders={
            "entity": "tower:output_dim=5;dropout=0.0;normalize=false",
        },
        embedding_input_dims={"entity": 3},
        use_wide=True,
    )

    output = model(
        {
            "num": torch.randn(6, 1),
            "cat": torch.randint(0, 3, (6, 1)),
            "multihash": torch.randint(0, 11, (6, 4)),
            "entity": torch.randn(6, 3),
        },
    )

    assert model.n_fields() == 5
    assert output.shape == (6,)


def _classification_data(n_rows: int = 48):
    rng = np.random.default_rng(123)
    frame = pl.DataFrame(
        {
            "num": rng.normal(size=n_rows),
            "cat": rng.choice(["a", "b", "c"], size=n_rows),
        },
    )
    target = ((frame["num"].to_numpy() > 0) | (frame["cat"].to_numpy() == "b")).astype(int)
    return frame, target


def test_destine_classifier_is_cloneable_and_trains(tmp_path) -> None:
    X, y = _classification_data()
    estimator = DESTINEClassifier(
        embedding_dim=8,
        attention_dim=8,
        num_heads=2,
        attention_layers=2,
        dnn_hidden_units=[8],
        epochs=2,
        batch_size=16,
        num_features=["num"],
        cat_features=["cat"],
        accelerator_config={"cpu": True},
        random_state=3,
    )
    fitted = clone(estimator).fit(X, y)

    probabilities = fitted.predict_proba(X.head(5))
    assert isinstance(fitted.model_, DESTINE)
    assert probabilities.shape == (5, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    path = tmp_path / "destine.pkl"
    fitted.save(path)
    restored = DESTINEClassifier.load(path)
    np.testing.assert_array_equal(restored.predict_proba(X.head(5)), probabilities)


@pytest.mark.parametrize("dataset", ["criteo", "avazu"])
def test_destine_bars_config_uses_estimator_parameters(dataset) -> None:
    path = Path(__file__).parents[1] / "exps/bars/configs" / f"config_destine_{dataset}_x1.yaml"
    config = yaml.safe_load(path.read_text())
    params = dict(config["model_params"])
    params.pop("eval_metric")  # Converted from the AUC shorthand by the shared adapter.
    model = DESTINEClassifier(**params)
    assert config["model"] == "destine"
    assert clone(model).get_params() == model.get_params()


def test_destine_uses_reference_adam_default_without_changing_dcn_default() -> None:
    assert DESTINEClassifier().optimizer == "adam"
    assert DCNClassifier().optimizer == "adamw"


@pytest.mark.parametrize(
    ("loss", "loss_type"),
    [("bpr", BPRLoss), ("listwise", ListwiseSoftmaxLoss)],
)
def test_destine_ranker_uses_existing_group_aware_training(
    loss: str,
    loss_type: type[BPRLoss | ListwiseSoftmaxLoss],
) -> None:
    X, y = _classification_data(32)
    group = np.repeat(np.arange(8), 4)
    ranker = DESTINERanker(
        loss=loss,
        embedding_dim=4,
        attention_dim=4,
        epochs=1,
        batch_size=8,
        num_features=["num"],
        cat_features=["cat"],
        accelerator_config={"cpu": True},
        random_state=5,
    ).fit(X, y, group=group)

    assert isinstance(ranker.loss_, loss_type)
    assert ranker.predict(X.head(5)).shape == (5,)


def test_destine_ranker_passes_eval_groups_to_group_aware_metric() -> None:
    X, y = _classification_data(32)
    group = np.repeat(np.arange(8), 4)
    seen_groups: list[np.ndarray] = []

    def group_count(y_true: np.ndarray, y_pred: np.ndarray, eval_group: np.ndarray) -> float:
        assert y_true.shape == y_pred.shape == eval_group.shape
        seen_groups.append(eval_group.copy())
        return float(np.unique(eval_group).size)

    ranker = DESTINERanker(
        loss="listwise",
        embedding_dim=4,
        attention_dim=4,
        epochs=1,
        batch_size=8,
        num_features=["num"],
        cat_features=["cat"],
        eval_metric=group_count,
        eval_metric_name="group_count",
        eval_metric_group_aware=True,
        accelerator_config={"cpu": True},
        random_state=11,
    ).fit(
        X,
        y,
        group=group,
        eval_set=(X, y, group),
    )

    assert len(seen_groups) == 1
    assert np.array_equal(seen_groups[0], group)
    assert ranker.history_[0]["val_group_count"] == 8.0
