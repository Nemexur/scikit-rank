"""Tests for FINAL model components."""

import pytest
import torch

from scikit_rank.modules.dcn import NumericEncoder
from scikit_rank.modules.finalnet import FactorizedInteraction, FinalBlock, FinalNet
from scikit_rank.modules.reducers import Concat


def test_factorized_interaction_sum_matches_formula() -> None:
    layer = FactorizedInteraction(2, 2, residual_type="sum", interaction_activation="identity")
    with torch.no_grad():
        layer._linear.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]] * 2))
        layer._linear.bias.zero_()

    x = torch.tensor([[2.0, 3.0]])
    assert torch.equal(layer(x), torch.tensor([[6.0, 12.0]]))


def test_factorized_interaction_concat_matches_formula() -> None:
    layer = FactorizedInteraction(2, 4, residual_type="concat", interaction_activation="identity")
    with torch.no_grad():
        layer._linear.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]] * 2))
        layer._linear.bias.zero_()

    x = torch.tensor([[2.0, 3.0]])
    assert torch.equal(layer(x), torch.tensor([[2.0, 3.0, 4.0, 9.0]]))


def test_concat_interaction_requires_even_output_width() -> None:
    with pytest.raises(ValueError, match="even output_dim"):
        FactorizedInteraction(2, 3)


def test_final_block_uses_reference_layer_order() -> None:
    block = FinalBlock(
        4,
        [6],
        hidden_activations="gelu",
        dropout=0.1,
        batch_norm=True,
    )
    assert isinstance(block._layers[0], FactorizedInteraction)
    assert isinstance(block._norms[0], torch.nn.BatchNorm1d)
    assert isinstance(block._activations[0], torch.nn.GELU)
    assert isinstance(block._dropouts[0], torch.nn.Dropout)
    assert block(torch.randn(3, 4)).shape == (3, 6)


def test_finalnet_averages_two_branch_logits() -> None:
    first_head = torch.nn.Linear(2, 1)
    second_head = torch.nn.Linear(2, 1)
    with torch.no_grad():
        first_head.weight.copy_(torch.tensor([[1.0, 0.0]]))
        first_head.bias.zero_()
        second_head.weight.copy_(torch.tensor([[0.0, 1.0]]))
        second_head.bias.zero_()
    model = FinalNet(
        layers={"num": NumericEncoder(2)},
        reducer=Concat(),
        block1=torch.nn.Identity(),
        head1=first_head,
        block2=torch.nn.Identity(),
        head2=second_head,
    )

    logits_a, logits_b = model.branch_logits({"num": torch.tensor([[2.0, 4.0]])})
    assert torch.equal(logits_a, torch.tensor([2.0]))
    assert torch.equal(logits_b, torch.tensor([4.0]))
    assert torch.equal(model({"num": torch.tensor([[2.0, 4.0]])}), torch.tensor([3.0]))


def test_finalnet_rejects_incomplete_second_branch() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        FinalNet(
            layers={"num": NumericEncoder(2)},
            reducer=Concat(),
            block1=torch.nn.Identity(),
            head1=torch.nn.Linear(2, 1),
            block2=torch.nn.Identity(),
        )
