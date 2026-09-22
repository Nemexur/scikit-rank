"""TabM composition over scikit-rank feature encoders."""

from collections.abc import Iterator
from typing import Literal

import tabm as tabm_lib
import torch

from scikit_rank.modules.losses import Loss

EnsembleAggregation = Literal[
    "mean",
    "binary_probability",
    "multiclass_probability",
    "coral_probability",
]


def aggregate_member_logits(
    member_logits: torch.Tensor,
    aggregation: EnsembleAggregation,
) -> torch.Tensor:
    """Aggregate TabM members while preserving the estimator score contract."""
    if member_logits.ndim < 2:
        raise ValueError(
            "TabM member logits must have shape [batch, members, ...], "
            f"got {tuple(member_logits.shape)}",
        )
    if aggregation == "mean":
        return member_logits.mean(dim=1)
    if aggregation == "binary_probability":
        if member_logits.ndim != 2:
            raise ValueError("Binary TabM logits must have shape [batch, members]")
        probability = member_logits.sigmoid().mean(dim=1)
        eps = torch.finfo(probability.dtype).eps
        return torch.logit(probability.clamp(eps, 1.0 - eps))
    if aggregation == "multiclass_probability":
        if member_logits.ndim != 3:
            raise ValueError(
                "Multiclass TabM logits must have shape [batch, members, classes]",
            )
        probability = member_logits.softmax(dim=-1).mean(dim=1)
        return probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
    if aggregation == "coral_probability":
        if member_logits.ndim != 3:
            raise ValueError(
                "CORAL TabM logits must have shape [batch, members, levels]",
            )
        probability = member_logits.sigmoid().mean(dim=1)
        eps = torch.finfo(probability.dtype).eps
        return torch.logit(probability.clamp(eps, 1.0 - eps))
    raise ValueError(f"Unknown TabM aggregation: {aggregation!r}")


class CoralEnsemble(torch.nn.Module):
    """One CORAL head per TabM member with a vectorized shared-weight projection."""

    is_coral = True

    def __init__(self, in_features: int, num_classes: int, *, k: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("CoralEnsemble requires at least two classes")
        self._projection = tabm_lib.LinearEnsemble(in_features, 1, bias=False, k=k)
        n_levels = num_classes - 1
        initial_bias = torch.arange(n_levels, 0, -1).float() / n_levels
        self._biases = torch.nn.Parameter(initial_bias.expand(k, -1).clone())

    @property
    def k(self) -> int:
        return self._projection.k

    def output_dim(self) -> int:
        return self._biases.size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._projection(x) + self._biases.unsqueeze(0)


class TabM(torch.nn.Module):
    """TabM/TabM-mini backbone with scikit-rank feature streams.

    :meth:`forward` returns aggregated logits followed by the member logits used
    by :class:`TabMEnsembleLoss`.
    """

    def __init__(
        self,
        *,
        layers: dict[str, torch.nn.Module] | torch.nn.ModuleDict,
        reducer: torch.nn.Module,
        backbone: torch.nn.Module,
        ensemble_view: torch.nn.Module,
        head: torch.nn.Module,
        aggregation: EnsembleAggregation,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("layers must contain at least one entry")
        self._layers = (
            layers if isinstance(layers, torch.nn.ModuleDict) else torch.nn.ModuleDict(layers)
        )
        self._reducer = reducer
        self._backbone = backbone
        self._ensemble_view = ensemble_view
        self._head = head
        self._aggregation = aggregation

    @property
    def k(self) -> int:
        return self._backbone.k

    def layers(self) -> torch.nn.ModuleDict:
        return self._layers

    def reducer(self) -> torch.nn.Module:
        return self._reducer

    def backbone(self) -> torch.nn.Module:
        return self._backbone

    def head(self) -> torch.nn.Module:
        return self._head

    def embedding_parameters(self) -> Iterator[torch.nn.Parameter]:
        return self._layers.parameters()

    def forward_members(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = {name: layer(inputs[name]) for name, layer in self._layers.items()}
        x = self._ensemble_view(self._reducer(encoded))
        output = self._head(self._backbone(x))
        return (
            output.squeeze(-1)
            if output.size(-1) == 1 and self._aggregation != "coral_probability"
            else output
        )

    def aggregate_members(self, member_logits: torch.Tensor) -> torch.Tensor:
        return aggregate_member_logits(member_logits, self._aggregation)

    def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        member_logits = self.forward_members(inputs)
        return self.aggregate_members(member_logits), member_logits


class TabMEnsembleLoss(Loss):
    """Apply a base loss to every TabM member and average the results."""

    def __init__(self, loss_fn: Loss) -> None:
        super().__init__()
        self._loss_fn = loss_fn

    @property
    def requires_group(self) -> bool:
        return self._loss_fn.requires_group

    def unwrap(self) -> Loss:
        """Return the public base loss."""
        return self._loss_fn

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
        *,
        extras: tuple[torch.Tensor, ...] = (),
    ) -> torch.Tensor:
        if len(extras) != 1:
            raise ValueError(
                "TabM ensemble loss requires exactly one member-logits tensor, "
                f"got {len(extras)} extras",
            )
        member_logits = extras[0]
        if member_logits.ndim < 2:
            raise ValueError(
                "TabM member logits must have shape [batch, members, ...], "
                f"got {tuple(member_logits.shape)}",
            )
        if scores.size(0) != member_logits.size(0):
            raise ValueError(
                "Aggregated and member logits have inconsistent batch lengths: "
                f"{scores.size(0)} != {member_logits.size(0)}",
            )
        return torch.stack(
            [
                self._loss_fn(member_logits[:, member], target, group)
                for member in range(member_logits.size(1))
            ],
        ).mean()
