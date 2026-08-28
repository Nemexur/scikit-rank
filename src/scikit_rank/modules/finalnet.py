"""FINAL model components.

FINAL (Factorized Interaction Layer for CTR Prediction, Zhu et al., SIGIR 2023)
models higher-order feature interactions by stacking factorized interaction
layers. Feature encoders and their reducer are supplied by a factory rather
than being constructed by the model itself.
"""

from __future__ import annotations
from collections.abc import Callable, Iterator, Sequence

import torch

# Registry for the activations used inside and between FINAL interaction layers.
FINALNET_ACTIVATION_BUILDERS: dict[str, Callable[[], torch.nn.Module]] = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
    "swish": torch.nn.SiLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "identity": torch.nn.Identity,
}


class FactorizedInteraction(torch.nn.Module):
    """One factorized interaction layer from FINAL.

    ``residual_type='sum'`` computes ``h2 + h1 * h2``. ``'concat'``
    concatenates ``h2`` and ``h1 * h2`` and therefore has the same parameter
    count as a plain ``Linear(input_dim, output_dim)``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        bias: bool = True,
        residual_type: str = "concat",
        interaction_activation: str | None = "relu",
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if residual_type not in ("sum", "concat"):
            raise ValueError("residual_type must be 'sum' or 'concat'")
        if residual_type == "concat" and output_dim % 2:
            raise ValueError("residual_type='concat' requires an even output_dim")
        if (
            interaction_activation is not None
            and interaction_activation not in FINALNET_ACTIVATION_BUILDERS
        ):
            raise ValueError(
                "interaction_activation must be one of "
                f"{list(FINALNET_ACTIVATION_BUILDERS)}, got {interaction_activation!r}",
            )

        self._residual_type = residual_type
        linear_output_dim = output_dim * 2 if residual_type == "sum" else output_dim
        self._linear = torch.nn.Linear(input_dim, linear_output_dim, bias=bias)
        self._activation = (
            torch.nn.Identity()
            if interaction_activation is None
            else FINALNET_ACTIVATION_BUILDERS[interaction_activation]()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h2, h1 = self._linear(x).chunk(2, dim=-1)
        h1 = self._activation(h1)
        h2 = self._activation(h2)
        interaction = h1 * h2
        if self._residual_type == "sum":
            return h2 + interaction
        return torch.cat([h2, interaction], dim=-1)


class FinalBlock(torch.nn.Module):
    """A stack of FINAL interaction layers.

    Each layer follows the reference order: factorized interaction, optional
    batch normalization, hidden activation, then dropout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: Sequence[int],
        *,
        hidden_activations: str | Sequence[str | None] | None = None,
        dropout: float | Sequence[float] = 0.0,
        batch_norm: bool = True,
        residual_type: str = "concat",
        interaction_activation: str | None = "relu",
    ) -> None:
        super().__init__()
        if not hidden_units:
            raise ValueError("hidden_units must not be empty")
        n_layers = len(hidden_units)

        if isinstance(dropout, Sequence) and not isinstance(dropout, str):
            dropout_rates = list(dropout)
            if len(dropout_rates) != n_layers:
                raise ValueError(f"dropout must have length {n_layers}, got {len(dropout_rates)}")
        else:
            dropout_rates = [dropout] * n_layers
        if any(rate < 0.0 or rate >= 1.0 for rate in dropout_rates):
            raise ValueError("dropout rates must be in [0, 1)")

        if isinstance(hidden_activations, Sequence) and not isinstance(hidden_activations, str):
            activations = list(hidden_activations)
            if len(activations) != n_layers:
                raise ValueError(
                    f"hidden_activations must have length {n_layers}, got {len(activations)}",
                )
        else:
            activations = [hidden_activations] * n_layers
        allowed_activations = (None, *FINALNET_ACTIVATION_BUILDERS)
        unknown = {name for name in activations if name not in allowed_activations}
        if unknown:
            raise ValueError(
                "hidden_activations must contain only "
                f"{list(FINALNET_ACTIVATION_BUILDERS)} or None, got {sorted(unknown)!r}",
            )

        dims = [input_dim, *hidden_units]
        self._layers = torch.nn.ModuleList(
            [
                FactorizedInteraction(
                    dims[i],
                    dims[i + 1],
                    residual_type=residual_type,
                    interaction_activation=interaction_activation,
                )
                for i in range(n_layers)
            ],
        )
        self._norms = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(dim) if batch_norm else torch.nn.Identity()
                for dim in hidden_units
            ],
        )
        self._activations = torch.nn.ModuleList(
            [
                torch.nn.Identity() if name is None else FINALNET_ACTIVATION_BUILDERS[name]()
                for name in activations
            ],
        )
        self._dropouts = torch.nn.ModuleList(
            [torch.nn.Dropout(rate) if rate > 0 else torch.nn.Identity() for rate in dropout_rates],
        )
        self._output_dim = hidden_units[-1]

    def output_dim(self) -> int:
        """Return the width produced by the last interaction layer."""
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, norm, activation, dropout in zip(
            self._layers,
            self._norms,
            self._activations,
            self._dropouts,
            strict=True,
        ):
            x = dropout(activation(norm(layer(x))))
        return x


class FinalNet(torch.nn.Module):
    """FINAL composition root with either one or two parallel blocks.

    The injected ``layers`` encode named input streams and ``reducer`` fuses
    them. With two blocks, :meth:`forward` returns their averaged logits, while
    :meth:`branch_logits` exposes each branch for a future distillation-aware
    training interface.
    """

    def __init__(
        self,
        *,
        layers: dict[str, torch.nn.Module],
        reducer: torch.nn.Module,
        block1: torch.nn.Module,
        head1: torch.nn.Module,
        block2: torch.nn.Module | None = None,
        head2: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("layers must contain at least one entry")
        if (block2 is None) != (head2 is None):
            raise ValueError("block2 and head2 must be supplied together")
        self._layers = torch.nn.ModuleDict(layers)
        self._reducer = reducer
        self._block1 = block1
        self._head1 = head1
        self._block2 = block2
        self._head2 = head2

    def layers(self) -> torch.nn.ModuleDict:
        """Return the named feature encoders."""
        return self._layers

    def reducer(self) -> torch.nn.Module:
        """Return the feature-stream reducer."""
        return self._reducer

    def embedding_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Yield feature-encoder parameters for embedding-only regularization."""
        return self._layers.parameters()

    def branch_logits(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return first-branch logits and optional second-branch logits."""
        encoded = {name: layer(inputs[name]) for name, layer in self._layers.items()}
        x = self._reducer(encoded)
        first = self._head1(self._block1(x)).squeeze(-1)
        if self._block2 is None or self._head2 is None:
            return first, None
        return first, self._head2(self._block2(x)).squeeze(-1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        first, second = self.branch_logits(inputs)
        return first if second is None else (first + second) * 0.5
