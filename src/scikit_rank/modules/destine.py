"""DESTINE components for disentangled field-wise self-attention.

The implementation follows the equations from Xu et al. (CIKM 2021):
queries and keys are centered across feature fields for the pairwise term,
while an independent unary term measures each field's sample-dependent importance.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class NumericFieldEmbeddings(torch.nn.Module):
    """Embed every scalar numeric feature as one DESTINE field."""

    def __init__(self, n_features: int, embedding_dim: int) -> None:
        super().__init__()
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        self._weight = torch.nn.Parameter(torch.empty(n_features, embedding_dim))
        self._bias = torch.nn.Parameter(torch.empty(n_features, embedding_dim))
        self._n_fields = n_features
        self._embedding_dim = embedding_dim
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize each scalar-to-field projection like ``Linear(1, d)``."""
        std = (2.0 / (1 + self._embedding_dim)) ** 0.5
        torch.nn.init.normal_(self._weight, mean=0.0, std=std)
        torch.nn.init.zeros_(self._bias)

    def output_dim(self) -> int:
        return self._n_fields * self._embedding_dim

    def n_fields(self) -> int:
        return self._n_fields

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self._weight + self._bias


class ReshapedFieldEncoder(torch.nn.Module):
    """Adapt a flattened encoder to ``[batch, fields, embedding_dim]``."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        *,
        n_fields: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        expected = n_fields * embedding_dim
        if not hasattr(encoder, "output_dim") or encoder.output_dim() != expected:
            actual = encoder.output_dim() if hasattr(encoder, "output_dim") else None
            raise ValueError(
                "field encoder must emit n_fields * embedding_dim values: "
                f"expected {expected}, got {actual}",
            )
        self._encoder = encoder
        self._n_fields = n_fields
        self._embedding_dim = embedding_dim

    def output_dim(self) -> int:
        return self._n_fields * self._embedding_dim

    def n_fields(self) -> int:
        return self._n_fields

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encoder(x)
        return encoded.reshape(encoded.size(0), self._n_fields, self._embedding_dim)


class MultiHashFieldEncoder(torch.nn.Module):
    """Turn multiple hashes per source feature into one averaged field."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        *,
        n_inputs: int,
        n_hashes: int,
        input_embedding_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if n_inputs % n_hashes != 0:
            raise ValueError(
                f"multihash n_inputs={n_inputs} must be divisible by n_hashes={n_hashes}",
            )
        self._encoder = encoder
        self._n_inputs = n_inputs
        self._n_hashes = n_hashes
        self._n_fields = n_inputs // n_hashes
        self._input_embedding_dim = input_embedding_dim
        self._embedding_dim = embedding_dim
        self._projection = (
            torch.nn.Identity()
            if input_embedding_dim == embedding_dim
            else torch.nn.Linear(input_embedding_dim, embedding_dim, bias=False)
        )

    def output_dim(self) -> int:
        return self._n_fields * self._embedding_dim

    def n_fields(self) -> int:
        return self._n_fields

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encoder(x).reshape(
            x.size(0),
            self._n_inputs,
            self._input_embedding_dim,
        )
        encoded = self._projection(encoded)
        return encoded.reshape(
            x.size(0),
            self._n_fields,
            self._n_hashes,
            self._embedding_dim,
        ).mean(dim=2)


class DenseFieldEncoder(torch.nn.Module):
    """Encode one dense external-vector stream as one DESTINE field."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        *,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if not hasattr(encoder, "output_dim"):
            raise TypeError("dense field encoder must expose output_dim()")
        input_dim = int(encoder.output_dim())
        self._encoder = encoder
        self._projection = (
            torch.nn.Identity()
            if input_dim == embedding_dim
            else torch.nn.Linear(input_dim, embedding_dim, bias=False)
        )
        self._embedding_dim = embedding_dim

    def output_dim(self) -> int:
        return self._embedding_dim

    def n_fields(self) -> int:
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._projection(self._encoder(x)).unsqueeze(1)


class DisentangledSelfAttention(torch.nn.Module):
    """Multi-head attention with separately normalized pairwise and unary scores."""

    def __init__(
        self,
        embedding_dim: int,
        attention_dim: int,
        num_heads: int = 2,
        dropout: float = 0.0,
        *,
        use_residual: bool = True,
        use_scale: bool = True,
        relu_before_attention: bool = False,
        unary_mode: Literal["paper", "static"] = "paper",
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or attention_dim <= 0:
            raise ValueError("embedding_dim and attention_dim must be positive")
        if num_heads <= 0 or attention_dim % num_heads != 0:
            raise ValueError(
                "attention_dim must be divisible by a positive num_heads, got "
                f"attention_dim={attention_dim}, num_heads={num_heads}",
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if unary_mode not in ("paper", "static"):
            raise ValueError(
                f"unary_mode must be 'paper' or 'static', got {unary_mode!r}",
            )

        self._attention_dim = attention_dim
        self._num_heads = num_heads
        self._head_dim = attention_dim // num_heads
        self._use_scale = use_scale
        self._relu_before_attention = relu_before_attention
        self._unary_mode = unary_mode
        self._query = torch.nn.Linear(embedding_dim, attention_dim)
        self._key = torch.nn.Linear(embedding_dim, attention_dim)
        self._value = torch.nn.Linear(embedding_dim, attention_dim)
        self._unary_query = (
            torch.nn.Linear(embedding_dim, attention_dim) if unary_mode == "paper" else None
        )
        self._unary = torch.nn.Linear(embedding_dim, num_heads) if unary_mode == "static" else None
        self._residual = torch.nn.Linear(embedding_dim, attention_dim) if use_residual else None
        self._dropout = torch.nn.Dropout(dropout)

    def attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return separately normalized pairwise and unary attention weights."""
        projected_query = self._query(query)
        projected_key = self._key(key)
        if self._relu_before_attention:
            projected_query = F.relu(projected_query)
            projected_key = F.relu(projected_key)

        q = torch.cat(projected_query.split(self._head_dim, dim=-1), dim=0)
        k = torch.cat(projected_key.split(self._head_dim, dim=-1), dim=0)
        uncentered_key = k
        q = q - q.mean(dim=1, keepdim=True)
        k = k - k.mean(dim=1, keepdim=True)
        pairwise_logits = torch.bmm(q, k.transpose(1, 2))
        if self._use_scale:
            pairwise_logits = pairwise_logits / math.sqrt(self._head_dim)
        pairwise = F.softmax(pairwise_logits, dim=-1)

        if self._unary_mode == "paper":
            assert self._unary_query is not None
            unary_query = self._unary_query(query)
            if self._relu_before_attention:
                unary_query = F.relu(unary_query)
            unary_query = torch.cat(unary_query.split(self._head_dim, dim=-1), dim=0)
            mean_unary_query = unary_query.mean(dim=1, keepdim=True)
            unary_logits = torch.bmm(
                mean_unary_query,
                uncentered_key.transpose(1, 2),
            )
            unary = F.softmax(unary_logits, dim=-1)
        else:
            assert self._unary is not None
            unary = F.softmax(self._unary(key), dim=1)
            unary = unary.permute(2, 0, 1).reshape(-1, 1, key.size(1))
        return pairwise, unary

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        residual = query
        pairwise, unary = self.attention_weights(query, key)
        projected_value = self._value(value)
        if self._relu_before_attention:
            projected_value = F.relu(projected_value)
        v = torch.cat(projected_value.split(self._head_dim, dim=-1), dim=0)
        output = torch.bmm(self._dropout(pairwise + unary), v)
        output = torch.cat(output.split(query.size(0), dim=0), dim=-1)
        if self._residual is not None:
            output = output + self._residual(residual)
        return output


class DESTINEWide(torch.nn.Module):
    """Memory-efficient linear term over the raw numeric/categorical fields."""

    def __init__(
        self,
        *,
        n_num_features: int,
        cardinalities: Sequence[int],
        n_outputs: int,
        multihash_cardinality: int | None = None,
        multihash_n_inputs: int = 0,
        multihash_n_hashes: int = 1,
        embedding_input_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self._num = (
            torch.nn.Linear(n_num_features, n_outputs, bias=False) if n_num_features else None
        )
        self._cat = torch.nn.Embedding(sum(cardinalities), n_outputs) if cardinalities else None
        offsets = torch.tensor([0, *cardinalities[:-1]], dtype=torch.long).cumsum(0)
        self.register_buffer("_cat_offsets", offsets)
        self._multihash = (
            torch.nn.Embedding(multihash_cardinality, n_outputs)
            if multihash_cardinality is not None and multihash_n_inputs
            else None
        )
        self._multihash_n_inputs = multihash_n_inputs
        self._multihash_n_hashes = multihash_n_hashes
        self._dense = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(width, n_outputs, bias=False)
                for name, width in (embedding_input_dims or {}).items()
            },
        )
        self._bias = torch.nn.Parameter(torch.zeros(n_outputs))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = inputs["num"].size(0)
        output = self._bias.unsqueeze(0).expand(batch_size, -1)
        if self._num is not None:
            output = output + self._num(inputs["num"])
        if self._cat is not None:
            output = output + self._cat(inputs["cat"] + self._cat_offsets).sum(dim=1)
        if self._multihash is not None:
            hashed = self._multihash(inputs["multihash"])
            hashed = hashed.reshape(
                batch_size,
                self._multihash_n_inputs // self._multihash_n_hashes,
                self._multihash_n_hashes,
                -1,
            )
            output = output + hashed.mean(dim=2).sum(dim=1)
        for name, linear in self._dense.items():
            output = output + linear(inputs[name])
        return output


class DESTINE(torch.nn.Module):
    """Disentangled Self-Attentive Network over tabular feature fields."""

    def __init__(
        self,
        *,
        layers: dict[str, torch.nn.Module],
        attention: Sequence[torch.nn.Module],
        last_residual: torch.nn.Module | None = None,
        attention_activation: bool,
        head: torch.nn.Module,
        dnn: torch.nn.Module | None = None,
        dnn_head: torch.nn.Module | None = None,
        wide: DESTINEWide | None = None,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("DESTINE needs at least one input field")
        if not attention:
            raise ValueError("attention must contain at least one layer")
        if (dnn is None) != (dnn_head is None):
            raise ValueError("dnn and dnn_head must be supplied together")
        for name, layer in layers.items():
            if not hasattr(layer, "n_fields"):
                raise TypeError(f"DESTINE layer {name!r} must expose n_fields()")

        self._layers = torch.nn.ModuleDict(layers)
        self._n_fields = sum(int(layer.n_fields()) for layer in layers.values())
        self._attention = torch.nn.ModuleList(attention)
        self._last_residual = last_residual
        self._attention_activation = attention_activation
        self._head = head
        self._dnn = dnn
        self._dnn_head = dnn_head
        self._wide = wide

    def layers(self) -> torch.nn.ModuleDict:
        return self._layers

    def head(self) -> torch.nn.Module:
        return self._head

    def n_fields(self) -> int:
        return self._n_fields

    def embedding_parameters(self) -> Iterator[torch.nn.Parameter]:
        return self._layers.parameters()

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        feature_embeddings = torch.cat(
            [layer(inputs[name]) for name, layer in self._layers.items()],
            dim=1,
        )
        cross = feature_embeddings
        for attention in self._attention:
            cross = attention(cross, cross, cross)
        if self._last_residual is not None:
            cross = cross + self._last_residual(feature_embeddings)
        if self._attention_activation:
            cross = F.relu(cross)
        output = self._head(cross.flatten(start_dim=1))
        if self._dnn is not None and self._dnn_head is not None:
            output = output + self._dnn_head(self._dnn(feature_embeddings.flatten(start_dim=1)))
        if self._wide is not None:
            output = output + self._wide(inputs)
        return output.squeeze(-1) if output.ndim > 1 and output.size(-1) == 1 else output
