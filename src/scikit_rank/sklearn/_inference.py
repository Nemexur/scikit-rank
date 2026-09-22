"""Shared tabular-model inference for sklearn estimator families."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch

if TYPE_CHECKING:
    from scikit_rank.preprocessing import TabularPreprocessor


def score_tabular_model(
    *,
    model: torch.nn.Module,
    preprocessor: TabularPreprocessor,
    frame: pl.DataFrame | pl.LazyFrame,
    batch_size: int,
    chunk_rows: int,
    accelerator_config: dict[str, object] | None,
) -> np.ndarray:
    """Score an eager or lazy feature frame and return model logits on CPU.

    Estimator families retain responsibility for input validation and task-level
    conversion of logits to predictions. This function owns only the established
    tabular batch protocol: preprocessor output becomes a tensor batch keyed by
    feature stream name. Models may return logits directly or as the first item
    of a tuple carrying additional training outputs.
    """
    device = _inference_device(accelerator_config)
    model.to(device).eval()
    try:
        if isinstance(frame, pl.LazyFrame):
            n_rows = frame.select(pl.len()).collect().item()
            chunks = [
                _score_frame(
                    model=model,
                    preprocessor=preprocessor,
                    frame=frame.slice(start, chunk_rows).collect(),
                    batch_size=batch_size,
                    device=device,
                )
                for start in range(0, n_rows, chunk_rows)
            ]
            return np.concatenate(chunks, axis=0)
        return _score_frame(
            model=model,
            preprocessor=preprocessor,
            frame=frame,
            batch_size=batch_size,
            device=device,
        )
    finally:
        model.cpu()


def _inference_device(accelerator_config: dict[str, object] | None) -> torch.device:
    if accelerator_config and accelerator_config.get("cpu"):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _score_frame(
    *,
    model: torch.nn.Module,
    preprocessor: TabularPreprocessor,
    frame: pl.DataFrame,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    num, cat, extra = preprocessor.transform(frame)
    outputs = []
    for start in range(0, len(num), batch_size):
        stop = start + batch_size
        batch = {
            "num": torch.from_numpy(num[start:stop]).to(device),
            "cat": torch.from_numpy(cat[start:stop]).to(device),
        }
        batch.update(
            {name: torch.from_numpy(arr[start:stop]).to(device) for name, arr in extra.items()},
        )
        model_output = model(batch)
        logits = model_output[0] if isinstance(model_output, tuple) else model_output
        outputs.append(logits.float().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0,), dtype=np.float32)
