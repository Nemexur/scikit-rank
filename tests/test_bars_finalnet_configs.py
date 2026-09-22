from __future__ import annotations
import inspect
from pathlib import Path
from typing import Any

import pytest

from exps.bars.train import ADAPTERS, load_config
from scikit_rank import FinalNetClassifier

CONFIG_DIR = Path(__file__).parents[1] / "exps/bars/configs"


@pytest.mark.parametrize(
    ("dataset", "reference"),
    [
        (
            "avazu_x1",
            {
                "block1_hidden_units": [800],
                "block1_dropout": 0.3,
                "block2_hidden_units": [800, 800],
                "block2_dropout": 0.2,
            },
        ),
        (
            "criteo_x1",
            {
                "block1_hidden_units": [400, 400],
                "block1_dropout": 0.1,
                "block2_hidden_units": [400, 400],
                "block2_dropout": 0.0,
            },
        ),
    ],
)
def test_finalnet_2b_configs_capture_selected_recbench_runs(
    dataset: str,
    reference: dict[str, Any],
) -> None:
    config = load_config(CONFIG_DIR / f"config_finalnet_2b_{dataset}.yaml")
    params = config["model_params"]

    assert config["model"] == "finalnet"
    assert set(config) == {
        "model",
        "dataset",
        "num_features",
        "cat_features",
        "data_dir",
        "max_train_rows",
        "verbose",
        "deterministic",
        "model_params",
    }
    assert params["block_type"] == "2B"
    assert params["block1_hidden_units"] == reference["block1_hidden_units"]
    assert params["block1_dropout"] == reference["block1_dropout"]
    assert params["block2_hidden_units"] == reference["block2_hidden_units"]
    assert params["block2_dropout"] == reference["block2_dropout"]
    assert params["use_field_gate"] is True
    assert params["use_2b_consistency_loss"] is True
    assert params["interaction_activation"] is None
    assert set(params) <= set(inspect.signature(FinalNetClassifier).parameters)


def test_bars_runner_registers_finalnet_classifier() -> None:
    adapter = ADAPTERS["finalnet"]

    assert adapter.estimator_class == "FinalNetClassifier"
