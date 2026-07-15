from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.exp import (
    infer_checkpoint_mode,
    lightning_monitor_name,
    resolve_checkpoint_selection,
    sync_distributed_flags,
)
from utils.main_utils import update_config
from utils.parser import create_parser, default_parser


def _merge_cli_with_config(config_values: dict[str, object]) -> dict[str, object]:
    args = create_parser().parse_args(
        [
            "--method",
            "EGTR",
            "--dataname",
            "VisualGenome",
            "--config_file",
            "configs/VisualGenome/EGTR_PredCls_ObservedOnly.py",
        ]
    ).__dict__
    merged = update_config(
        args,
        config_values,
        exclude_keys=["method", "val_batch_size", "drop_path", "warmup_epoch"],
    )
    for key, value in default_parser().items():
        if merged.get(key) is None:
            merged[key] = value
    return merged


def test_config_checkpoint_metric_overrides_parser_default() -> None:
    merged = _merge_cli_with_config(
        {
            "method": "EGTR",
            "checkpoint_selection_metric": "val_predcls_mR@100",
            "metric_for_bestckpt": "predcls_mR@100",
            "checkpoint_selection_mode": "max",
        }
    )

    assert merged["checkpoint_selection_metric"] == "val_predcls_mR@100"
    assert merged["metric_for_bestckpt"] == "predcls_mR@100"
    monitor, mode = resolve_checkpoint_selection(SimpleNamespace(**merged))
    assert monitor == "predcls_mR@100"
    assert mode == "max"


def test_protocol_metric_can_map_to_lightning_monitor_name() -> None:
    assert lightning_monitor_name("val_predcls_mR@100") == "predcls_mR@100"
    assert lightning_monitor_name("val_sgdet_R@50") == "sgdet_R@50"
    assert lightning_monitor_name("val_loss") == "val_loss"

    args = SimpleNamespace(
        checkpoint_selection_metric="val_predcls_mR@100",
        metric_for_bestckpt=None,
        checkpoint_selection_mode=None,
    )
    monitor, mode = resolve_checkpoint_selection(args)
    assert monitor == "predcls_mR@100"
    assert mode == "max"


def test_checkpoint_mode_inference_for_loss_and_recall_metrics() -> None:
    assert infer_checkpoint_mode("val_loss") == "min"
    assert infer_checkpoint_mode("test_loss") == "min"
    assert infer_checkpoint_mode("val_predcls_R@50") == "max"
    assert infer_checkpoint_mode("val_predcls_mR@100") == "max"


def test_explicit_checkpoint_mode_is_validated() -> None:
    assert infer_checkpoint_mode("val_predcls_mR@100", "min") == "min"
    try:
        infer_checkpoint_mode("val_predcls_mR@100", "larger")
    except ValueError as exc:
        assert "checkpoint_selection_mode" in str(exc)
    else:
        raise AssertionError("invalid checkpoint_selection_mode was accepted")


def test_multi_gpu_runtime_sets_distributed_sampler_flags() -> None:
    args = SimpleNamespace(device="cuda", gpus=[0, 1], dist=False, distributed=False)
    if not torch.cuda.is_available():
        assert sync_distributed_flags(args) is False
        return

    assert sync_distributed_flags(args) is True
    assert args.dist is True
    assert args.distributed is True


def test_single_gpu_runtime_does_not_force_distributed_sampler_flags() -> None:
    args = SimpleNamespace(device="cuda", gpus=[0], dist=False, distributed=False)
    assert sync_distributed_flags(args) is False
    assert args.dist is False
    assert args.distributed is False


if __name__ == "__main__":
    test_config_checkpoint_metric_overrides_parser_default()
    test_protocol_metric_can_map_to_lightning_monitor_name()
    test_checkpoint_mode_inference_for_loss_and_recall_metrics()
    test_explicit_checkpoint_mode_is_validated()
    test_multi_gpu_runtime_sets_distributed_sampler_flags()
    test_single_gpu_runtime_does_not_force_distributed_sampler_flags()
    print("checkpoint_selection_config_tests_passed")
