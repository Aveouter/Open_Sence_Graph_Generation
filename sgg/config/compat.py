# sgg/config/compat.py
"""
Backward-compatible config loader.

Converts legacy Python config files (configs/VisualGenome/RelTR.py)
to the new TaskConfig dataclass format. This preserves the existing
configuration interface while enabling the new structured config system.
"""

from types import SimpleNamespace
from typing import Any, Dict

from sgg.config.schema import (
    TaskConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
)


def from_legacy_py(py_path: str) -> TaskConfig:
    """
    Load a legacy .py config file and convert to TaskConfig.

    Uses the existing Config._file2dict mechanism to parse the Python file,
    then maps the flat dict to the structured dataclass hierarchy.

    Args:
        py_path: Path to the .py config file.

    Returns:
        TaskConfig with all fields populated.
    """
    from utils.config_utils import Config
    raw = Config._file2dict(py_path)

    # Build sub-configs from flat dict
    dataset = DatasetConfig(
        name=raw.get('dataname', 'VisualGenome'),
        data_root=raw.get('data_root', './data/VisualGenome'),
        num_workers=raw.get('num_workers', 4),
        batch_size=raw.get('batch_size', 4),
        val_batch_size=raw.get('val_batch_size', 4),
        entity_nums=raw.get('num_classes', raw.get('entity_nums', 151)),
        rel_nums=raw.get('num_rel_classes', raw.get('rel_nums', 51)),
        metrics=raw.get('metrics', []),
        dataname=raw.get('dataname', 'VisualGenome'),
    )

    model = ModelConfig(
        name=raw.get('method', 'RelTR').lower(),
        backbone=raw.get('backbone', 'resnet50'),
        hidden_dim=raw.get('hidden_dim', 256),
        dropout=raw.get('dropout', 0.1),
        nheads=raw.get('nheads', 8),
        dim_feedforward=raw.get('dim_feedforward', 2048),
        enc_layers=raw.get('enc_layers', 6),
        dec_layers=raw.get('dec_layers', 6),
        num_entities=raw.get('num_entities', 100),
        num_triplets=raw.get('num_triplets', 200),
        aux_loss=raw.get('aux_loss', True),
        pre_norm=raw.get('pre_norm', False),
        dilation=raw.get('dilation', False),
        position_embedding=raw.get('position_embedding', 'sine'),
        set_cost_class=raw.get('set_cost_class', 1.0),
        set_cost_bbox=raw.get('set_cost_bbox', 5.0),
        set_cost_giou=raw.get('set_cost_giou', 2.0),
        set_iou_threshold=raw.get('set_iou_threshold', 0.7),
        bbox_loss_coef=raw.get('bbox_loss_coef', 5.0),
        giou_loss_coef=raw.get('giou_loss_coef', 2.0),
        rel_loss_coef=raw.get('rel_loss_coef', 1.0),
        eos_coef=raw.get('eos_coef', 0.1),
    )

    optimizer = OptimizerConfig(
        name=raw.get('opt', 'adamw'),
        lr=raw.get('lr', 1e-4),
        lr_backbone=raw.get('lr_backbone', 1e-5),
        weight_decay=raw.get('weight_decay', 1e-4),
        clip_max_norm=raw.get('clip_max_norm', 0.1),
    )

    scheduler = SchedulerConfig(
        name=raw.get('sched', 'cosine'),
        warmup_epoch=raw.get('warmup_epoch', 0),
        warmup_lr=raw.get('warmup_lr', 1e-5),
        min_lr=raw.get('min_lr', 1e-6),
    )

    # Collect recognized keys from all sub-config dataclasses
    _sub_config_keys = set()
    for _dc in [DatasetConfig, ModelConfig, OptimizerConfig, SchedulerConfig]:
        _sub_config_keys.update(_dc.__dataclass_fields__.keys())

    known_keys = _sub_config_keys
    known_keys.update({
        'method', 'dataname', 'num_classes', 'num_rel_classes',
        'res_dir', 'ex_name', 'ckpt_path', 'gpus',
        'data_root', 'metrics', 'opt', 'sched', 'config_file',
        'overwrite', 'res_dir', 'ex_name', 'ckpt_path', 'gpus',
        'drop_path', 'drop', 'epoch', 'log_step', 'fp16',
        'deterministic', 'torchscript', 'fps', 'device',
        'distributed', 'num_nodes', 'metric_for_bestckpt',
        'model_type', 'use_alignment', 'prototype_path',
        'clip_model', 'clip_dim', 'num_hierarchy_levels',
        'hierarchy_weights', 'temperature', 'align_loss_coef',
    })

    extra = {k: v for k, v in raw.items() if k not in known_keys}

    return TaskConfig(
        task_name='image_sgg',
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        max_epochs=raw.get('epoch', 20),
        seed=raw.get('seed', 42),
        test_only=raw.get('test', False),
        ckpt_path=raw.get('ckpt_path', None),
        output_dir=raw.get('res_dir', './results'),
        experiment_name=raw.get('ex_name', 'debug'),
        gpus=raw.get('gpus', [0]),
        extra=extra,
    )


def load_config(path: str) -> SimpleNamespace:
    """
    Universal config loader. Accepts both .py and .yaml paths.

    Args:
        path: Path to config file (.py or .yaml).

    Returns:
        argparse.Namespace compatible with existing code.
    """
    if path.endswith('.yaml') or path.endswith('.yml'):
        from sgg.config.loader import from_yaml
        task_config = from_yaml(path)
    else:
        task_config = from_legacy_py(path)

    return task_config.to_namespace()
