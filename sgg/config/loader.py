# sgg/config/loader.py
"""
YAML config loader for the SGG framework.

Supports loading structured TaskConfig from YAML recipe files.
For legacy .py config files, use sgg.config.compat.from_legacy_py().
"""

from pathlib import Path
from typing import Any, Dict

import yaml

from sgg.config.schema import (
    TaskConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
)


def from_yaml(path: str) -> TaskConfig:
    """
    Load a YAML recipe file and construct a TaskConfig.

    Args:
        path: Path to a .yaml recipe file.

    Returns:
        TaskConfig populated from the YAML file.
    """
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Empty YAML config: {path}")

    # Parse sub-configs
    dataset_raw = raw.get('dataset', {})
    dataset = DatasetConfig(
        name=dataset_raw.get('name', 'visual_genome'),
        data_root=dataset_raw.get('data_root', './data/VisualGenome'),
        num_workers=dataset_raw.get('num_workers', 4),
        batch_size=dataset_raw.get('batch_size', 4),
        val_batch_size=dataset_raw.get('val_batch_size', 4),
        entity_nums=dataset_raw.get('entity_nums', 151),
        rel_nums=dataset_raw.get('rel_nums', 51),
        metrics=dataset_raw.get('metrics', []),
        dataname=dataset_raw.get('dataname', 'VisualGenome'),
    )

    model_raw = raw.get('model', {})
    model = ModelConfig(
        name=model_raw.get('name', 'reltr'),
        backbone=model_raw.get('backbone', 'resnet50'),
        hidden_dim=model_raw.get('hidden_dim', 256),
        dropout=model_raw.get('dropout', 0.1),
        nheads=model_raw.get('nheads', 8),
        dim_feedforward=model_raw.get('dim_feedforward', 2048),
        enc_layers=model_raw.get('enc_layers', 6),
        dec_layers=model_raw.get('dec_layers', 6),
        num_entities=model_raw.get('num_entities', 100),
        num_triplets=model_raw.get('num_triplets', 200),
        aux_loss=model_raw.get('aux_loss', True),
        pre_norm=model_raw.get('pre_norm', False),
        dilation=model_raw.get('dilation', False),
        position_embedding=model_raw.get('position_embedding', 'sine'),
        set_cost_class=model_raw.get('set_cost_class', 1.0),
        set_cost_bbox=model_raw.get('set_cost_bbox', 5.0),
        set_cost_giou=model_raw.get('set_cost_giou', 2.0),
        set_iou_threshold=model_raw.get('set_iou_threshold', 0.7),
        bbox_loss_coef=model_raw.get('bbox_loss_coef', 5.0),
        giou_loss_coef=model_raw.get('giou_loss_coef', 2.0),
        rel_loss_coef=model_raw.get('rel_loss_coef', 1.0),
        eos_coef=model_raw.get('eos_coef', 0.1),
    )

    opt_raw = raw.get('optimizer', {})
    optimizer = OptimizerConfig(
        name=opt_raw.get('name', 'adamw'),
        lr=opt_raw.get('lr', 1e-4),
        lr_backbone=opt_raw.get('lr_backbone', 1e-5),
        weight_decay=opt_raw.get('weight_decay', 1e-4),
        clip_max_norm=opt_raw.get('clip_max_norm', 0.1),
    )

    sched_raw = raw.get('scheduler', {})
    scheduler = SchedulerConfig(
        name=sched_raw.get('name', 'cosine'),
        warmup_epoch=sched_raw.get('warmup_epoch', 0),
        warmup_lr=sched_raw.get('warmup_lr', 1e-5),
        min_lr=sched_raw.get('min_lr', 1e-6),
    )

    return TaskConfig(
        task_name=raw.get('task_name', 'image_sgg'),
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        max_epochs=raw.get('max_epochs', 20),
        seed=raw.get('seed', 42),
        test_only=raw.get('test_only', False),
        ckpt_path=raw.get('ckpt_path', None),
        output_dir=raw.get('output_dir', './results'),
        experiment_name=raw.get('experiment_name', 'debug'),
        gpus=raw.get('gpus', [0]),
    )
