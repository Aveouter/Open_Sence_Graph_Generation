# sgg/config/schema.py
"""
Configuration dataclass schemas for the SGG framework.

Provides typed, validated configuration objects with sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetConfig:
    """Dataset-specific configuration."""
    name: str = 'visual_genome'
    data_root: str = './data/VisualGenome'
    num_workers: int = 4
    batch_size: int = 4
    val_batch_size: int = 4
    image_size: List[int] = field(default_factory=lambda: [224, 224])
    entity_nums: int = 151
    rel_nums: int = 51
    metrics: List[str] = field(default_factory=lambda: [
        "sgdet_R@10", "sgdet_R@20", "sgdet_R@50",
        "sgdet_mR@10", "sgdet_mR@20", "sgdet_mR@50",
        "predcls_R@10", "predcls_R@20",
        "predcls_mR@10", "predcls_mR@20",
        "sgcls_R@10", "sgcls_R@20",
        "sgcls_mR@10", "sgcls_mR@20",
    ])
    dataname: str = 'VisualGenome'  # Legacy compat


@dataclass
class ModelConfig:
    """Model-specific configuration."""
    name: str = 'reltr'
    backbone: str = 'resnet50'
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    dim_feedforward: int = 2048
    enc_layers: int = 6
    dec_layers: int = 6
    num_entities: int = 100
    num_triplets: int = 200
    aux_loss: bool = True
    pre_norm: bool = False
    dilation: bool = False
    return_interm_layers: bool = False
    position_embedding: str = 'sine'

    # Matcher
    set_cost_class: float = 1.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0
    set_iou_threshold: float = 0.7

    # Loss weights
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    rel_loss_coef: float = 1.0
    eos_coef: float = 0.1


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = 'adamw'
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    clip_max_norm: float = 0.1
    betas: tuple = (0.9, 0.999)
    momentum: float = 0.9


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    name: str = 'cosine'
    warmup_epoch: int = 0
    warmup_lr: float = 1e-5
    min_lr: float = 1e-6
    decay_epoch: int = 10
    decay_rate: float = 0.1
    final_div_factor: float = 1e4


@dataclass
class TaskConfig:
    """Top-level task configuration, composing all sub-configs."""
    task_name: str = 'image_sgg'
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Experiment
    max_epochs: int = 20
    seed: int = 42
    test_only: bool = False
    ckpt_path: Optional[str] = None
    output_dir: str = './results'
    experiment_name: str = 'debug'
    gpus: List[int] = field(default_factory=lambda: [0])

    # Extra kwargs for backward compatibility
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_namespace(self):
        """Convert to argparse.Namespace for backward compatibility."""
        from types import SimpleNamespace
        result = {}
        # Flatten all configs into a single dict
        for field_name in ['dataset', 'model', 'optimizer', 'scheduler']:
            sub = getattr(self, field_name)
            for k, v in sub.__dict__.items():
                result[k] = v
        # Top-level fields
        for k, v in self.__dict__.items():
            if k not in ('dataset', 'model', 'optimizer', 'scheduler'):
                result[k] = v

        # Legacy name mappings (old code uses 'method' and 'dataname')
        result['method'] = self.model.name
        result['dataname'] = self.dataset.dataname

        # Merge extra
        result.update(self.extra)
        return SimpleNamespace(**result)
