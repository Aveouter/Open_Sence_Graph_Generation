# sgg/__init__.py
"""
SGG Framework — Standardized Scene Graph Generation Toolkit
============================================================

A modular, extensible framework for scene graph generation research,
supporting image, video, panoptic, and 3D SGG tasks.

## Package Structure

    sgg/
    ├── config/         Configuration system (YAML + Schema + Recipes)
    ├── framework/      Abstract base classes (BaseModel, BaseDataset, etc.)
    ├── tasks/          Task implementations (ImageSGG, VideoSGG, ...)
    ├── models/         Model adapters (wrapping raw nn.Modules)
    ├── modules/        Raw network layer implementations
    ├── losses/         Loss functions and post-processors
    ├── data/           Dataset loading, transforms, DataModules
    ├── evaluation/     Evaluation metrics, adapters, result formatting
    ├── training/       Optimizers, schedulers, callbacks
    ├── utils/          General-purpose utilities
    └── registry.py     Universal component registry

## Quick Start

    # Load config
    from sgg.config.compat import from_legacy_py
    args = from_legacy_py('configs/VisualGenome/RelTR.py').to_namespace()

    # Build model via adapter
    from sgg.models.reltr_adapter import RelTRAdapter
    adapter = RelTRAdapter()
    adapter.build(args)

    # Or use registry
    from sgg.registry import MODEL_REGISTRY
    AdapterCls = MODEL_REGISTRY.get('reltr')
    adapter = AdapterCls(); adapter.build(args)

## Adding a New Model

    @MODEL_REGISTRY.register('my_model')
    class MyModelAdapter(BaseModelAdapter):
        def build(self, config): ...
        def forward(self, inputs, targets=None): ...
        # ... implement abstract methods

## Adding a New Dataset

    @DATASET_REGISTRY.register('my_dataset')
    class MyDataset(BaseSGGDataset):
        @property
        def num_entity_classes(self): return N
        @property
        def num_predicate_classes(self): return M
        def build_loaders(self, **kwargs): ...

Version: 2.0.0
"""

__version__ = "2.0.0"

# Re-export core APIs for convenience
from sgg.registry import (
    Registry,
    MODEL_REGISTRY,
    DATASET_REGISTRY,
    TASK_REGISTRY,
    LOSS_REGISTRY,
    EVALUATOR_REGISTRY,
    METRIC_ADAPTER_REGISTRY,
)

from sgg.framework.base_model import BaseModelAdapter
from sgg.framework.base_dataset import BaseSGGDataset
from sgg.framework.base_evaluator import BaseEvaluator
from sgg.framework.base_task import BaseTask
