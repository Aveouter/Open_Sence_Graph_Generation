# sgg/framework/base_task.py
"""
Abstract base class for SGG tasks.

A Task is a PyTorch LightningModule that composes a model adapter,
a data module, and an evaluator to solve a specific SGG problem.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import lightning.pytorch as pl

from sgg.framework.base_model import BaseModelAdapter
from sgg.framework.base_evaluator import BaseEvaluator


class BaseTask(ABC, pl.LightningModule):
    """
    Abstract LightningModule for scene graph generation tasks.

    Holds:
      - model_adapter: BaseModelAdapter (model + criterion + postprocessors)
      - evaluator: BaseEvaluator (metric computation)
      - config: Task configuration

    Subclasses implement:
      - training_step / validation_step / test_step
      - _prepare_inputs() -> adapter-specific format
      - on_validation_epoch_end / on_test_epoch_end
    """

    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        evaluator: BaseEvaluator,
        config: Any,
    ):
        super().__init__()
        self.model_adapter = model_adapter
        self.evaluator = evaluator
        self.config = config
        self.save_hyperparameters(ignore=['model_adapter', 'evaluator'])

        # Per-epoch output caches
        self._val_outputs = []
        self._test_outputs = []

    @property
    def model(self):
        """Convenience: access the underlying nn.Module."""
        return self.model_adapter.model

    @property
    def criterion(self):
        """Convenience: access the loss criterion."""
        return self.model_adapter.criterion

    @property
    def postprocessors(self):
        """Convenience: access post-processing modules."""
        return self.model_adapter.postprocessors

    def configure_optimizers(self):
        """Default: delegate to training utilities."""
        from sgg.training.optim_scheduler import get_optim_scheduler
        return get_optim_scheduler(self.model, self.config)

    @abstractmethod
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Single training step. Must return a loss tensor."""

    @abstractmethod
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """Single validation step."""

    @abstractmethod
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """Single test step."""

    @abstractmethod
    def on_validation_epoch_end(self) -> Optional[Dict[str, float]]:
        """Aggregate validation outputs and compute metrics."""

    @abstractmethod
    def on_test_epoch_end(self) -> Optional[Dict[str, float]]:
        """Aggregate test outputs and compute metrics."""
