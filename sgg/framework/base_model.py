# sgg/framework/base_model.py
"""
Abstract base class for model adapters.

A ModelAdapter wraps a raw nn.Module together with its loss criterion
and post-processors, providing a unified interface for the task layer.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch.nn as nn


class BaseModelAdapter(ABC):
    """
    Wraps a raw nn.Module + Criterion + PostProcessors into a unified interface.

    Responsibilities:
      - Build the model, criterion, and postprocessors from config
      - Provide a unified forward() interface that hides model-specific I/O formats
      - Expose underlying components for task-layer access
      - Declare output schema for evaluator adapter selection

    Subclasses must implement all abstract methods.
    """

    @abstractmethod
    def build(self, config) -> None:
        """
        Build the model, criterion, and postprocessors from configuration.

        Args:
            config: Model configuration (argparse.Namespace or dataclass).
        """

    @abstractmethod
    def forward(
        self, inputs: Any, targets: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Unified forward pass.

        Args:
            inputs: Model inputs (format depends on the model).
            targets: Optional ground-truth targets for loss computation.

        Returns:
            Dict with keys:
              - 'outputs': raw model predictions (tensor dict)
              - 'loss_dict': per-component loss scalars (if targets provided)
              - 'total_loss': aggregated loss (if targets provided)
        """

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        """Return the underlying nn.Module."""

    @property
    @abstractmethod
    def criterion(self) -> Optional[nn.Module]:
        """Return the loss criterion (may be None for inference-only models)."""

    @property
    @abstractmethod
    def postprocessors(self) -> Dict[str, nn.Module]:
        """Return post-processing modules keyed by type (e.g., 'bbox')."""

    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        """
        Return model output format description for evaluator adapter selection.

        Example:
            {
                'model_family': 'reltr',
                'output_keys': ['sub_boxes', 'obj_boxes', 'sub_logits',
                                'obj_logits', 'rel_logits', 'pred_boxes',
                                'pred_logits'],
                'supported_tasks': ['sgdet', 'predcls', 'sgcls'],
                'has_triplet_indices': True,
            }
        """
