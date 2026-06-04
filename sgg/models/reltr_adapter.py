# sgg/models/reltr_adapter.py
"""
RelTR Model Adapter.

Wraps the RelTR nn.Module, SetCriterion, and PostProcess into a unified
BaseModelAdapter interface. Registered as 'reltr' in MODEL_REGISTRY.
"""

from typing import Any, Dict, Optional

import torch.nn as nn

from sgg.registry import MODEL_REGISTRY
from sgg.framework.base_model import BaseModelAdapter
from sgg.modules.reltr import build as build_reltr


@MODEL_REGISTRY.register('reltr')
class RelTRAdapter(BaseModelAdapter):
    """
    Adapter for the RelTR (Relation Transformer) model.

    Wraps:
      - RelTR nn.Module (backbone + transformer + prediction heads)
      - SetCriterion (Hungarian-matching-based loss)
      - PostProcess (bbox conversion to COCO format)

    Output schema:
      Model family: 'reltr'
      Output keys: pred_logits, pred_boxes, sub_logits, sub_boxes,
                   obj_logits, obj_boxes, rel_logits
      Supported tasks: sgdet, predcls, sgcls
      Has triplet matching indices: True
    """

    def __init__(self):
        super().__init__()
        self._model: Optional[nn.Module] = None
        self._criterion: Optional[nn.Module] = None
        self._postprocessors: Dict[str, nn.Module] = {}
        self._weight_dict: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # BaseModelAdapter interface
    # ------------------------------------------------------------------

    def build(self, config) -> None:
        """
        Build RelTR model, criterion, and postprocessors from config.

        Delegates to the existing build() factory which returns
        (model, criterion, postprocessors).
        """
        self._model, self._criterion, self._postprocessors = build_reltr(config)
        # Store loss weights for weighted loss aggregation
        self._weight_dict = getattr(self._criterion, 'weight_dict', {})

    def forward(
        self, inputs: Any, targets: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Forward pass through RelTR.

        Args:
            inputs: NestedTensor (images + masks) or list of Tensors.
            targets: Optional list of target dicts for loss computation.

        Returns:
            {'outputs': ..., 'loss_dict': ..., 'total_loss': ...}
            When targets is None, loss_dict and total_loss are omitted.
        """
        if self._model is None:
            raise RuntimeError("Model not built. Call build() first.")

        # Convert inputs to NestedTensor if needed
        from sgg.utils.misc import NestedTensor, nested_tensor_from_tensor_list

        if isinstance(inputs, (list, tuple)):
            samples = nested_tensor_from_tensor_list(inputs)
        elif isinstance(inputs, NestedTensor):
            samples = inputs
        else:
            # Assume it's a tensor batch
            samples = nested_tensor_from_tensor_list(inputs)

        outputs = self._model(samples)

        result = {'outputs': outputs}

        if targets is not None and self._criterion is not None:
            # Move targets to device
            targets = self._move_targets_to_device(targets)
            loss_dict = self._criterion(outputs, targets)
            total_loss = sum(
                loss_dict[k] * self._weight_dict.get(k, 1.0)
                for k in loss_dict
            )
            result['loss_dict'] = loss_dict
            result['total_loss'] = total_loss

        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Model not built. Call build() first.")
        return self._model

    @property
    def criterion(self) -> Optional[nn.Module]:
        return self._criterion

    @property
    def postprocessors(self) -> Dict[str, nn.Module]:
        return self._postprocessors

    @property
    def weight_dict(self) -> Dict[str, float]:
        return self._weight_dict

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            'model_family': 'reltr',
            'output_keys': [
                'pred_logits', 'pred_boxes',
                'sub_logits', 'sub_boxes',
                'obj_logits', 'obj_boxes',
                'rel_logits',
            ],
            'supported_tasks': ['sgdet', 'predcls', 'sgcls'],
            'has_triplet_indices': True,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _move_targets_to_device(self, targets):
        """Recursively move all tensors in target dicts to model device."""
        device = next(self._model.parameters()).device
        moved = []
        for t in targets:
            if isinstance(t, dict):
                moved.append({
                    k: v.to(device) if hasattr(v, 'to') else v
                    for k, v in t.items()
                })
            else:
                moved.append(t)
        return moved
