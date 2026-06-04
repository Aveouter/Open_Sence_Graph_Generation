# ssgoos/models/hstrnet_adapter.py
"""
HSTRNet Model Adapter.

Wraps the HSTRNetModel into a unified BaseModelAdapter interface.
Registered as 'hstrnet' in MODEL_REGISTRY.
"""

from typing import Any, Dict, Optional

import torch.nn as nn

from ssgoos.registry import MODEL_REGISTRY
from ssgoos.framework.base_model import BaseModelAdapter


@MODEL_REGISTRY.register('hstrnet')
class HSTRNetAdapter(BaseModelAdapter):
    """
    Adapter for the HSTRNet (Hierarchical Spatio-Temporal Relation Network) model.

    Handles flexible input formats:
      - NestedTensor: tensors=[B, C, H, W]
      - Tensor: [B, C, H, W] or [B, T, C, H, W]
      - Dict with 'images' key
      - Tuple of (images, targets)
    """

    def __init__(self):
        super().__init__()
        self._model: Optional[nn.Module] = None
        self._criterion: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # BaseModelAdapter interface
    # ------------------------------------------------------------------

    def build(self, config) -> None:
        """Build HSTRNet model and criterion from config."""
        from ssgoos.modules.hstrnet import HSTRNetModel
        self._model = HSTRNetModel(config)
        self._criterion = self._build_criterion(config)

    def forward(
        self, inputs: Any, targets: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Forward pass through HSTRNet.

        Args:
            inputs: NestedTensor, Tensor [B,C,H,W] or [B,T,C,H,W],
                    or dict with 'images' key.
            targets: Optional targets for loss computation.

        Returns:
            {'outputs': ..., 'loss_dict': ..., 'total_loss': ...}
        """
        if self._model is None:
            raise RuntimeError("Model not built. Call build() first.")

        outputs = self._model(inputs, targets=targets)

        result = {'outputs': outputs}

        if targets is not None and self._criterion is not None:
            loss_dict = self._criterion(outputs, targets)
            total_loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
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
        return {}

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            'model_family': 'hstrnet',
            'output_keys': [
                'object_labels', 'object_logits',
                'relation_scores', 'pair_indices',
            ],
            'supported_tasks': ['sgdet'],
            'has_triplet_indices': False,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_criterion(self, config):
        """Build HSTRNet loss criterion."""
        from ssgoos.losses.loss import loss_construction
        loss_name = getattr(config, 'loss', 'hstrnet_loss')
        return loss_construction(loss_name, config)
