# ssgoos/tasks/image_ssgoos.py
"""
Image-based Scene Graph Generation Task.

A PyTorch LightningModule that composes a model adapter, evaluator,
and data module for solving image SGG (sgdet / predcls / sgcls).

Registered as 'image_sgg' in TASK_REGISTRY.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ssgoos.registry import TASK_REGISTRY
from ssgoos.framework.base_task import BaseTask
from ssgoos.framework.base_model import BaseModelAdapter
from ssgoos.framework.base_evaluator import BaseEvaluator
from ssgoos.evaluation.orchestrator import (
    gather_epoch_outputs_to_rank0,
    merge_pred_outputs,
    merge_targets,
    average_losses,
)


@TASK_REGISTRY.register('image_sgg')
class ImageSceneGraphTask(BaseTask):
    """
    LightningModule for image-level scene graph generation.

    Supports three evaluation modes:
      - sgdet: Full scene graph detection (boxes + labels + predicates)
      - predcls: Predicate classification with GT boxes + labels
      - sgcls: Scene graph classification with GT boxes

    The task is model-agnostic: it works with any model adapter
    that follows the BaseModelAdapter interface.
    """

    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        evaluator: BaseEvaluator,
        config: Any,
    ):
        super().__init__(model_adapter, evaluator, config)

        # Task configuration
        self.metric_names = getattr(config, 'metrics', [])
        self.rel_nums = getattr(config, 'rel_nums', 51)
        self.entity_nums = getattr(config, 'entity_nums', 151)
        self.save_dir = getattr(config, 'save_dir', './results/Debug')

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Standard training step: forward pass + loss computation."""
        images, targets = self._unpack_batch(batch)
        result = self.model_adapter(images, targets)
        loss = result['total_loss']

        # Log per-component losses
        loss_dict = result.get('loss_dict', {})
        for name, value in loss_dict.items():
            self.log(f'train_{name}', value, on_step=True, on_epoch=False)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx) -> Dict[str, Any]:
        """Validation step: forward pass + output caching."""
        images, targets = self._unpack_batch(batch)
        result = self.model_adapter(images, targets)

        # Cache outputs for epoch-end evaluation
        self._val_outputs.append({
            'outputs': self._detach_outputs(result['outputs']),
            'targets': targets,
            'loss_dict': {
                k: v.item() if torch.is_tensor(v) else v
                for k, v in result.get('loss_dict', {}).items()
            },
            'triplet_indices': self._extract_triplet_indices(result),
        })

        # Log val loss
        loss = result.get('total_loss', torch.tensor(0.0))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> Optional[Dict[str, float]]:
        """Aggregate validation outputs and compute metrics."""
        return self._run_epoch_eval('val')

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        """Test step: forward pass + output caching."""
        images, targets = self._unpack_batch(batch)

        # Inference only (no loss for speed)
        with torch.no_grad():
            result = self.model_adapter(images, targets)

        self._test_outputs.append({
            'outputs': self._detach_outputs(result['outputs']),
            'targets': targets,
            'loss_dict': {
                k: v.item() if torch.is_tensor(v) else v
                for k, v in result.get('loss_dict', {}).items()
            },
            'triplet_indices': self._extract_triplet_indices(result),
        })

        loss = result.get('total_loss', torch.tensor(0.0))
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self) -> Optional[Dict[str, float]]:
        """Aggregate test outputs and compute metrics."""
        return self._run_epoch_eval('test')

    # ------------------------------------------------------------------
    # Evaluation Pipeline
    # ------------------------------------------------------------------

    def _run_epoch_eval(self, stage: str) -> Dict[str, float]:
        """
        Core evaluation pipeline: gather, merge, compute metrics.

        Args:
            stage: 'val' or 'test'.

        Returns:
            Dict of metric name -> value.
        """
        outputs = self._val_outputs if stage == 'val' else self._test_outputs

        # DDP-safe gather to rank 0
        merged = gather_epoch_outputs_to_rank0(
            outputs, self.save_dir, stage, self.current_epoch
        )

        if merged is None:
            # Not rank 0 in DDP mode
            return {}

        # Merge predictions and targets
        preds = merge_pred_outputs([m['outputs'] for m in merged])
        tgts = merge_targets([m['targets'] for m in merged])
        triplet_indices = self._merge_triplet_indices(merged)
        avg_loss = average_losses([m['loss_dict'] for m in merged])

        # Compute metrics
        from ssgoos.evaluation.metric_adapters import metric as compute_metric
        metric_results = compute_metric(
            preds, tgts, self.metric_names,
            rel_nums=self.rel_nums,
            entity_nums=self.entity_nums,
            triplet_indices=triplet_indices,
        )

        # Log
        for k, v in avg_loss.items():
            self.log(f'{stage}_avg_{k}', v, rank_zero_only=True)

        for k, v in metric_results.items():
            self.log(f'{stage}_{k}', v, rank_zero_only=True)

        # Clear caches
        if stage == 'val':
            self._val_outputs = []
        else:
            self._test_outputs = []

        return {**avg_loss, **metric_results}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch):
        """Unpack batch into (images, targets)."""
        if isinstance(batch, (list, tuple)):
            return batch[0], batch[1]
        elif isinstance(batch, dict):
            return batch.get('images', batch.get('samples')), batch.get('targets')
        return batch, None

    def _detach_outputs(self, outputs):
        """Detach tensors in output dict for CPU caching."""
        if outputs is None:
            return {}
        detached = {}
        for k, v in outputs.items():
            if torch.is_tensor(v):
                detached[k] = v.detach().cpu()
            elif isinstance(v, list) and all(torch.is_tensor(x) for x in v):
                detached[k] = [x.detach().cpu() for x in v]
            else:
                detached[k] = v
        return detached

    def _extract_triplet_indices(self, result):
        """
        Extract Hungarian matching indices for predcls/sgcls evaluation.
        RelTR stores these in criterion.indices after forward.
        """
        criterion = self.model_adapter.criterion
        if criterion is not None and hasattr(criterion, 'indices'):
            return criterion.indices
        return None

    def _merge_triplet_indices(self, merged_outputs):
        """Merge triplet matching indices across batches."""
        indices_list = []
        for m in merged_outputs:
            if m.get('triplet_indices') is not None:
                indices_list.append(m['triplet_indices'])
        return indices_list if indices_list else None
