# src/methods/usg_method.py
"""
USG method wrapper for PyTorch Lightning.

Integrates USG-Par (CVPR 2025) into the OpenSGG training framework.
Output format is compatible with RelTR evaluation pipeline (SGDet/SGCLS/PredCLS).
"""

import torch
from .base_method import Base_method
from utils.misc import NestedTensor, nested_tensor_from_tensor_list
from src.models.usg import build_usg


class USG_Method(Base_method):
    """USG Lightning method.

    Training batch convention:
        batch = (images, targets)
          images: NestedTensor or list[Tensor(3, H, W)]
          targets: list[dict], each dict contains:
            - "labels": LongTensor[num_obj] — object class labels
            - "boxes":  FloatTensor[num_obj, 4] — (cx, cy, w, h) normalized
            - "rel_annotations": LongTensor[num_rel, 3] — (sub_idx, obj_idx, rel_label)

    Evaluation: SGDet mode by default (box detection + predicate classification).
    """

    def __init__(self, **args):
        super().__init__(**args)
        self.weight_dict = None

    def _build_criterion(self, **args):
        """USG criterion is built inside _build_model(). Skip base class construction."""
        return None

    def _build_model(self, **args):
        """Build USG model + criterion."""
        model, criterion = build_usg(self.hparams)
        self.criterion = criterion
        self.weight_dict = dict(getattr(criterion, 'weight_dict', {}))
        return model

    # ---------- helpers (override base) ----------

    def _to_nested_tensor(self, images) -> NestedTensor:
        """Convert images to NestedTensor and move to device."""
        if isinstance(images, NestedTensor):
            samples = images
        elif isinstance(images, (list, tuple)):
            samples = nested_tensor_from_tensor_list(images)
        else:
            raise TypeError(f"images should be NestedTensor or list of tensors, got {type(images)}")

        tensors = samples.tensors.to(self.device)
        mask = samples.mask.to(self.device) if samples.mask is not None else None
        return NestedTensor(tensors, mask)

    def _move_targets_to_device(self, targets):
        """Move target dicts to device."""
        return [
            {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in targets
        ]

    def _compute_losses(self, outputs, targets):
        """Compute loss dict and total from model outputs via criterion."""
        loss_dict = self.criterion(outputs, targets)

        total_loss = 0.0
        if not self.weight_dict:
            for v in loss_dict.values():
                if torch.is_tensor(v):
                    total_loss = total_loss + v
            return loss_dict, total_loss

        for k, v in loss_dict.items():
            if k in self.weight_dict:
                total_loss = total_loss + v * self.weight_dict[k]

        return loss_dict, total_loss

    # ---------- forward / predict ----------

    def forward(self, images, targets=None, **kwargs):
        """Forward pass for USG."""
        samples = self._to_nested_tensor(images)
        outputs = self.model(samples, targets=targets)

        out = {'outputs': outputs}

        if targets is not None:
            targets = self._move_targets_to_device(targets)
            loss_dict, total_loss = self._compute_losses(outputs, targets)
            out['loss_dict'] = loss_dict
            out['loss'] = total_loss

        return out

    # ---------- training / eval steps ----------

    def training_step(self, batch, batch_idx):
        images, targets = self._split_batch(batch)

        samples = self._to_nested_tensor(images)
        targets = self._move_targets_to_device(targets)

        outputs = self.model(samples)
        loss_dict, total_loss = self._compute_losses(outputs, targets)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'train_{k}', v, on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    @torch.no_grad()
    def _eval_step(self, batch, prefix: str):
        """USG eval: forward pass + Hungarian matching for score boosting."""
        images, targets = self._split_batch(batch)

        samples = self._to_nested_tensor(images)
        targets = self._move_targets_to_device(targets)

        outputs = self.model(samples)
        loss_dict, total_loss = self._compute_losses(outputs, targets)

        self.log(f'{prefix}_loss', total_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'{prefix}_{k}', v, on_step=False, on_epoch=True,
                         prog_bar=False, sync_dist=True)

        # Extract Hungarian matching indices for score boosting / PredCLS matching
        triplet_indices = None
        if hasattr(self.criterion, 'indices') and self.criterion.indices is not None:
            triplet_indices = [
                (src.cpu().clone(), tgt.cpu().clone())
                for src, tgt in self.criterion.indices[1]
            ]

        return outputs, targets, loss_dict, total_loss, triplet_indices

    def validation_step(self, batch, batch_idx):
        outputs, targets, loss_dict, total_loss, triplet_indices = \
            self._eval_step(batch, 'val')
        self._cache_step_output(
            self.val_outputs, outputs, targets, loss_dict, total_loss,
            triplet_indices=triplet_indices,
        )
        return total_loss

    def test_step(self, batch, batch_idx):
        outputs, targets, loss_dict, total_loss, triplet_indices = \
            self._eval_step(batch, 'test')
        self._cache_step_output(
            self.test_outputs, outputs, targets, loss_dict, total_loss,
            triplet_indices=triplet_indices,
        )
        return total_loss
