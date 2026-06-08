# src/methods/flowsg_method.py
"""
FlowSG method wrapper for PyTorch Lightning.

Integrates FlowSG (CVPR 2026) into the OpenSGG training framework.
FlowSG is a flow-matching-based scene graph generator that progressively
refines bounding boxes and classifies objects/relations in a joint graph
transformer.

Output format is compatible with RelTR evaluation pipeline (SGDet/SGCLS/PredCLS).
"""

import torch
from .base_method import Base_method
from utils.misc import NestedTensor, nested_tensor_from_tensor_list
from src.models.flowsg import build_flowsg


class FlowSG_Method(Base_method):
    """FlowSG Lightning method.

    Training batch convention:
        batch = (images, targets)
          images: Tensor[B, 3, H, W] or list[Tensor(3, H, W)]
          targets: list[dict], each dict contains:
            - "labels": LongTensor[num_obj] — object class labels
            - "boxes":  FloatTensor[num_obj, 4] — (cx, cy, w, h) normalized
            - "rel_annotations": LongTensor[num_rel, 3] — (sub_idx, obj_idx, rel_label)

    Evaluation: SGDet mode (full end-to-end: boxes + labels + predicates)
    """

    def __init__(self, **args):
        super().__init__(**args)

    def _build_criterion(self, **args):
        """FlowSG criterion is built inside _build_model(). Skip base class construction."""
        return None

    def _build_model(self, **args):
        """Build FlowSG model + criterion."""
        model, criterion = build_flowsg(self.hparams)
        self.criterion = criterion
        return model

    # ---------- helpers (override base) ----------

    def _to_nested_tensor(self, images) -> NestedTensor:
        """Convert images to NestedTensor and move to device."""
        if isinstance(images, NestedTensor):
            samples = images
        else:
            samples = nested_tensor_from_tensor_list(images)

        tensors = samples.tensors.to(self.device)
        mask = samples.mask.to(self.device) if samples.mask is not None else None
        return NestedTensor(tensors, mask)

    def _move_targets_to_device(self, targets):
        """Move target tensors to device."""
        moved = []
        for t in targets:
            t2 = {}
            for k, v in t.items():
                t2[k] = v.to(self.device) if torch.is_tensor(v) else v
            moved.append(t2)
        return moved

    def _compute_losses(self, outputs, targets):
        """Compute loss dict and total from model outputs."""
        loss_dict = self.criterion(outputs, targets)

        total_loss = 0.0
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                total_loss = total_loss + v

        return loss_dict, total_loss

    # ---------- forward / predict ----------

    def forward(self, images, targets=None, **kwargs):
        """Forward pass for training or inference.

        Args:
            images: Tensor or NestedTensor of batched images
            targets: list[dict] or None

        Returns:
            dict with keys: outputs, loss_dict (if targets), loss (if targets)
        """
        samples = self._to_nested_tensor(images)
        outputs = self.model(samples, targets=targets)

        out = {"outputs": outputs}

        if targets is not None:
            targets = self._move_targets_to_device(targets)
            loss_dict = self.criterion(outputs, targets)

            total_loss = 0.0
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    total_loss = total_loss + v

            out["loss_dict"] = loss_dict
            out["loss"] = total_loss

        return out

    # ---------- training_step ----------

    def training_step(self, batch, batch_idx):
        images, targets = batch
        samples = self._to_nested_tensor(images)
        targets = self._move_targets_to_device(targets)

        outputs = self.model(samples, targets=targets)
        loss_dict = self.criterion(outputs, targets)

        total_loss = 0.0
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                total_loss = total_loss + v

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'train_{k}', v, on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    # ---------- eval steps (single forward pass) ----------

    @torch.no_grad()
    def _eval_step(self, batch, prefix: str):
        """FlowSG eval: single forward pass provides both loss and eval outputs.

        The training-mode forward already returns eval-compatible keys
        (sub_boxes, obj_boxes, sub_logits, obj_logits, rel_logits), so we
        don't need a second inference pass.
        """
        images, targets = self._split_batch(batch)
        samples = self._to_nested_tensor(images)
        targets = self._move_targets_to_device(targets)

        # Single forward pass — training mode returns eval keys too
        outputs = self.model(samples, targets=targets)
        loss_dict, total_loss = self._compute_losses(outputs, targets)

        self.log(f'{prefix}_loss', total_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'{prefix}_{k}', v, on_step=False, on_epoch=True,
                         prog_bar=False, sync_dist=True)

        return outputs, targets, loss_dict, total_loss

    def validation_step(self, batch, batch_idx):
        outputs, targets, loss_dict, total_loss = self._eval_step(batch, 'val')
        self._cache_step_output(self.val_outputs, outputs, targets, loss_dict, total_loss)
        return total_loss

    def test_step(self, batch, batch_idx):
        outputs, targets, loss_dict, total_loss = self._eval_step(batch, 'test')
        self._cache_step_output(self.test_outputs, outputs, targets, loss_dict, total_loss)
        return total_loss
