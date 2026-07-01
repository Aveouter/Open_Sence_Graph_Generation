"""
USG-Par method wrapper for PyTorch Lightning.

Integrates USG-Par (Wu et al., CVPR 2025) into the OpenSGG training framework.
USG-Par is a universal scene graph parser that uses learnable object queries,
a Relation Proposal Constructor (RPC), and a transformer-based relation decoder.

Output format is RelTR-shaped, but routed through the USG SGDet metric adapter
because official USG uses sigmoid BCE predicate scores with no background channel.
"""

import torch
from .base_method import Base_method
from utils.misc import NestedTensor, nested_tensor_from_tensor_list
from src.models.usg import build_usg


class USG_Method(Base_method):
    """USG-Par Lightning method.

    Training batch convention:
        batch = (images, targets)
          images: Tensor[B, 3, H, W] or list[Tensor(3, H, W)]
          targets: list[dict], each dict contains:
            - "labels": LongTensor[num_obj] — object class labels
            - "boxes":  FloatTensor[num_obj, 4] — (cx, cy, w, h) normalized
            - "rel_annotations": LongTensor[num_rel, 3] — (sub_idx, obj_idx, rel_label)

    Evaluation: SGDet mode (full end-to-end: boxes + labels + predicates).
    """

    def __init__(self, **args):
        super().__init__(**args)
        # Convert official warmup_steps → warmup_epoch for the
        # epoch-based scheduler.  Done here so the config only
        # needs warmup_steps, matching the official code exactly.
        warmup_steps = getattr(self.hparams, "warmup_steps", None)
        if warmup_steps is not None and warmup_steps > 0:
            steps_per_epoch = getattr(self.hparams, "steps_per_epoch", None)
            if steps_per_epoch is None or steps_per_epoch <= 0:
                import logging
                logging.warning(
                    "USG: warmup_steps=%d but steps_per_epoch=%s — "
                    "cannot convert to warmup_epoch. Skipping warmup.",
                    warmup_steps, steps_per_epoch,
                )
            else:
                # Result may be a fraction (e.g. 1000/14431 ≈ 0.07).
                # timm CosineLRScheduler accepts float warmup_t and
                # compares it with integer epochs — rounding is correct.
                self.hparams.warmup_epoch = warmup_steps / steps_per_epoch

    def _build_criterion(self, **args):
        """USG criterion is built inside _build_model(). Skip base class construction."""
        return None

    def _build_model(self, **args):
        """Build USG model + criterion."""
        model, criterion = build_usg(self.hparams)
        self.criterion = criterion
        return model

    # ---------- helpers ----------

    def _to_nested_tensor(self, images) -> NestedTensor:
        """Convert images to NestedTensor and move to device."""
        if isinstance(images, NestedTensor):
            samples = images
        else:
            samples = nested_tensor_from_tensor_list(images)
        tensors = samples.tensors.to(self.device)
        mask = samples.mask.to(self.device) if samples.mask is not None else None
        return NestedTensor(tensors, mask)

    def _compute_losses(self, outputs, targets):
        """Compute loss dict and total from model outputs.

        Uses the criterion's weighted 'loss_total' directly,
        since component losses ('loss_obj_cls', etc.) are unweighted.
        """
        loss_dict = self.criterion(outputs, targets)
        total_loss = loss_dict["loss_total"]
        return loss_dict, total_loss

    # ---------- forward ----------

    def forward(self, images, targets=None, **kwargs):
        """Forward pass for training or inference."""
        samples = self._to_nested_tensor(images)
        outputs = self.model(samples, targets=targets)
        out = {"outputs": outputs}
        if targets is not None:
            targets = self._move_targets_to_device(targets)
            loss_dict, total_loss = self._compute_losses(outputs, targets)
            out["loss_dict"] = loss_dict
            out["loss"] = total_loss
        return out

    # ---------- training_step ----------

    def training_step(self, batch, batch_idx):
        images, targets = batch
        out = self.forward(images, targets)
        total_loss = out["loss"]
        loss_dict = out.get("loss_dict", {})
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f"train_{k}", v, on_step=True, on_epoch=True, prog_bar=False)
        return total_loss

    # ---------- eval steps ----------

    @torch.no_grad()
    def _eval_step(self, batch, prefix: str):
        """USG eval: single forward pass provides both loss and eval outputs."""
        images, targets = self._split_batch(batch)
        samples = self._to_nested_tensor(images)
        targets = self._move_targets_to_device(targets)

        outputs = self.model(samples, targets=targets)
        loss_dict, total_loss = self._compute_losses(outputs, targets)

        # Strip internal keys not needed for evaluation/metrics caching
        outputs.pop("rpc_output", None)
        outputs.pop("queries", None)
        outputs.pop("class_logits", None)

        self.log(
            f"{prefix}_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(
                    f"{prefix}_{k}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        return outputs, targets, loss_dict, total_loss

    def validation_step(self, batch, batch_idx):
        outputs, targets, loss_dict, total_loss = self._eval_step(batch, "val")
        self._cache_step_output(
            self.val_outputs, outputs, targets, loss_dict, total_loss
        )
        return total_loss

    def test_step(self, batch, batch_idx):
        outputs, targets, loss_dict, total_loss = self._eval_step(batch, "test")
        self._cache_step_output(
            self.test_outputs, outputs, targets, loss_dict, total_loss
        )
        return total_loss
