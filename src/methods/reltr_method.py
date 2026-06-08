import torch
from .base_method import Base_method
from utils.misc import NestedTensor, nested_tensor_from_tensor_list
from src.models.reltr import build as build_reltr


class RelTR_Method(Base_method):
    """
    Lightning wrapper for RelTR.
    """

    def __init__(self, **args):
        super().__init__(**args)
        self.weight_dict = None

    def _build_model(self, **args):
        model, criterion, postprocessors = build_reltr(self.hparams)
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.weight_dict = dict(getattr(criterion, 'weight_dict', {}))
        return model

    # ---------- helpers (override base) ----------

    def _to_nested_tensor(self, images) -> NestedTensor:
        """Convert images to NestedTensor on the correct device.

        Handles both:
          - NestedTensor (from collate_fn) — just move to device
          - list[Tensor(C,H,W)] — convert via nested_tensor_from_tensor_list
        """
        if isinstance(images, NestedTensor):
            samples = images
        elif isinstance(images, (list, tuple)):
            # Validate: expect list of 3D tensors [C, H, W]
            fixed_images = []
            for i, img in enumerate(images):
                if not torch.is_tensor(img):
                    raise TypeError(f"images[{i}] is not Tensor, got {type(img)}")
                if img.dim() != 3:
                    raise ValueError(f"images[{i}] shape must be [C,H,W], got {img.shape}")
                fixed_images.append(img)
            samples = nested_tensor_from_tensor_list(fixed_images)
        else:
            raise TypeError(f"images should be NestedTensor or list of tensors, got {type(images)}")

        tensors = samples.tensors.to(self.device)
        mask = samples.mask.to(self.device) if samples.mask is not None else None
        return NestedTensor(tensors, mask)

    def _move_targets_to_device(self, targets):
        """Move target dicts to device, with validation."""
        moved_targets = []
        for i, t in enumerate(targets):
            if not isinstance(t, dict):
                raise TypeError(f"targets[{i}] is not dict, got {type(t)}: {t}")
            moved_targets.append(
                {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in t.items()}
            )
        return moved_targets

    def _compute_losses(self, outputs, targets):
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

    def forward(self, images, targets=None, return_predictions=False,
                target_sizes=None, **kwargs):
        samples = self._to_nested_tensor(images)
        outputs = self.model(samples)
        out = {'outputs': outputs}

        if targets is not None:
            targets = self._move_targets_to_device(targets)
            loss_dict, total_loss = self._compute_losses(outputs, targets)
            out['loss_dict'] = loss_dict
            out['loss'] = total_loss

        if return_predictions:
            if target_sizes is None:
                _, _, H, W = samples.tensors.shape
                target_sizes = torch.tensor(
                    [[H, W]] * samples.tensors.shape[0],
                    device=self.device
                )
            else:
                target_sizes = target_sizes.to(self.device)

            out['bbox_results'] = self.postprocessors['bbox'](outputs, target_sizes)

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
        """RelTR-specific eval: also extracts Hungarian matcher indices."""
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
