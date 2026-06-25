import torch
from .base_method import Base_method
from src.models.ci_adversarial import build_ci_adversarial


class CiAdversarial_Method(Base_method):
    """Lightning wrapper for the CI Adversarial transformer SGG model."""

    def __init__(self, **args):
        super().__init__(**args)
        self.weight_dict = None

    def _build_model(self, **args):
        model, criterion, postprocessors = build_ci_adversarial(self.hparams)
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.weight_dict = dict(getattr(criterion, "weight_dict", {}))
        return model

    def _compute_losses(self, outputs, targets):
        loss_dict = self.criterion(outputs, targets)
        total_loss = loss_dict.pop("loss", 0.0)
        return loss_dict, total_loss

    # ------------------------------------------------------------------
    # Forward / training / eval
    # ------------------------------------------------------------------

    def forward(self, images, targets=None, return_predictions=False, **kwargs):
        samples = self._to_nested_tensor(images)
        outputs = self.model(samples)
        out = {"outputs": outputs}

        if targets is not None:
            targets = self._move_targets_to_device(targets)
            loss_dict, total_loss = self._compute_losses(outputs, targets)
            out["loss_dict"] = loss_dict
            out["loss"] = total_loss

        return out

    def training_step(self, batch, batch_idx):
        images, targets = self._split_batch(batch)
        samples = self._to_nested_tensor(images)
        targets = self._move_targets_to_device(targets)

        outputs = self.model(samples)
        loss_dict, total_loss = self._compute_losses(outputs, targets)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f"train_{k}", v, on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    @torch.no_grad()
    def _eval_step(self, batch, prefix: str):
        images, targets = self._split_batch(batch)
        samples = self._to_nested_tensor(images)
        targets = self._move_targets_to_device(targets)

        outputs = self.model(samples)
        loss_dict, total_loss = self._compute_losses(outputs, targets)

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
