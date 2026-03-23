import torch
import torch.nn as nn

from .base_method import Base_method
from src.models.HSTRNet import HSTRNetModel


class HSTRNet_Method(Base_method):
    r"""HSTRNet

    HSTRNet method wrapper for the OpenSTL-style training framework.
    Responsible for:
    1. building the model
    2. defining the forward interface
    3. adapting batch format to the training framework
    """

    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **args):
        return HSTRNetModel(self.hparams)

    def forward(self, batch_x, batch_y=None, **kwargs):
        """
        Args:
            batch_x:
                - if tensor: expected shape [B, T, C, H, W]
                - if dict: should contain key 'images', optionally 'targets'
            batch_y:
                optional labels / targets
            **kwargs:
                extra arguments reserved for compatibility

        Returns:
            model outputs
        """
        targets = None

        if isinstance(batch_x, dict):
            images = batch_x["images"]
            targets = batch_x.get("targets", batch_y)
        else:
            images = batch_x
            targets = batch_y

        outputs = self.model(images, targets=targets)
        return outputs

    def training_step(self, batch, batch_idx):
        """
        Compatible with common batch formats:
        1. batch = (batch_x, batch_y)
        2. batch = {'images': ..., 'targets': ...}
        """
        if isinstance(batch, dict):
            batch_x = batch["images"]
            batch_y = batch.get("targets", None)
        else:
            batch_x, batch_y = batch

        outputs = self.forward(batch_x, batch_y)

        if isinstance(outputs, dict):
            loss = outputs["loss"]
        elif torch.is_tensor(outputs):
            loss = outputs
        else:
            raise TypeError(
                f"Unsupported model output type: {type(outputs)}. "
                f"Expected dict with key 'loss' or a tensor loss."
            )

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            batch_x = batch["images"]
            batch_y = batch.get("targets", None)
        else:
            batch_x, batch_y = batch

        outputs = self.forward(batch_x, batch_y)

        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        return outputs

    def test_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            batch_x = batch["images"]
            batch_y = batch.get("targets", None)
        else:
            batch_x, batch_y = batch

        outputs = self.forward(batch_x, batch_y)
        return outputs