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

        if batch_y is not None:
            loss_dict = self.criterion(outputs, batch_y)

            if isinstance(loss_dict, dict):
                outputs["loss_dict"] = loss_dict
                outputs["loss"] = loss_dict.get("loss_total", None)
                if outputs["loss"] is None:
                    outputs["loss"] = sum(
                        v for v in loss_dict.values() if torch.is_tensor(v)
                    )
            else:
                outputs["loss"] = loss_dict

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

        # 验证阶段显式计算 loss，而不是依赖 outputs 里自带 "loss"
        if batch_y is not None:
            loss_dict = self.criterion(outputs, batch_y)

            if isinstance(loss_dict, dict):
                loss = loss_dict.get("loss_total", None)
                if loss is None:
                    loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))
            else:
                loss = loss_dict

            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

            # 可选：把各子损失也记录出来，方便看训练过程
            if isinstance(loss_dict, dict):
                for k, v in loss_dict.items():
                    if torch.is_tensor(v):
                        self.log(f'val_{k}', v, on_step=False, on_epoch=True, prog_bar=False)

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