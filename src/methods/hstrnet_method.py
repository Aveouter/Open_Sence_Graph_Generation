import torch
import torch.nn as nn

from .base_method import Base_method
from src.models.HSTRNet import HSTRNetModel


class HSTRNet_Method(Base_method):
    r"""HSTRNet

    HSTRNet method wrapper for the OpenSGG training framework.
    Responsible for:
    1. building the model
    2. defining the forward interface
    3. adapting batch format to the training framework
    4. caching outputs for evaluation (PredCLS mode by default)

    HSTRNet uses query-based object encoding with hierarchical prototype
    relation learning. It supports PredCLS evaluation (given GT boxes + labels,
    predict predicates) and can be extended to SGCLS with box predictions.
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
            model outputs dict
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
        """
        Validation step with proper output caching for epoch-end evaluation.

        Caches model outputs + targets to self.val_outputs so that
        on_validation_epoch_end can run the full evaluation pipeline.
        """
        if isinstance(batch, dict):
            batch_x = batch["images"]
            batch_y = batch.get("targets", None)
        else:
            batch_x, batch_y = batch

        outputs = self.forward(batch_x, batch_y)

        # Compute loss
        if batch_y is not None:
            loss_dict = self.criterion(outputs, batch_y)

            if isinstance(loss_dict, dict):
                loss = loss_dict.get("loss_total", None)
                if loss is None:
                    loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))
            else:
                loss = loss_dict

            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            # Log individual loss terms
            if isinstance(loss_dict, dict):
                for k, v in loss_dict.items():
                    if torch.is_tensor(v):
                        self.log(f'val_{k}', v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        else:
            loss = torch.tensor(0.0, device=self.device)
            loss_dict = {}

        # Cache outputs for epoch-end evaluation
        # HSTRNet returns: object_logits, final_predicate_logits, relation_pair_indices, etc.
        self.val_outputs.append({
            'outputs': {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in outputs.items()
            },
            'targets': self._serialize_targets(batch_y) if batch_y is not None else [],
            'loss_dict': {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in (loss_dict.items() if isinstance(loss_dict, dict) else {})
            },
            'total_loss': loss.detach().cpu() if torch.is_tensor(loss) else loss,
        })

        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step with proper output caching for epoch-end evaluation.
        """
        if isinstance(batch, dict):
            batch_x = batch["images"]
            batch_y = batch.get("targets", None)
        else:
            batch_x, batch_y = batch

        outputs = self.forward(batch_x, batch_y)

        # Compute loss
        if batch_y is not None:
            loss_dict = self.criterion(outputs, batch_y)

            if isinstance(loss_dict, dict):
                loss = loss_dict.get("loss_total", None)
                if loss is None:
                    loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))
            else:
                loss = loss_dict

            self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            if isinstance(loss_dict, dict):
                for k, v in loss_dict.items():
                    if torch.is_tensor(v):
                        self.log(f'test_{k}', v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        else:
            loss = torch.tensor(0.0, device=self.device)
            loss_dict = {}

        # Cache outputs for epoch-end evaluation
        self.test_outputs.append({
            'outputs': {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in outputs.items()
            },
            'targets': self._serialize_targets(batch_y) if batch_y is not None else [],
            'loss_dict': {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in (loss_dict.items() if isinstance(loss_dict, dict) else {})
            },
            'total_loss': loss.detach().cpu() if torch.is_tensor(loss) else loss,
        })

        return loss

    def _serialize_targets(self, targets):
        """Serialize targets to CPU for caching across DDP ranks.

        Handles both list[dict] and dict batch formats.
        """
        if isinstance(targets, dict):
            # Batched dict format: convert to list of dicts
            # Determine batch size from first tensor value
            batch_size = None
            for v in targets.values():
                if torch.is_tensor(v):
                    batch_size = v.shape[0]
                    break
            if batch_size is None:
                return [targets]

            result = []
            for i in range(batch_size):
                entry = {}
                for k, v in targets.items():
                    if torch.is_tensor(v):
                        entry[k] = v[i].detach().cpu()
                    elif isinstance(v, list):
                        entry[k] = v[i]
                    else:
                        entry[k] = v
                result.append(entry)
            return result
        elif isinstance(targets, (list, tuple)):
            return [
                {
                    kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv)
                    for kk, vv in t.items()
                }
                for t in targets
            ]
        return targets
