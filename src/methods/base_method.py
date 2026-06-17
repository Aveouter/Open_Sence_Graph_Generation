import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import lightning as l

from src.core import get_optim_scheduler, timm_schedulers
from src.core import metric
from src.loss import *


class Base_method(l.LightningModule):

    def __init__(self, **args):
        super().__init__()

        self.save_hyperparameters()

        self.criterion = self._build_criterion(**args)
        self.model = self._build_model(**args)
        self.metric = args['metrics']
        self.rel_nums = args.get('rel_nums', None)
        self.entity_nums = args.get('entity_nums', None)

        self.test_outputs = []
        self.val_outputs = []

    def _build_criterion(self, **args):
        loss_name = args['loss'] if 'loss' in args else None
        return loss_construction(loss_name) if loss_name is not None else None

    def _build_model(self, **args):
        raise NotImplementedError

    def _compute_losses(self, outputs, targets):
        raise NotImplementedError(
            "Subclasses must implement _compute_losses, or override "
            "validation_step / test_step to not call _eval_step."
        )

    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams,
            self.hparams.epoch,
            self.model,
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" if by_epoch else "step"
            },
        }

    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def _to_nested_tensor(self, images):
        """Convert images to a format the model accepts.

        Base implementation: pass-through.  Subclasses that need NestedTensor
        conversion should override (see RelTR_Method, FlowSG_Method).
        """
        return images

    def _move_targets_to_device(self, targets):
        """Move a list/tuple of target dicts to the current device."""
        return [
            {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in targets
        ]

    def _split_batch(self, batch):
        """Split a batch into (images, targets).

        Handles the format produced by ``utils.collate_fn``:
            (NestedTensor, tuple_of_targets)

        Also accepts:
            [img_tensor, target_dict, ...]  — list of per-sample pairs
            {'images': ..., 'targets': ...} — dict format (HSTRNet)
        """
        if isinstance(batch, dict):
            return batch['images'], batch.get('targets', None)

        if isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                raise ValueError("empty batch")
            first = batch[0]
            # Format A: collate_fn output — (NestedTensor, tuple_of_targets)
            #   first is a NestedTensor (has .tensors attr)
            if hasattr(first, 'tensors'):
                return batch[0], batch[1]
            # Format B: list of (image_tensor, target_dict) pairs
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                images = [b[0] for b in batch]
                targets = [b[1] for b in batch]
                return images, targets

        raise TypeError(
            f"Unsupported batch format: {type(batch)}. "
            f"Expected (NestedTensor, targets) tuple or list of (img, target) pairs."
        )

    def _cache_step_output(self, outputs_store, outputs, targets, loss_dict, total_loss,
                           triplet_indices=None):
        """Cache one step's outputs for later aggregation.

        Parameters
        ----------
        triplet_indices : list of (src_idx, tgt_idx) tuples, optional
            Hungarian matcher indices for RelTR score boosting / PredCLS matching.
        """
        entry = {
            'outputs': {
                k: (
                    [t.detach().cpu() if torch.is_tensor(t) else t for t in v]
                    if isinstance(v, list)
                    else (v.detach().cpu() if torch.is_tensor(v) else v)
                )
                for k, v in outputs.items()
                if k != 'aux_outputs'  # skip auxiliary decoder outputs to save memory
            },
            'targets': [
                {
                    kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv)
                    for kk, vv in t.items()
                }
                for t in targets
            ],
            'loss_dict': {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in loss_dict.items()
            },
            'total_loss': total_loss.detach().cpu() if torch.is_tensor(total_loss) else total_loss,
        }
        if triplet_indices is not None:
            entry['triplet_indices'] = triplet_indices
        outputs_store.append(entry)

    def _eval_step(self, batch, prefix: str):
        """Shared eval logic for validation_step and test_step."""
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

        return outputs, targets, loss_dict, total_loss

    def validation_step(self, batch, batch_idx):
        outputs, targets, loss_dict, total_loss = self._eval_step(batch, 'val')
        self._cache_step_output(self.val_outputs, outputs, targets, loss_dict, total_loss)
        return total_loss

    def test_step(self, batch, batch_idx):
        outputs, targets, loss_dict, total_loss = self._eval_step(batch, 'test')
        self._cache_step_output(self.test_outputs, outputs, targets, loss_dict, total_loss)
        return total_loss

    # ------------------------------------------------------------------
    # DDP gathering helpers
    # ------------------------------------------------------------------

    def _gather_step_outputs(self, step_outputs):
        """Gather per-rank step outputs into a single list on every rank.

        Uses ``dist.all_gather_object`` so each rank ends up with the full
        set of outputs from all ranks.  This is necessary for correct
        ranking-based metrics (R@K, mR@K) which are not decomposable across
        data partitions.
        """
        if not dist.is_available() or not dist.is_initialized():
            return step_outputs

        world_size = dist.get_world_size()
        if world_size <= 1:
            return step_outputs

        # all_gather_object pickles arbitrary Python objects — safe since
        # step_outputs are already on CPU (detach().cpu() in _cache_step_output).
        gathered = [None] * world_size
        try:
            dist.all_gather_object(gathered, step_outputs)
        except RuntimeError as e:
            logging.warning('all_gather_object failed (%s), falling back to local outputs.', e)
            return step_outputs

        all_outputs = []
        for rank_outputs in gathered:
            if rank_outputs is not None:
                all_outputs.extend(rank_outputs)
        return all_outputs

    # ------------------------------------------------------------------
    # Aggregation & epoch-end evaluation
    # ------------------------------------------------------------------

    def _aggregate_step_outputs(self, step_outputs):
        pred_all = {}
        for key in step_outputs[0]['outputs'].keys():
            batch_items = [x['outputs'][key] for x in step_outputs]

            if torch.is_tensor(batch_items[0]):
                pred_all[key] = torch.cat(batch_items, dim=0)
            elif isinstance(batch_items[0], np.ndarray):
                pred_all[key] = np.concatenate(batch_items, axis=0)
            else:
                # Flatten per-batch lists into a flat per-image list
                if isinstance(batch_items[0], list):
                    pred_all[key] = sum(batch_items, [])
                else:
                    pred_all[key] = batch_items

        true_all = []
        for x in step_outputs:
            true_all.extend(x['targets'])

        avg_loss_dict = {}
        loss_keys = step_outputs[0]['loss_dict'].keys()
        for k in loss_keys:
            vals = []
            for x in step_outputs:
                v = x['loss_dict'][k]
                vals.append(v.item() if torch.is_tensor(v) else float(v))
            avg_loss_dict[k] = sum(vals) / max(len(vals), 1)

        total_losses = []
        for x in step_outputs:
            v = x['total_loss']
            total_losses.append(v.item() if torch.is_tensor(v) else float(v))
        avg_total_loss = sum(total_losses) / max(len(total_losses), 1)

        # Merge triplet matching indices (Hungarian matcher) if available
        triplet_match_indices = None
        for x in step_outputs:
            indices = x.get('triplet_indices')
            if indices is None:
                break  # not available for this model
            if triplet_match_indices is None:
                triplet_match_indices = []
            triplet_match_indices.extend(indices)

        return pred_all, true_all, avg_loss_dict, avg_total_loss, triplet_match_indices

    def _run_epoch_end(self, step_outputs, prefix: str):
        if len(step_outputs) == 0:
            logging.warning('%s_outputs is empty at epoch end.', prefix)
            return

        # Gather across DDP ranks (required for correct ranking metrics)
        step_outputs = self._gather_step_outputs(step_outputs)
        if not step_outputs:
            return

        pred_all, true_all, avg_loss_dict, avg_total_loss, triplet_match_indices = \
            self._aggregate_step_outputs(step_outputs)

        # rel_nums convention differs by model:
        #   RelTR / FlowSG / HSTRNet: 51 = 50 preds + 1 bg  →  eval needs 50
        #   EGTR (official):          50 = 50 preds (no bg) →  eval needs 50
        # Heuristic: if rel_nums > 50, bg is included and we subtract 1;
        # otherwise rel_nums is already the pure predicate count.
        rel_nums = self.hparams.rel_nums
        if rel_nums is not None and rel_nums > 50:
            rel_nums = rel_nums - 1
        eval_res, eval_log = metric(
            pred=pred_all,
            true=true_all,
            metrics=self.metric,
            rel_nums=rel_nums,
            entity_nums=self.hparams.entity_nums,
            triplet_match_indices=triplet_match_indices,
        )

        eval_res[f'{prefix}_loss'] = avg_total_loss
        for k, v in avg_loss_dict.items():
            eval_res[f'{prefix}_{k}'] = v

        for k, v in eval_res.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                self.log(
                    k,
                    v,
                    prog_bar=('R@50' in k or 'mR@50' in k or k == f'{prefix}_loss'),
                    sync_dist=True,
                )

        if self.trainer.is_global_zero:
            print(eval_log)

        step_outputs.clear()
        return eval_res

    def on_validation_epoch_end(self):
        return self._run_epoch_end(self.val_outputs, prefix='val')

    def on_test_epoch_end(self):
        return self._run_epoch_end(self.test_outputs, prefix='test')
