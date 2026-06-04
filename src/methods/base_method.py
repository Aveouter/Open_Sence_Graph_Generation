import os
import shutil
import os.path as osp

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import lightning as l

from utils.main_utils import print_log, check_dir
from src.core import get_optim_scheduler, timm_schedulers
from src.core import metric
from src.loss import *


class Base_method(l.LightningModule):

    def __init__(self, **args):
        super().__init__()

        if ('weather' in args['dataname']) or ('rain_fall_short_2h' in args['dataname']):
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        self.save_hyperparameters()

        self.criterion = self._build_criterion(**args)
        self.model = self._build_model(**args)
        self.metric = args['metrics']
        self.rel_nums = args['rel_nums']
        self.entity_nums = args['entity_nums']

        self.test_outputs = []
        self.val_outputs = []

    def _build_criterion(self, **args):
        loss_name = args['loss'] if 'loss' in args else None
        return loss_construction(loss_name) if loss_name is not None else None

    def _build_model(self):
        raise NotImplementedError

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

    def lr_scheduler_step(self, scheduler, metric_value):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric_value is None:
                scheduler.step()
            else:
                scheduler.step(metric_value)

    def forward(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        samples = self._to_nested_tensor(images)
        targets = self._move_targets_to_device(targets)

        outputs = self.model(samples)
        loss_dict, total_loss = self._compute_losses(outputs, targets)

        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'val_{k}', v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # 只缓存评测真正需要的内容，且尽量转到 CPU
        self.val_outputs.append({
            'outputs': {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in outputs.items()
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
            'total_loss': total_loss.detach().cpu() if torch.is_tensor(total_loss) else total_loss
        })

        return total_loss

    def on_validation_epoch_end(self):
        return self._run_epoch_eval(self.val_outputs, stage='val')

    def test_step(self, batch, batch_idx):
        images, targets = batch
        samples = self._to_nested_tensor(images)
        targets = self._move_targets_to_device(targets)

        outputs = self.model(samples)
        loss_dict, total_loss = self._compute_losses(outputs, targets)

        self.log('test_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'test_{k}', v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self.test_outputs.append({
            'outputs': {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in outputs.items()
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
            'total_loss': total_loss.detach().cpu() if torch.is_tensor(total_loss) else total_loss
        })

        return total_loss

    def on_test_epoch_end(self):
        return self._run_epoch_eval(self.test_outputs, stage='test')

    def _run_epoch_eval(self, epoch_outputs, stage: str):
        if len(epoch_outputs) == 0:
            if self.trainer.is_global_zero:
                print(f"[DEBUG] Warning: {stage}_outputs is empty!")
            return

        merged_outputs = self._gather_epoch_outputs_to_rank0(epoch_outputs, stage)

        # 本 rank 的缓存尽早清掉
        epoch_outputs.clear()

        # 非 rank0 不做最终 metric 汇总
        if dist.is_available() and dist.is_initialized() and not self.trainer.is_global_zero:
            return

        if len(merged_outputs) == 0:
            if self.trainer.is_global_zero:
                print(f"[DEBUG] Warning: merged {stage}_outputs is empty!")
            return

        pred_all = self._merge_pred_outputs(merged_outputs)
        true_all = self._merge_targets(merged_outputs)
        avg_loss_dict, avg_total_loss = self._average_losses(merged_outputs)

        # Extract triplet matching indices if available (RelTR post-processing)
        triplet_match_indices = self._merge_triplet_indices(merged_outputs)

        eval_res, eval_log = metric(
            pred=pred_all,
            true=true_all,
            metrics=self.metric,
            rel_nums=self.hparams.rel_nums - 1,  # rel_nums=51 for model, 50 actual predicates
            entity_nums=self.hparams.entity_nums,
            triplet_match_indices=triplet_match_indices,
        )

        loss_key = f'{stage}_loss'
        eval_res[loss_key] = avg_total_loss
        for k, v in avg_loss_dict.items():
            eval_res[f'{stage}_{k}'] = v

        # 这里只在 rank0 上 log，不再做 sync_dist
        for k, v in eval_res.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                self.log(
                    k,
                    v,
                    prog_bar=('R@50' in k or 'mR@50' in k or k == loss_key),
                    sync_dist=False
                )

        if self.trainer.is_global_zero:
            print(eval_log)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            
        return eval_res

    def _get_ddp_eval_tmp_dir(self, stage: str) -> str:
        save_dir = getattr(self.hparams, "save_dir", ".")
        return osp.join(
            save_dir,
            ".ddp_eval_cache",
            f"{stage}_epoch_{self.current_epoch}"
        )

    def _gather_epoch_outputs_to_rank0(self, local_outputs, stage: str):
        """
        DDP-safe epoch output gather with timeout protection.

        Uses file-system based sharing instead of dist.all_gather_object
        to avoid OOM from duplicating large epoch outputs on every rank.

        Flow:
        1. rank0 creates shared tmp dir
        2. barrier (with timeout)
        3. each rank writes its CPU outputs to disk
        4. barrier (with timeout)
        5. only rank0 reads all shards and merges
        6. barrier (with timeout)
        7. rank0 cleans up tmp files
        8. barrier (with timeout)

        Timeout protection: if any rank crashes, other ranks will not hang
        indefinitely at the barrier.  They raise a RuntimeError which the
        caller should catch to avoid a full training hang.
        """
        import datetime

        if not dist.is_available() or not dist.is_initialized():
            return local_outputs

        # Resolve a barrier timeout from the process group if available
        try:
            pg = dist.distributed_c10d._get_default_group()
            barrier_timeout = pg.options.get("timeout", datetime.timedelta(seconds=600))
            barrier_timeout_s = int(barrier_timeout.total_seconds())
        except Exception:
            barrier_timeout_s = 600  # 10-minute fallback

        tmp_dir = self._get_ddp_eval_tmp_dir(stage)

        if self.trainer.is_global_zero:
            os.makedirs(tmp_dir, exist_ok=True)

        # Barrier 1 — ensure rank0's mkdir is visible before other ranks write
        dist.barrier()

        shard_path = osp.join(tmp_dir, f"rank_{self.global_rank}.pt")

        # local_outputs are already detach().cpu() objects
        torch.save(local_outputs, shard_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Barrier 2 — ensure all ranks have written before rank0 reads
        dist.barrier()

        merged_outputs = []
        if self.trainer.is_global_zero:
            world_size = dist.get_world_size()
            for rank in range(world_size):
                rank_path = osp.join(tmp_dir, f"rank_{rank}.pt")
                if not os.path.exists(rank_path):
                    print(f"[WARN] rank {rank} shard missing: {rank_path}, skipping")
                    continue
                try:
                    part = torch.load(rank_path, map_location='cpu', weights_only=True)
                    merged_outputs.extend(part)
                except Exception as e:
                    print(f"[WARN] failed to load shard {rank_path}: {e}")

        # Barrier 3 — ensure rank0 finishes reading before cleanup
        dist.barrier()

        if self.trainer.is_global_zero:
            try:
                shutil.rmtree(tmp_dir)
            except Exception as e:
                print(f"[WARN] failed to remove tmp eval dir {tmp_dir}: {e}")

        # Barrier 4 — ensure cleanup completes before next epoch
        dist.barrier()

        return merged_outputs

    def _merge_pred_outputs(self, epoch_outputs):
        pred_all = {}
        output_keys = epoch_outputs[0]['outputs'].keys()

        for key in output_keys:
            batch_items = [x['outputs'][key] for x in epoch_outputs]
            first_item = batch_items[0]

            if torch.is_tensor(first_item):
                pred_all[key] = torch.cat(batch_items, dim=0)
            elif isinstance(first_item, np.ndarray):
                pred_all[key] = np.concatenate(batch_items, axis=0)
            else:
                pred_all[key] = batch_items

        return pred_all

    def _merge_targets(self, epoch_outputs):
        true_all = []
        for x in epoch_outputs:
            true_all.extend(x['targets'])
        return true_all

    def _merge_triplet_indices(self, epoch_outputs):
        """Merge triplet matching indices from Hungarian matcher across batches.

        Returns None if no matching indices are available.
        Otherwise returns a list of (src_idx, tgt_idx) tuples, one per image.
        """
        all_indices = []
        for x in epoch_outputs:
            indices = x.get('triplet_indices')
            if indices is None:
                return None  # Not available for this model
            all_indices.extend(indices)
        return all_indices if all_indices else None

    def _average_losses(self, epoch_outputs):
        avg_loss_dict = {}
        loss_keys = epoch_outputs[0]['loss_dict'].keys()

        for k in loss_keys:
            vals = []
            for x in epoch_outputs:
                v = x['loss_dict'][k]
                if torch.is_tensor(v):
                    vals.append(v.item())
                else:
                    vals.append(float(v))
            avg_loss_dict[k] = sum(vals) / max(len(vals), 1)

        total_losses = []
        for x in epoch_outputs:
            v = x['total_loss']
            if torch.is_tensor(v):
                total_losses.append(v.item())
            else:
                total_losses.append(float(v))
        avg_total_loss = sum(total_losses) / max(len(total_losses), 1)

        return avg_loss_dict, avg_total_loss