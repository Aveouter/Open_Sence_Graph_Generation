import numpy as np
import torch.nn as nn
import os.path as osp
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
    
    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch):
        NotImplementedError
    
    def training_step(self, batch, batch_idx):
        NotImplementedError

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

        # 只缓存评测真正需要的内容
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
        # print(f"[DEBUG] on_validation_epoch_end: total {len(self.val_outputs)} batches")

        if len(self.val_outputs) == 0:
            print("[DEBUG] Warning: val_outputs is empty!")
            return

        # 1. 聚合整个验证集的 outputs
        pred_all = {}
        for key in self.val_outputs[0]['outputs'].keys():
            batch_items = [x['outputs'][key] for x in self.val_outputs]

            if torch.is_tensor(batch_items[0]):
                pred_all[key] = torch.cat(batch_items, dim=0)
            elif isinstance(batch_items[0], np.ndarray):
                pred_all[key] = np.concatenate(batch_items, axis=0)
            else:
                pred_all[key] = batch_items

        # 2. 聚合整个验证集的 targets（list[dict] 直接拼接列表）
        true_all = []
        for x in self.val_outputs:
            true_all.extend(x['targets'])

        # 3. 计算平均 loss_dict，便于日志记录
        avg_loss_dict = {}
        loss_keys = self.val_outputs[0]['loss_dict'].keys()
        for k in loss_keys:
            vals = []
            for x in self.val_outputs:
                v = x['loss_dict'][k]
                if torch.is_tensor(v):
                    vals.append(v.item())
                else:
                    vals.append(float(v))
            avg_loss_dict[k] = sum(vals) / max(len(vals), 1)

        total_losses = []
        for x in self.val_outputs:
            v = x['total_loss']
            if torch.is_tensor(v):
                total_losses.append(v.item())
            else:
                total_losses.append(float(v))
        avg_total_loss = sum(total_losses) / max(len(total_losses), 1)

        # 4. 调统一 metric 函数
        eval_metrics = self.metric
        eval_res, eval_log = metric(
            pred=pred_all,
            true=true_all,
            metrics=eval_metrics,
            rel_nums=self.hparams.rel_nums,
            entity_nums=self.hparams.entity_nums
        )

        # 5. 把 loss 类信息也补进结果里
        eval_res['val_loss'] = avg_total_loss
        for k, v in avg_loss_dict.items():
            eval_res[f'val_{k}'] = v

        # 6. log
        for k, v in eval_res.items():
            if isinstance(v, (int, float)):
                self.log(k, v, prog_bar=('R@50' in k or 'mR@50' in k or k == 'val_loss'), sync_dist=True)

        print(eval_log)

        results_all = eval_res

        # 7. 清空缓存
        self.val_outputs.clear()

        return results_all

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
        # print(f"[DEBUG] on_test_epoch_end: total {len(self.test_outputs)} batches")

        if len(self.test_outputs) == 0:
            print("[DEBUG] Warning: test_outputs is empty!")
            return

        pred_all = {}
        for key in self.test_outputs[0]['outputs'].keys():
            batch_items = [x['outputs'][key] for x in self.test_outputs]

            if torch.is_tensor(batch_items[0]):
                pred_all[key] = torch.cat(batch_items, dim=0)
            elif isinstance(batch_items[0], np.ndarray):
                pred_all[key] = np.concatenate(batch_items, axis=0)
            else:
                pred_all[key] = batch_items

        true_all = []
        for x in self.test_outputs:
            true_all.extend(x['targets'])

        avg_loss_dict = {}
        loss_keys = self.test_outputs[0]['loss_dict'].keys()
        for k in loss_keys:
            vals = []
            for x in self.test_outputs:
                v = x['loss_dict'][k]
                if torch.is_tensor(v):
                    vals.append(v.item())
                else:
                    vals.append(float(v))
            avg_loss_dict[k] = sum(vals) / max(len(vals), 1)

        total_losses = []
        for x in self.test_outputs:
            v = x['total_loss']
            if torch.is_tensor(v):
                total_losses.append(v.item())
            else:
                total_losses.append(float(v))
        avg_total_loss = sum(total_losses) / max(len(total_losses), 1)
        test_metrics = self.metric

        eval_res, eval_log = metric(
            pred=pred_all,
            true=true_all,
            metrics=test_metrics,
            rel_nums=self.hparams.rel_nums,
            entity_nums=self.hparams.entity_nums
        )

        eval_res['test_loss'] = avg_total_loss
        for k, v in avg_loss_dict.items():
            eval_res[f'test_{k}'] = v

        for k, v in eval_res.items():
            if isinstance(v, (int, float)):
                self.log(k, v, prog_bar=('R@50' in k or 'mR@50' in k or k == 'test_loss'), sync_dist=True)

        print(eval_log)

        results_all = eval_res
        self.test_outputs.clear()
        return results_all