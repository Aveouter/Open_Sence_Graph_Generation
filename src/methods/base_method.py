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
        self.model = self._build_model(**args)
        print(args)
        self.criterion = loss_construction(args['loss'])
        self.test_outputs = []
        self.val_outputs = []

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
        batch_x, batch_y, region = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        
        outputs = {
            'inputs': batch_x.cpu().numpy(),
            'preds': pred_y.cpu().numpy(),
            'trues': batch_y.cpu().numpy(),
            'region': region
        }
        
        self.val_outputs.append(outputs)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return outputs

    def on_validation_epoch_end(self):
        print(f"[DEBUG] on_validation_epoch_end: total {len(self.val_outputs)} batches")

        if len(self.val_outputs) == 0:
            print("[DEBUG] Warning: val_outputs is empty!")
            return

        results_all = {}

        # ------- 合并所有批次 -------
        for k in self.val_outputs[0].keys():
            batch_list = [b[k] for b in self.val_outputs]
            print(f"[DEBUG] merging key '{k}', {len(batch_list)} batches")
            try:
                results_all[k] = np.concatenate(batch_list, axis=0)
                print(f"[DEBUG] {k} shape = {results_all[k].shape}")
            except Exception as e:
                print(f"[DEBUG] concat error on key {k}: {e}")
                raise e

        # ------- metric threshold -------
        thr = self.hparams.get('metric_threshold', None)
        print(f"[DEBUG] metric_threshold = {thr}")

        if self.hparams.test_mean == 0 and self.hparams.test_std == 0:
            self.hparams.test_mean = None
            self.hparams.test_std = None

        # ------- 调用 metric -------
        eval_res, eval_log = metric(
            results_all['preds'],
            results_all['trues'],
            results_all['region'],
            self.hparams.test_mean,
            self.hparams.test_std,
            metrics=self.metric_list,
            channel_names=self.channel_names,
            spatial_norm=self.spatial_norm,
            threshold=thr
        )

        print(f"[DEBUG] eval_res keys = {list(eval_res.keys())}")

        # ------- 打印验证日志 -------
        if self.trainer.is_global_zero:
            print_log("[VAL] " + eval_log)

        # 清空缓存
        self.val_outputs.clear()

        return results_all

        
    def test_step(self, batch, batch_idx):
        batch_x, batch_y ,region = batch
        pred_y = self(batch_x, batch_y)
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy(),'region':region}
        self.test_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        results_all = {}

        # ------- Debug 1: 检查 test_outputs -------
        print(f"[DEBUG] len(test_outputs) = {len(self.test_outputs)}")
        if len(self.test_outputs) == 0:
            print("[DEBUG] Warning: self.test_outputs 为空，可能 test_step 没有返回任何结果。")

        # ------- 收集所有批次的预测结果 -------
        for k in self.test_outputs[0].keys():
            batch_list = [batch[k] for batch in self.test_outputs]
            print(f"[DEBUG] merging key '{k}', {len(batch_list)} batches, type={type(batch_list[0])}")
            try:
                results_all[k] = np.concatenate(batch_list, axis=0)
                print(f"[DEBUG] {k} shape after concat: {results_all[k].shape}")
            except Exception as e:
                print(f"[DEBUG] concat error on key {k}: {e}")
                raise e

        # ------- Debug 2: 检查传入 metric() 的参数 -------
        thr = self.hparams.get('metric_threshold', None)
        print(f"[DEBUG] metric_threshold from hparams = {thr}")

        if thr is None:
            print("[DEBUG] ⚠️ Warning: metric_threshold 是 None，将使用默认阈值（如 0.5）")
        if self.hparams.test_mean == 0 and self.hparams.test_std == 0:
            self.hparams.test_mean = None
            self.hparams.test_std = None
        # ------- 调用 metric 函数 -------
        eval_res, eval_log = metric(
            results_all['preds'],
            results_all['trues'],
            results_all['region'],
            self.hparams.test_mean,
            self.hparams.test_std,
            metrics=self.metric_list,
            channel_names=self.channel_names,
            spatial_norm=self.spatial_norm,
            threshold=thr
        )

        # ------- Debug 3: 检查 metric 输出 -------
        print(f"[DEBUG] eval_res keys = {list(eval_res.keys())}")
        print(f"[DEBUG] eval_log type = {type(eval_log)}")

        results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

        # ------- 保存 -------
        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

            for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
                print(f"[DEBUG] saved {np_data}.npy to {folder_path}")

        return results_all
