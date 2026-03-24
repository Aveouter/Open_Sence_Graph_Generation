import os
import sys
import time
import os.path as osp

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from lightning import seed_everything, Trainer
from lightning.pytorch.profilers import AdvancedProfiler
import lightning.pytorch.callbacks as lc

from src.methods import method_maps
from data.dataloaders.base_data import BaseDataModule
from utils import (
    get_dataset,
    measure_throughput,
    SetupCallback,
    EpochEndCallback,
    BestCheckpointCallback,
)


class BaseExperiment(object):
    """Experiment class specialized for the current RelTR-based scene graph task."""

    def __init__(self, args, dataloaders=None, strategy='auto'):
        self.args = args
        self.config = self.args.__dict__
        self.method = None
        self.args.method = self.args.method.lower()
        self._dist = self.args.dist

        base_dir = args.res_dir if args.res_dir is not None else 'results'
        save_dir = osp.join(
            base_dir,
            args.ex_name if not args.ex_name.startswith(args.res_dir)
            else args.ex_name.split(args.res_dir + '/')[-1]
        )
        ckpt_dir = osp.join(save_dir, 'checkpoints')

        seed_everything(args.seed)

        self.data = self._get_data(dataloaders)
        self.method = method_maps[self.args.method](
            steps_per_epoch=len(self.data.train_loader),
            save_dir=save_dir,
            **self.config
        )

        callbacks, self.save_dir = self._load_callbacks(args, save_dir, ckpt_dir)
        self.trainer = self._init_trainer(self.args, callbacks, strategy)

    def _init_trainer(self, args, callbacks, strategy):
        profiler = AdvancedProfiler(dirpath="logs", filename="profile.txt")

        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        devices = args.gpus if accelerator == 'gpu' else 1

        return Trainer(
            devices=devices,
            max_epochs=args.epoch,
            strategy=strategy,
            profiler=profiler,
            accelerator=accelerator,
            callbacks=callbacks,
            log_every_n_steps=1,
        )

    def _load_callbacks(self, args, save_dir, ckpt_dir):
        method_info = None
        if self._dist == 0 and (not self.args.no_display_method_info):
            method_info = self.display_method_info(args)

        setup_callback = SetupCallback(
            prefix='train' if (not args.test) else 'test',
            setup_time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            save_dir=save_dir,
            ckpt_dir=ckpt_dir,
            args=args,
            method_info=method_info,
            argv_content=sys.argv + [f"gpus: {torch.cuda.device_count()}"],
        )

        ckpt_callback = BestCheckpointCallback(
            monitor=args.metric_for_bestckpt,
            filename='best-{epoch:02d}-{val_loss:.3f}',
            mode='min',
            save_last=True,
            dirpath=ckpt_dir,
            verbose=True,
            every_n_epochs=args.log_step,
        )

        epochend_callback = EpochEndCallback()

        callbacks = [setup_callback, ckpt_callback, epochend_callback]
        if args.sched:
            callbacks.append(lc.LearningRateMonitor(logging_interval=None))

        return callbacks, save_dir

    def _get_data(self, dataloaders=None):
        """Prepare datasets and dataloaders."""
        if dataloaders is None:
            train_loader, vali_loader, test_loader = get_dataset(self.args.dataname, self.config)
        else:
            train_loader, vali_loader, test_loader = dataloaders

        vali_loader = test_loader if vali_loader is None else vali_loader
        return BaseDataModule(train_loader, vali_loader, test_loader)

    def train(self):
        self.trainer.fit(
            self.method,
            self.data,
            ckpt_path=self.args.ckpt_path if self.args.ckpt_path else None
        )

    def test(self):
        """
        Test behavior:
        1. If args.ckpt_path is provided, use it.
        - .ckpt: use Lightning restore
        - .pth/.pt: manually load state_dict
        2. Else if args.test is True, try to load save_dir/checkpoints/last.ckpt first,
        then fall back to best checkpoint in that folder.
        3. Else directly test current model state.
        """
        ckpt_path = None

        if self.args.ckpt_path:
            ckpt_path = self.args.ckpt_path
        elif self.args.test:
            ckpt_dir = osp.join(self.save_dir, 'checkpoints')
            last_ckpt = osp.join(ckpt_dir, 'last.ckpt')

            if osp.exists(last_ckpt):
                ckpt_path = last_ckpt
            elif osp.isdir(ckpt_dir):
                ckpt_files = [
                    osp.join(ckpt_dir, f)
                    for f in os.listdir(ckpt_dir)
                    if f.endswith('.ckpt') and f.startswith('best-')
                ]
                if len(ckpt_files) > 0:
                    ckpt_files.sort(key=lambda x: osp.getmtime(x), reverse=True)
                    ckpt_path = ckpt_files[0]

        # 没有提供权重，直接测试当前模型
        if ckpt_path is None:
            print('[Info] No checkpoint provided, testing current model state.')
            return self.trainer.test(self.method, self.data)

        ext = osp.splitext(ckpt_path)[1].lower()

        # 1) Lightning checkpoint
        if ext == '.ckpt':
            print(f'[Info] Testing with Lightning checkpoint: {ckpt_path}')
            return self.trainer.test(self.method, self.data, ckpt_path=ckpt_path)

        # 2) 普通 PyTorch 权重
        if ext in ['.pth', '.pt']:
            print(f'[Info] Loading PyTorch weights from: {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            if isinstance(ckpt, dict):
                if 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                elif 'model_state_dict' in ckpt:
                    state_dict = ckpt['model_state_dict']
                elif 'model' in ckpt:
                    state_dict = ckpt['model']
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt

            # 去掉 module. 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[len('module.'):]] = v
                else:
                    new_state_dict[k] = v

            missing, unexpected = self.method.model.load_state_dict(new_state_dict, strict=False)

            print(f'[Info] Missing keys: {len(missing)}')
            if len(missing) > 0:
                print(missing[:20])

            print(f'[Info] Unexpected keys: {len(unexpected)}')
            if len(unexpected) > 0:
                print(unexpected[:20])

            return self.trainer.test(self.method, self.data)

        raise ValueError(f'Unsupported checkpoint format: {ckpt_path}')

    def display_method_info(self, args):
        """
        Show model structure / FLOPs / throughput.
        This version is adapted for RelTR-like image relation generation tasks.
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if args.device == 'cuda' and torch.cuda.is_available():
            assign_gpu = 'cuda:' + (str(args.gpus[0]) if len(args.gpus) == 1 else '0')
            device = torch.device(assign_gpu)

        _, C, H, W = args.in_shape
        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()

        # Dummy input specialized for RelTR/current task
        if args.method == 'reltr':
            # RelTR expects image list -> NestedTensor
            input_dummy = torch.ones(1, C, H, W).to(device)
        else:
            # fallback for other methods if you still use this exp.py
            input_dummy = torch.ones(1, C, H, W).to(device)

        # FLOPs
        try:
            model = self.method.model.to(device)
            model.eval()

            if args.method == 'reltr':
                image_list = [img.to(device) for img in input_dummy]
                nested_dummy = self.method._to_nested_tensor(image_list)
                flops_obj = FlopCountAnalysis(model, nested_dummy)
            else:
                flops_obj = FlopCountAnalysis(model, input_dummy)

            print(f'FLOPs of {args.method}:\n', flops_obj)
            flops = flop_count_table(flops_obj)
        except Exception as e:
            print(f'[Warning] FLOPs calculation failed for {args.method}: {e}')
            flops = f'Unavailable ({type(e).__name__})'

        # Throughput
        if args.fps:
            try:
                if args.method == 'reltr':
                    fps = 'Throughputs of {}: skipped for RelTR nested input\n'.format(args.method)
                else:
                    fps_value = measure_throughput(self.method.model.to(device), input_dummy)
                    fps = 'Throughputs of {}: {:.3f}\n'.format(args.method, fps_value)
            except Exception as e:
                fps = f'Throughputs of {args.method}: Unavailable ({type(e).__name__})\n'
        else:
            fps = ''

        return info, flops, fps, dash_line