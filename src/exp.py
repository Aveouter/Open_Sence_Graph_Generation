import os
import sys
import time
import os.path as osp
from datetime import datetime

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from lightning import seed_everything, Trainer
from lightning.pytorch.loggers import CSVLogger
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
from utils.path_utils import (
    resolve_output_paths,
    normalize_ex_name,
    ensure_unique_run_dir,
    collect_metadata,
    write_metadata,
    save_eval_results,
)


class BaseExperiment(object):
    """Experiment class specialized for the current RelTR-based scene graph task."""

    def __init__(self, args, dataloaders=None, strategy='auto'):
        self.args = args
        self.config = self.args.__dict__
        self.method = None
        self.args.method = self.args.method.lower()
        self._dist = self.args.dist

        # ---- normalise & resolve output paths ----
        args.ex_name = normalize_ex_name(args.ex_name)
        paths = resolve_output_paths(args)
        run_dir, actual_ex_name = ensure_unique_run_dir(
            paths['run_dir'],
            overwrite=getattr(args, 'overwrite', False),
        )
        # re-resolve with the potentially-incremented directory
        paths = resolve_output_paths(args, run_dir_override=run_dir)
        self.paths = paths
        self.save_dir = run_dir  # kept for backward-compat references

        # ---- metadata (written immediately so a record exists even on early failure) ----
        start_time = datetime.now().isoformat()
        metadata = collect_metadata(args, run_dir, start_time)
        metadata['ex_name'] = actual_ex_name   # CRITICAL: use resolved name, not raw input
        write_metadata(run_dir, metadata)

        seed_everything(args.seed)

        self.data = self._get_data(dataloaders)
        self.method = method_maps[self.args.method](
            steps_per_epoch=len(self.data.train_loader),
            save_dir=run_dir,
            **self.config
        )

        callbacks, self.save_dir = self._load_callbacks(args, paths)
        self.trainer = self._init_trainer(self.args, callbacks, strategy, paths)

    def _normalize_devices(self, devices):
        if devices is None:
            return 1
        # Convert single-element list to int 1 to prevent Lightning from spawning DDP workers
        # (devices=[0] causes DDP; devices=1 uses single GPU without DDP)
        if isinstance(devices, (list, tuple)) and len(devices) == 1:
            return 1
        return devices

    def _count_devices(self, devices):
        if devices == "auto":
            return torch.cuda.device_count()

        if isinstance(devices, (list, tuple)):
            return len(devices)

        if isinstance(devices, str):
            if devices == "auto":
                return torch.cuda.device_count()
            if "," in devices:
                return len([d for d in devices.split(",") if d.strip() != ""])
            return int(devices)

        return int(devices)

    def _resolve_trainer_runtime(self, args, strategy):
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

        if accelerator == 'gpu':
            raw_devices = getattr(args, 'gpus', 1)
            devices = self._normalize_devices(raw_devices)
        else:
            devices = 1

        device_count = self._count_devices(devices)

        if strategy is not None and strategy != 'auto':
            resolved_strategy = strategy
        else:
            if accelerator == 'gpu' and device_count > 1:
                resolved_strategy = 'ddp'
            else:
                resolved_strategy = 'auto'

        return accelerator, devices, device_count, resolved_strategy

    def _init_trainer(self, args, callbacks, strategy, paths):
        accelerator, devices, device_count, resolved_strategy = self._resolve_trainer_runtime(args, strategy)

        profiler = None
        if device_count <= 1:
            profiler = AdvancedProfiler(
                dirpath=paths['profiler_dir'],
                filename="profile.txt",
            )

        # CSVLogger writes lightning metrics directly into run_dir/lightning/
        logger = CSVLogger(
            save_dir=paths['lightning_dir'],
            name='',
            version='',
        )

        trainer_kwargs = dict(
            devices=devices,
            max_epochs=args.epoch,
            strategy=resolved_strategy,
            profiler=profiler,
            accelerator=accelerator,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=1,
        )

        if hasattr(args, 'num_nodes'):
            trainer_kwargs['num_nodes'] = args.num_nodes

        return Trainer(**trainer_kwargs)

    def _load_callbacks(self, args, paths):
        method_info = None
        if self._dist == 0 and (not self.args.no_display_method_info):
            method_info = self.display_method_info(args)

        setup_callback = SetupCallback(
            prefix='train' if (not args.test) else 'test',
            setup_time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            paths=paths,
            args=args,
            method_info=method_info,
            argv_content=sys.argv + [f"gpus: {torch.cuda.device_count()}"],
        )

        ckpt_callback = BestCheckpointCallback(
            monitor=args.metric_for_bestckpt,
            filename='best-{epoch:02d}-{val_loss:.3f}',
            mode='min',
            save_last=True,
            dirpath=paths['ckpt_dir'],
            verbose=True,
            every_n_epochs=args.log_step,
        )

        epochend_callback = EpochEndCallback()

        callbacks = [setup_callback, ckpt_callback, epochend_callback]
        if args.sched:
            callbacks.append(lc.LearningRateMonitor(logging_interval=None))

        return callbacks, paths['run_dir']

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
        """Test the model, loading a checkpoint if available.

        Handles size mismatches between checkpoint and current model config
        (e.g., different entity_nums / rel_nums) by truncating or padding
        weight tensors.  Works for both Lightning (.ckpt) and raw PyTorch
        (.pth / .pt) checkpoints.
        """
        ckpt_path = None
        ckpt_dir = self.paths['ckpt_dir']

        if self.args.ckpt_path:
            ckpt_path = self.args.ckpt_path
        elif self.args.test:
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

        if ckpt_path is None:
            print('[Info] No checkpoint provided, testing current model state.')
            result = self.trainer.test(self.method, self.data)
            self._save_test_results(result)
            return result

        # Load and adapt weights — unified for both .ckpt and .pth/.pt
        print(f'[Info] Loading checkpoint: {ckpt_path}')
        state_dict = self._load_checkpoint_state_dict(ckpt_path)
        self._adapt_state_dict(state_dict, self.method.model)
        result = self.trainer.test(self.method, self.data)
        self._save_test_results(result)
        return result

    # ------------------------------------------------------------------
    # Checkpoint loading helpers (shared by train / test)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_checkpoint_state_dict(ckpt_path: str):
        """Extract a raw state dict from a .ckpt, .pth, or .pt file.

        Uses ``weights_only=True`` (PyTorch safe loading).  If a legacy
        checkpoint requires ``weights_only=False``, manually import it
        with ``torch.load(..., weights_only=False)`` and call
        ``_adapt_state_dict`` yourself.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

        if not isinstance(ckpt, dict):
            return ckpt  # raw state dict

        # Lightning .ckpt: state_dict is nested under 'state_dict'
        if 'state_dict' in ckpt:
            raw = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            raw = ckpt['model_state_dict']
        elif 'model' in ckpt:
            raw = ckpt['model']
        else:
            raw = ckpt

        # Strip 'model.' prefix if present (Lightning wraps the model)
        result = {}
        for k, v in raw.items():
            if k.startswith('model.'):
                result[k[6:]] = v
            else:
                result[k] = v
        return result

    @staticmethod
    def _adapt_state_dict(state_dict, model):
        """Load *state_dict* into *model*, adapting mismatched tensor shapes.

        For each parameter whose shape differs between the checkpoint and the
        current model config, each dimension is independently truncated (if
        ckpt is larger) or zero-padded (if ckpt is smaller).

        This allows a checkpoint trained with, e.g., entity_nums=150 to be
        loaded into a model configured with entity_nums=151.
        """
        if not isinstance(state_dict, dict):
            raise TypeError(f'Expected dict state_dict, got {type(state_dict)}')

        model_state = model.state_dict()
        adapted = 0
        for k in list(state_dict.keys()):
            if k not in model_state:
                continue
            ckpt_w = state_dict[k]
            model_w = model_state[k]
            if ckpt_w.shape == model_w.shape:
                continue

            print(f'[Info] Size mismatch for {k}: ckpt {list(ckpt_w.shape)} '
                  f'vs model {list(model_w.shape)}, adapting...')

            w = ckpt_w
            for dim_idx in range(w.dim()):
                if w.shape[dim_idx] > model_w.shape[dim_idx]:
                    w = w.index_select(
                        dim_idx,
                        torch.arange(model_w.shape[dim_idx], device=w.device))
                elif w.shape[dim_idx] < model_w.shape[dim_idx]:
                    pad_shape = list(w.shape)
                    pad_shape[dim_idx] = model_w.shape[dim_idx] - w.shape[dim_idx]
                    pad = torch.zeros(pad_shape, dtype=w.dtype, device=w.device)
                    w = torch.cat([w, pad], dim=dim_idx)
            state_dict[k] = w
            adapted += 1

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if adapted > 0:
            print(f'[Info] Adapted {adapted} weight(s) with size mismatch')
        if len(missing) > 0:
            print(f'[Info] Missing keys ({len(missing)}): {missing[:10]}...' if len(missing) > 10
                  else f'[Info] Missing keys ({len(missing)}): {missing}')
        if len(unexpected) > 0:
            print(f'[Info] Unexpected keys ({len(unexpected)}): {unexpected[:10]}...' if len(unexpected) > 10
                  else f'[Info] Unexpected keys ({len(unexpected)}): {unexpected}')

    def _save_test_results(self, result):
        """Persist Lightning test results into ``eval/<eval_mode>/``."""
        if result is None or len(result) == 0:
            return

        eval_mode = getattr(self.args, 'eval_mode', 'sgdet')
        # Lightning test returns a list of dicts; merge them
        if isinstance(result, list):
            merged = {}
            for d in result:
                if isinstance(d, dict):
                    merged.update({k: float(v) for k, v in d.items()})
            save_eval_results(self.paths['eval_dir'], eval_mode, merged)
        elif isinstance(result, dict):
            save_eval_results(self.paths['eval_dir'], eval_mode,
                              {k: float(v) for k, v in result.items()})

    def display_method_info(self, args):
        """
        Show model structure / FLOPs / throughput.
        This version is adapted for RelTR-like image relation generation tasks.
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if getattr(args, 'device', 'cuda') == 'cuda' and torch.cuda.is_available():
            gpus = getattr(args, 'gpus', [0])
            if isinstance(gpus, (list, tuple)) and len(gpus) > 0:
                assign_gpu = 'cuda:' + str(gpus[0])
            else:
                assign_gpu = 'cuda:0'
            device = torch.device(assign_gpu)

        _, C, H, W = args.in_shape
        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()

        if args.method == 'reltr':
            input_dummy = torch.ones(1, C, H, W).to(device)
        else:
            input_dummy = torch.ones(1, C, H, W).to(device)

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
