import sys
import time
import os.path as osp
from fvcore.nn import FlopCountAnalysis, flop_count_table

import torch

from src.methods import method_maps
from data.dataloaders.base_data import BaseDataModule
from utils import (get_dataset, measure_throughput, SetupCallback, EpochEndCallback, BestCheckpointCallback)
from lightning.pytorch.profilers import AdvancedProfiler
from lightning import seed_everything, Trainer
import lightning.pytorch.callbacks as lc

class BaseExperiment(object):
    """The basic class of PyTorch training and evaluation."""
    def __init__(self, args, dataloaders=None, strategy='auto'):
        """Initialize experiments (non-dist as an example)""" # 初始化实验（以非分布式训练为例）
        # 设置实验参数和配置
        self.args = args
        self.config = self.args.__dict__
        self.method = None
        self.args.method = self.args.method.lower()
        self._dist = self.args.dist
        # 创建工作目录和检查点保存路径
        base_dir = args.res_dir if args.res_dir is not None else 'results'
        save_dir = osp.join(base_dir, args.ex_name if not args.ex_name.startswith(args.res_dir) \
            else args.ex_name.split(args.res_dir+'/')[-1])
        ckpt_dir = osp.join(save_dir, 'checkpoints')

        seed_everything(args.seed)  # 初始化随机种子以确保实验可复现
        self.data = self._get_data(dataloaders)  # 加载数据集和数据加载器
        self.method = method_maps[self.args.method](steps_per_epoch=len(self.data.train_loader), \
            save_dir=save_dir, **self.config)  # 实例化预测模型（从 method_maps 中根据方法名选择）

        callbacks, self.save_dir = self._load_callbacks(args, save_dir, ckpt_dir)  # 配置回调函数（callbacks）
        self.trainer = self._init_trainer(self.args, callbacks, strategy)  # 初始化 Lightning Trainer
    
    def _init_trainer(self, args, callbacks, strategy):
        profiler = AdvancedProfiler(dirpath="logs", filename="profile.txt")
                    
        return Trainer(devices=args.gpus,  # Use these GPUs
                       max_epochs=args.epoch,  # Maximum number of epochs to train for
                       strategy=strategy,   # 'ddp', 'deepspeed_stage_2', 'ddp_find_unused_parameters_false'
                       profiler=profiler,
                       accelerator='gpu',  # Use distributed data parallel
                       callbacks=callbacks
                    )
    
    def _load_callbacks(self, args, save_dir, ckpt_dir):
        method_info = None
        if self._dist == 0:
            if not self.args.no_display_method_info:
                method_info = self.display_method_info(args)

        setup_callback = SetupCallback(
            prefix = 'train' if (not args.test) else 'test',
            setup_time = time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            save_dir = save_dir,
            ckpt_dir = ckpt_dir,
            args = args,
            method_info = method_info,
            argv_content = sys.argv + ["gpus: {}".format(torch.cuda.device_count())],
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
        """Prepare datasets and dataloaders"""
        if dataloaders is None:
            train_loader, vali_loader, test_loader = \
                get_dataset(self.args.dataname, self.config)
        else:
            train_loader, vali_loader, test_loader = dataloaders
    
        vali_loader = test_loader if vali_loader is None else vali_loader
        return BaseDataModule(train_loader, vali_loader, test_loader)
    
    def train(self):
        self.trainer.fit(self.method, self.data, ckpt_path=self.args.ckpt_path if self.args.ckpt_path else None)

    def test(self):
        if self.args.test == True:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            ckpt = torch.load(osp.join(self.save_dir, 'checkpoints', 'best-epoch=01-val_loss=0.333.ckpt'), map_location = device)
            self.method.load_state_dict(ckpt['state_dict'])    
        self.trainer.test(self.method, self.data)
    
    def display_method_info(self, args):
        """Plot the basic infomation of supported methods"""  # 针对不同预测方法创建虚拟输入，计算并展示：模型结构信息、FLOPS、吞吐量/FPS（每秒处理帧数，可选）
        device = torch.device(args.device)
        if args.device == 'cuda':
            assign_gpu = 'cuda:' + (str(args.gpus[0]) if len(args.gpus) == 1 else '0')
            device = torch.device(assign_gpu)
        T, C, H, W = args.in_shape
        if args.method in ['simvp', 'tau', 'mmvp', 'wast']:
            input_dummy = torch.ones(1, args.pre_seq_length, C, H, W).to(device)
        elif args.method in ['hstrnet', 'reltr']:
            input_dummy = torch.ones(1, args.pre_seq_length, C, H, W).to(device)
        elif args.method == 'phydnet':
            _tmp_input1 = torch.ones(1, args.pre_seq_length, C, H, W).to(device)
            _tmp_input2 = torch.ones(1, args.aft_seq_length, C, H, W).to(device)
            _tmp_constraints = torch.zeros((49, 7, 7)).to(device)
            input_dummy = (_tmp_input1, _tmp_input2, _tmp_constraints)
        elif args.method in ['convlstm', 'predrnnpp', 'predrnn', 'mim', 'e3dlstm', 'mau']:
            Hp, Wp = H // args.patch_size, W // args.patch_size
            Cp = args.patch_size ** 2 * C
            _tmp_input = torch.ones(1, args.total_length, Hp, Wp, Cp).to(device)
            _tmp_flag = torch.ones(1, args.aft_seq_length - 1, Hp, Wp, Cp).to(device)
            input_dummy = (_tmp_input, _tmp_flag)
        elif args.method in ['swinlstm_d', 'swinlstm_b']:
            input_dummy = torch.ones(1, self.args.total_length, H, W, C).to(device)
        elif args.method == 'predrnnv2':
            Hp, Wp = H // args.patch_size, W // args.patch_size
            Cp = args.patch_size ** 2 * C
            _tmp_input = torch.ones(1, args.total_length, Hp, Wp, Cp).to(device)
            _tmp_flag = torch.ones(1, args.total_length - 2, Hp, Wp, Cp).to(device)
            input_dummy = (_tmp_input, _tmp_flag)
        elif args.method == 'prednet':
           input_dummy = torch.ones(1, 1, C, H, W, requires_grad=True).to(device)
        else:
            raise ValueError(f'Invalid method name {args.method}')

        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()
        if str(args.method).lower() != 'reltr':
            flops = FlopCountAnalysis(self.method.model.to(device), input_dummy)
            print('FLOPs of {}: \n'.format(args.method), flops)
            flops = flop_count_table(flops)
        else:
            flops = "0M"

        if args.fps:
            fps = measure_throughput(self.method.model.to(device), input_dummy)
            fps = 'Throughputs of {}: {:.3f}\n'.format(args.method, fps)
        else:
            fps = ''
        return info, flops, fps, dash_line