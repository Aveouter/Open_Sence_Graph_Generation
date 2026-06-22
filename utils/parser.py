# Copyright (c) CIGIT HPC Lab. All rights reserved

import argparse


def create_parser():
    parser = argparse.ArgumentParser(
        description='OpenSTL train/test a model')
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--dist', action='store_true', default=False,
                        help='Whether to use distributed training (DDP)')
    parser.add_argument('--output_dir', default='./outputs', type=str,
                        help='Root directory for all outputs (runs, pretrained weights, etc.)')
    parser.add_argument('--ex_name', '-ex', default='Debug', type=str,
                        help='Experiment name. Auto-prefixed with date if not already YYYY-MM-DD_.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing run directory instead of auto-incrementing (_001, _002, ...). '
                             'Also allows CLI args to overwrite config file values.')
    parser.add_argument('--eval_mode', default='sgdet', type=str,
                        choices=['predcls', 'sgcls', 'sgdet'],
                        help='Evaluation mode for saving results (default: sgdet).')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Whether to use Native AMP for mixed precision training (PyTorch=>1.6.0)')
    parser.add_argument('--torchscript', action='store_true', default=False,
                        help='Whether to use torchscripted model')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fps', action='store_true', default=False,
                        help='Whether to measure inference speed (FPS)')
    parser.add_argument('--test', action='store_true', default=False, help='Only performs testing')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='Enable PyTorch Lightning AdvancedProfiler (single-GPU only)')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='whether to set deterministic options for CUDNN backend (reproducable)')

    # dataset parameters
    parser.add_argument('--batch_size', '-b', default=None, type=int, help='Training batch size')
    parser.add_argument('--val_batch_size', '-vb', default=None, type=int, help='Validation batch size')
    parser.add_argument('--accumulate_grad_batches', default=None, type=int,
                        help='Gradient accumulation steps (effective bs = batch_size × this)')
    parser.add_argument('--num_workers', default=None, type=int)
    parser.add_argument('--dataset_size', default=None, type=int,
                        help='Optional train subset size for quick smoke tests')
    parser.add_argument('--val_dataset_size', default=None, type=int,
                        help='Optional val/test subset size for quick smoke tests')
    parser.add_argument('--test_dataset_size', default=None, type=int,
                        help='Optional test subset size; overrides val_dataset_size in eval mode')
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--dataname', '-d', default='VisualGenome', type=str,
                        choices=['ThyroTriples', 'VisualGenome', 'OpenImageV6',],
                        help='Dataset name (default: "rain_fall_short_2h")')
    parser.add_argument('--pre_seq_length', default=None, type=int, help='Sequence length before prediction')
    parser.add_argument('--aft_seq_length', default=None, type=int, help='Sequence length after prediction')
    parser.add_argument('--total_length', default=None, type=int, help='Total Sequence length for prediction')
    parser.add_argument('--use_augment', action='store_true', default=False,
                        help='Whether to use image augmentations for training')
    parser.add_argument('--use_prefetcher', action='store_true', default=False,
                        help='Whether to use prefetcher for faster data loading')
    parser.add_argument('--drop_last', action='store_true', default=False,
                        help='Whether to drop the last batch in the val data loading')

    # method parameters
    parser.add_argument('--method', '-m', default='HSTRNet', type=str,
                        choices=[
                            'RelTR', 'HSTRNet', 'EGTR', 'FlowSG',
                            'Motifs', 'VCTree', 'TDE', 'IMP', 'Transformer',
                            'GPS_Net', 'PE_NET', 'SQUAT', 'SHA_GCL', 'REACT',
                            'CVC',
                        ],
                        help='Name of SGG method to train (default: "HSTRNet")')
    parser.add_argument('--config_file', '-c', default=None, type=str,
                        help='Path to the default config file')
    parser.add_argument('--model_type', default=None, type=str,
                        help='Name of model for SimVP (default: None)')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate(default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate for SimVP (default: 0.)')
    # Training parameters (optimizer)
    parser.add_argument('--epoch', '-e', default=None, type=int, help='end epochs (default: 200)')
    parser.add_argument('--log_step', default=1, type=int, help='Log interval by step')
    parser.add_argument('--opt', default=None, type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam")')
    parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: None, use opt default)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer sgd momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=None, type=float, help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip_mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--no_display_method_info', action='store_true', default=False,
                        help='Do not display method info')

    # Training parameters (scheduler)
    parser.add_argument('--sched', default=None, type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "onecycle"')
    parser.add_argument('--lr', default=None, type=float, help='Learning rate (default: 1e-3)')
    parser.add_argument('--lr_k_decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--final_div_factor', type=float, default=1e4,
                        help='min_lr = initial_lr/final_div_factor for onecycle scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_epoch', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--filter_bias_and_bn', type=bool, default=False,
                        help='Whether to set the weight decay of bias and bn to 0')

    # lightning
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--metric_for_bestckpt', default='val_loss', type=str)
    parser.add_argument('--ckpt_path', default=None, type=str)

    # CLIP hierarchical alignment parameters
    parser.add_argument('--use_alignment', action='store_true', default=False,
                        help='Whether to use CLIP hierarchical semantic alignment')
    parser.add_argument('--prototype_path', default='data/VisualGenome/clip_prototypes.pth', type=str,
                        help='Path to precomputed CLIP prototypes')
    parser.add_argument('--clip_model', default='ViT-B-32', type=str,
                        help='OpenCLIP model name for prototype construction')
    parser.add_argument('--clip_dim', default=512, type=int,
                        help='CLIP text embedding dimension')
    parser.add_argument('--num_hierarchy_levels', default=3, type=int,
                        help='Number of hierarchy cut levels for alignment')
    parser.add_argument('--hierarchy_weights', nargs='+', default=[0.2, 0.3, 0.5], type=float,
                        help='Weights for each hierarchy level (coarse to fine)')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature for InfoNCE alignment softmax')
    parser.add_argument('--align_loss_coef', default=0.2, type=float,
                        help='Weight coefficient for alignment loss')

    return parser


def default_parser():
    default_values = {
        # Set-up parameters
        'device': 'cuda',
        'dist': False,
        'output_dir': './outputs',
        'ex_name': 'Debug',
        'fp16': False,
        'torchscript': False,
        'seed': 42,
        'fps': False,
        'test': False,
        'deterministic': False,
        # dataset parameters
        'num_workers': 2,
        'data_root': './data',
        'dataname': 'mmnist',
        'use_augment': False,
        'use_prefetcher': False,
        'drop_last': False,
        # method parameters
        'method': 'RelTR',
        'config_file': None,
        'model_type': 'gSTA',
        'drop': 0,
        'drop_path': 0,
        'overwrite': False,
        # Training parameters (optimizer)
        'epoch': 200,
        'log_step': 1,
        'opt': 'adam',
        'opt_eps': None,
        'opt_betas': None,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'clip_grad': None,
        'clip_mode': 'norm',
        'no_display_method_info': False,
        # Training parameters (scheduler)
        'sched': 'onecycle',
        'lr': 1e-3,
        'lr_k_decay': 1.0,
        'warmup_lr': 1e-5,
        'min_lr': 1e-6,
        'final_div_factor': 1e4,
        'warmup_epoch': 0,
        'decay_epoch': 100,
        'decay_rate': 0.1,
        'filter_bias_and_bn': False,
        'accumulate_grad_batches': 1,
        # Lightning parameters
        'gpus': [2,3,4,5],
        'metric_for_bestckpt': 'val_loss'
    }
    return default_values
