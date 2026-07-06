# Copyright (c) Team of XiaoguangLin, CIGIT HPC Lab. All rights reserved
# This source code is licensed under the MIT license that can be found in the LICENSE file.
# Author: Xinyu Liu, Xiaoguang Lin

import os as _os

# 减少 CUDA 显存碎片化 (PyTorch >= 2.1 only)
try:
    import torch
    if hasattr(torch, '__version__'):
        ver = tuple(int(x) for x in torch.__version__.split('.')[:2])
        if ver >= (2, 1):
            _os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
except (ImportError, ValueError, TypeError, AttributeError):
    pass

if __name__ == '__main__':
    import os.path as osp
    from types import SimpleNamespace
    import warnings
    warnings.filterwarnings('ignore')

    # --- Framework imports ---
    from utils.main_utils import get_dist_info, load_config, update_config
    from src.exp import BaseExperiment
    from utils.parser import create_parser, default_parser

    import torch

    # Config file name alias map — some CLI method names differ from config
    # file basenames (e.g. CLI "GPSNet" → config "GPS_Net.py").
    _CONFIG_FILE_ALIASES = {
        'gpsnet': 'GPS_Net',
        'penet': 'PE_NET',
        'shagcl': 'SHA_GCL',
        # accept CLI names with underscores too
        'gps_net': 'GPS_Net',
        'pe_net': 'PE_NET',
        'sha_gcl': 'SHA_GCL',
    }

    args = create_parser().parse_args()
    config = args.__dict__

    if args.config_file is None:
        method_key = args.method.lower()
        config_basename = _CONFIG_FILE_ALIASES.get(method_key, args.method)
        cfg_path = osp.join(".", "configs", args.dataname, config_basename + ".py")
    else:
        cfg_path = args.config_file
    if args.overwrite:
        config = update_config(config, load_config(cfg_path),
                               exclude_keys=['method'])
    else:
        loaded_cfg = load_config(cfg_path)
        config = update_config(config, loaded_cfg,
                               exclude_keys=['method', 'val_batch_size',
                                             'drop_path', 'warmup_epoch'])
        default_values = default_parser()
        for attribute in default_values.keys():
            if config[attribute] is None:
                config[attribute] = default_values[attribute]
    args = SimpleNamespace(**config)

    # Re-derive metrics from the final eval_mode (which may differ from the
    # config file after CLI override).
    if hasattr(args, 'eval_mode'):
        ks = [20, 50, 100]
        args.metrics = [f"{args.eval_mode}_R@{k}" for k in ks] + \
                       [f"{args.eval_mode}_mR@{k}" for k in ks]

    print('>'*35 + ' training ' + '<'*35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()

    if args.test:
        if rank == 0:
            print('>' * 35 + ' testing ' + '<' * 35)
            print(f'[Info] ckpt_path: {args.ckpt_path}')
        result = exp.test()
    else:
        if rank == 0:
            print('>' * 35 + ' training ' + '<' * 35)
            if args.ckpt_path is not None:
                print(f'[Info] resume / finetune from ckpt: {args.ckpt_path}')
        exp.train()









        
