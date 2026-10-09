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
    import warnings
    from types import SimpleNamespace
    warnings.filterwarnings('ignore')

    # --- Framework imports ---
    import torch

    from src.config_contract import merge_exclude_keys, metrics_for_eval_mode
    from src.exp import BaseExperiment
    from src.method_registry import BY_CLI_NAME
    from utils.main_utils import get_dist_info, load_config, update_config
    from utils.parser import create_parser, default_parser

    args = create_parser().parse_args()
    config = args.__dict__

    if args.config_file is None:
        config_basename = BY_CLI_NAME[args.method].config_stem
        cfg_path = osp.join(".", "configs", args.dataname, config_basename + ".py")
    else:
        cfg_path = args.config_file
    if args.overwrite:
        config = update_config(config, load_config(cfg_path),
                               exclude_keys=merge_exclude_keys(overwrite=True))
    else:
        loaded_cfg = load_config(cfg_path)
        config = update_config(config, loaded_cfg,
                               exclude_keys=merge_exclude_keys(overwrite=False))
        default_values = default_parser()
        for attribute in default_values.keys():
            if config[attribute] is None:
                config[attribute] = default_values[attribute]
    args = SimpleNamespace(**config)

    # Re-derive metrics from the final eval_mode (which may differ from the
    # config file after CLI override).
    if hasattr(args, 'eval_mode'):
        args.metrics = metrics_for_eval_mode(args.eval_mode)

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









        
