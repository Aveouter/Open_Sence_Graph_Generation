# Copyright (c) Team of XiaoguangLin, CIGIT HPC Lab. All rights reserved
# This source code is licensed under the MIT license that can be found in the LICENSE file.
# Author: Xinyu Liu, Xiaoguang Lin

import os as _os

# 减少 CUDA 显存碎片化，必须放在任何 cuda 操作之前
_os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

if __name__ == '__main__':
    import os.path as osp
    from types import SimpleNamespace
    import warnings
    warnings.filterwarnings('ignore')

    # --- Framework imports ---
    from utils.main_utils import get_dist_info, load_config, update_config
    from utils.misc import is_main_process
    from src.exp import BaseExperiment
    from utils.parser import create_parser, default_parser

    import torch
    import gc

    args = create_parser().parse_args()
    config = args.__dict__

    cfg_path = osp.join(".", "configs", args.dataname , args.method+".py") \
        if args.config_file is None else args.config_file
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









        
