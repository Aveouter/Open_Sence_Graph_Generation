# Copyright (c) Team of XiaoguangLin, CIGIT HPC Lab. All rights reserved
# This source code is licensed under the MIT license that can be found in the LICENSE file.
# Author: Xinyu Liu, Xiaoguang Lin

import os.path as osp
import warnings
warnings.filterwarnings('ignore')

from src.exp import BaseExperiment
from utils import (create_parser, default_parser, get_dist_info, load_config,
                           update_config)
import torch
import gc

if __name__ == '__main__':
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
    
    print('>'*35 + ' training ' + '<'*35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    if rank == 0:
        print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()
    
    
    
