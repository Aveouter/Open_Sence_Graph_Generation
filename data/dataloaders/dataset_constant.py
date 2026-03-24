from typing import Dict, Tuple, List
from pathlib import Path
import re
import pandas as pd

dataset_parameters = {
    'ThyroTriples': {
        'in_shape': (16, 3, 64, 64),      # (T, C, H, W)
        # 'out_shape' : (3, 8, 64, 64),
        'pre_seq_length': 16,             # 输入：过去16帧
        'aft_seq_length': 1,              # 输出：预测一个场景图
        'metrics': ['predcls_mR@20','predcls_mR@50','sgcls_R@50','sgdet_R@100'],
    },

    'VisualGenome': {
        'eval': True,
        'distributed': False,
        'data_root': './data/VisualGenome/',
        'in_shape': (1, 3, 224, 224),     # (T, C, H, W) 单帧输入
        'pre_seq_length': 1,
        'aft_seq_length': 1,
        'metrics': ['predcls_mR@20','predcls_mR@50','sgcls_R@50','sgdet_R@100'],
        'rel_nums': 111,
        'entity_nums':111,
    },

    'OpenImage': {
        'data_root': './data/OpenImage',
        'in_shape': (1, 3, 224, 224),     # (T, C, H, W) 单帧输入
        'pre_seq_length': 1,
        'aft_seq_length': 1,
        'metrics': ['predcls_mR@20','predcls_mR@50','sgcls_R@50','sgdet_R@100'],
        'rel_nums': 111,
        'entity_nums':111,
 
    },
}