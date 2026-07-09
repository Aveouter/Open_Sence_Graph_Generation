import os

dataset_parameters = {
    'VisualGenome': {
        'eval': True,
        'distributed': False,
        'data_root': './data/VisualGenome_sample/'
                     if os.environ.get('CI_SMOKE_TEST') == '1'
                     else './data/VisualGenome/',
        'in_shape': (1, 3, 224, 224),     # (T, C, H, W) 单帧输入
        'pre_seq_length': 1,
        'aft_seq_length': 1,
        'metrics': [
                        # SGDet: full end-to-end triplet generation
                        "sgdet_R@10", "sgdet_R@20", "sgdet_R@50",
                        "sgdet_mR@10", "sgdet_mR@20", "sgdet_mR@50",
                        # PredCLS: predicate only, given GT boxes + labels
                        "predcls_R@10", "predcls_R@20",
                        "predcls_mR@10", "predcls_mR@20",
                        # SGCLS: predicate + label, given GT boxes
                        "sgcls_R@10", "sgcls_R@20",
                        "sgcls_mR@10", "sgcls_mR@20",
                    ],
        # 151 = 150 real classes + 1 background index (for model constructor)
        # 50  = number of actual predicates (excl. background, for evaluation)
        'rel_nums': 51,
        'entity_nums': 151,
    },

    'OpenImageV6': {
        'data_root': './data/OpenImage',
        'in_shape': (1, 3, 224, 224),     # (T, C, H, W) 单帧输入
        'pre_seq_length': 1,
        'aft_seq_length': 1,
        'metrics': ['predcls_mR@20','predcls_mR@50','sgcls_R@50','sgdet_R@100'],
        'rel_nums': 10,
        'entity_nums': 57,
    },

    'OpenImage': {
        'data_root': './data/OpenImage',
        'in_shape': (1, 3, 224, 224),     # (T, C, H, W) 单帧输入
        'pre_seq_length': 1,
        'aft_seq_length': 1,
        'metrics': ['predcls_mR@20','predcls_mR@50','sgcls_R@50','sgdet_R@100'],
        'rel_nums': 10,
        'entity_nums': 57,
    },
}