# FREQ Standard Metric Validation

Command:

```bash
python - <<'PY'
from utils.parser import create_parser, default_parser
from utils.main_utils import load_config, update_config, get_dataset
from src.methods import method_maps
from src.core.metrics import metric
import torch

cfg = create_parser().parse_args([]).__dict__
loaded = load_config('configs/VisualGenome/FREQ.py')
cfg = update_config(cfg, loaded, exclude_keys=['method'])
for k, v in default_parser().items():
    if cfg.get(k) is None:
        cfg[k] = v
cfg.update({'method': 'freq', 'dataname': 'VisualGenome', 'device': 'cpu',
            'dataset_size': 2, 'val_dataset_size': 2, 'test_dataset_size': 2,
            'batch_size': 1, 'val_batch_size': 1, 'num_workers': 0})
_, val_loader, _ = get_dataset('VisualGenome', cfg)
method = method_maps['freq'](steps_per_epoch=1, save_dir='outputs/reproduction/freq_metric_slice', **cfg)
images, targets = next(iter(val_loader))
targets = method._move_targets_to_device(targets)
with torch.no_grad():
    outputs = method.forward(images, targets)['outputs']
res, log = metric(pred=outputs, true=targets,
                  metrics=['predcls_R@20', 'predcls_R@50', 'predcls_R@100',
                           'predcls_mR@20', 'predcls_mR@50', 'predcls_mR@100'],
                  rel_nums=cfg['rel_nums'], entity_nums=cfg['entity_nums'])
print(res)
print(log)
PY
```

Result on the tiny slice:

- `predcls_R@20`: 0.0
- `predcls_R@50`: 0.0
- `predcls_R@100`: 0.0
- `predcls_mR@20`: 0.0
- `predcls_mR@50`: 0.0
- `predcls_mR@100`: 0.0

Interpretation:

- This validates that FREQ outputs are accepted by the standard project metric
  path.
- It does not establish paper-scale accuracy or paper-number alignment.
