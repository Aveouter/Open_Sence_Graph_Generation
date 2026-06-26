# Configuration System

Python module-style configs loaded via `utils.config_utils.load_config()`.
Used by `train.py` with the `--config_file` argument.

## Convention

```
configs/<DatasetName>/<ModelName>.py
```

```
configs/
└── VisualGenome/
    ├── HSTRNet.py
    └── RelTR.py          # Template for new configs
```

## Adding a New Config

1. Copy `configs/_template.py` → `configs/<Dataset>/<ModelName>.py`
2. Modify parameters as needed
3. Run: `python train.py -d <DatasetName> -m <ModelName> -c configs/<Dataset>/<ModelName>.py`
