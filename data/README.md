# Data Directory

## Structure Convention

```
data/
├── dataloaders/           # Data loading code (PyTorch DataLoader wrappers)
│   ├── base_data.py       # LightningDataModule base
│   ├── dataloader.py      # Generic dataloader utilities
│   ├── coco.py            # COCO dataset support
│   ├── dataloader_VisualGenome.py
│   ├── dataset_constant.py
│   └── utils.py
├── <DatasetName>/         # Raw dataset files (one folder per dataset)
│   ├── train.json         # Training annotations
│   ├── val.json           # Validation annotations
│   ├── test.json          # Test annotations
│   ├── rel.json           # Relationship metadata
│   └── images/            # Image files
└── README.md
```

## Adding a New Dataset

1. Create a new folder: `data/<DatasetName>/`
2. Place annotation JSON files and images inside
3. Add a corresponding dataloader in `data/dataloaders/dataloader_<DatasetName>.py`
4. Register the dataset name in `data/dataloaders/dataloader.py` or `src/exp.py`

## Current Datasets

| Dataset | Folder | Dataloader |
|---|---|---|
| VisualGenome | `data/VisualGenome/` | `dataloaders/dataloader_VisualGenome.py` |
| COCO | (pending) | `dataloaders/coco.py` |

Note: The dataset files in `VisualGenome/` are git-ignored. Download them separately.
