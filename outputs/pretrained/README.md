# Checkpoints

Pre-trained model weights for evaluation and fine-tuning.

## Directory Layout

```
checkpoints/
├── reltr/
│   └── reltr_vg.pth              # RelTR checkpoint (VisualGenome)
├── egtr/
│   └── egtr__.../                 # EGTR PyTorch Lightning checkpoint dir
└── README.md
```

## Download Links

| Model | Dataset | Path | Source |
|---|---|---|---|
| RelTR | VisualGenome | `reltr/reltr_vg.pth` | Official RelTR release |
| EGTR | VisualGenome | `egtr/` | See `submodules/EGTR/` |

## Adding a New Checkpoint

1. Create a subdirectory: `checkpoints/<model_name>/`
2. Place `.pth`, `.pt`, or `.ckpt` files inside
3. Update this README with download link/source
