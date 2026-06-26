# Adding A New Model

This guide describes the standard path for adding a new SGG method to OpenSGG.
Use it when a model should be available through the main training entry point:

```bash
python train.py --method <MethodName> --dataname VisualGenome
```

## 1. Choose The Method Name

Use one canonical name and keep it consistent across:

- CLI argument in `utils/parser.py`
- Config file name in `configs/<Dataset>/<MethodName>.py`
- Config field `method = '<MethodName>'`
- Registry key in `src/methods/__init__.py`
- README and experiment commands

The current `train.py` config loader builds the default config path as:

```python
configs/<dataname>/<method>.py
```

If the CLI name and config file name differ, pass `--config_file` explicitly or
add a deliberate alias layer.

## 2. Add The Model Implementation

Place the core model in `src/models/`.

Recommended layout:

```text
src/models/my_model.py
```

Minimum expectation:

- The module exposes either a model class or a `build(args)` helper.
- `forward(...)` returns tensors needed by the loss and evaluator.
- Tensor shapes are documented in short comments or docstrings where they are
  not obvious.
- Dataset label conventions are explicit, especially background class
  placement for objects and predicates.

Example skeleton:

```python
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, num_classes: int, num_rel_classes: int, hidden_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes
        self.hidden_dim = hidden_dim

    def forward(self, samples):
        return {
            "pred_logits": None,
            "pred_boxes": None,
            "rel_logits": None,
        }


def build(args):
    model = MyModel(
        num_classes=args.num_classes,
        num_rel_classes=args.num_rel_classes,
        hidden_dim=args.hidden_dim,
    )
    criterion = None
    postprocessors = {}
    return model, criterion, postprocessors
```

Use existing files such as `src/models/reltr.py` and `src/modules/egtr/` as
references for end-to-end methods.

## 3. Add The Lightning Method Wrapper

Create a wrapper in `src/methods/`. This wrapper adapts the model to the common
training, validation, and testing flow.

```text
src/methods/my_model_method.py
```

For most models, subclass `Base_method`:

```python
import torch

from .base_method import Base_method
from src.models.my_model import build as build_my_model


class MyModel_Method(Base_method):
    def _build_model(self, **args):
        model, criterion, postprocessors = build_my_model(self.hparams)
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.weight_dict = getattr(criterion, "weight_dict", {})
        return model

    def _compute_losses(self, outputs, targets):
        loss_dict = self.criterion(outputs, targets)
        total_loss = sum(
            loss_dict[k] * self.weight_dict.get(k, 1.0)
            for k in loss_dict
            if torch.is_tensor(loss_dict[k])
        )
        return loss_dict, total_loss

    def training_step(self, batch, batch_idx):
        images, targets = self._split_batch(batch)
        samples = self._to_nested_tensor(images)
        targets = self._move_targets_to_device(targets)
        outputs = self.model(samples)
        loss_dict, total_loss = self._compute_losses(outputs, targets)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss
```

Override these helpers when needed:

| Helper | When to override |
|--------|------------------|
| `_to_nested_tensor` | The model expects `NestedTensor` or another image container. |
| `_move_targets_to_device` | Targets contain nested tensors or method-specific fields. |
| `_eval_step` | Evaluation must cache matcher indices, postprocessed outputs, or extra metadata. |
| `forward` | External scripts need a stable inference API. |

## 4. Register The Method

Edit `src/methods/__init__.py`:

```python
from .my_model_method import MyModel_Method

method_maps = {
    # ...
    "mymodel": MyModel_Method,
}
```

Keep the key normalized and make sure it matches how `BaseExperiment` looks up
methods. If the rest of the repo uses lowercase keys, register the lowercase
form and normalize the CLI value before lookup.

## 5. Add The Config

Create:

```text
configs/VisualGenome/MyModel.py
```

Start from `configs/_template.py` or a nearby method config. Include at least:

```python
method = "MyModel"
loss = "my_model_loss"

lr = 1e-4
weight_decay = 1e-4
epoch = 20
batch_size = 4
val_batch_size = 4

dataset = "VisualGenome"
num_classes = 151
num_rel_classes = 51
entity_nums = 151
rel_nums = 51

device = "cuda"
num_workers = 4
seed = 42
```

For quick smoke runs, prefer CLI overrides instead of committing tiny
debug-only configs:

```bash
python train.py --method MyModel --dataname VisualGenome \
  --dataset_size 8 --val_dataset_size 8 --epoch 1 --num_workers 0
```

## 6. Expose The CLI Name

If the method should be selectable without `--config_file`, add it to the
choices in `utils/parser.py`:

```python
parser.add_argument(
    "--method",
    choices=[
        # ...
        "MyModel",
    ],
)
```

Then verify that the resulting default config path exists:

```text
configs/VisualGenome/MyModel.py
```

## 7. Connect Losses And Metrics

If the model uses an existing loss, set the config `loss` field accordingly.
If it needs a new loss:

1. Add the criterion implementation under `src/losses/` or `src/loss.py`,
   following the local pattern.
2. Register the loss name in the loss construction path.
3. Return a `loss_dict` whose keys are stable and meaningful.

For evaluation, check `src/core/metrics.py`. Add method-specific metadata only
when the output convention differs from the default, for example:

- predicate background index is first instead of last
- relation logits need a restricted softmax scope
- boxes are normalized differently
- evaluation needs matcher indices or postprocessed triplets

## 8. Add A Smoke Test

At minimum, run:

```bash
python -m compileall -q train.py src data utils tools configs
python tools/ci_validate.py
```

Then run a small train/eval check:

```bash
python train.py --method MyModel --dataname VisualGenome \
  --dataset_size 8 \
  --val_dataset_size 8 \
  --epoch 1 \
  --batch_size 2 \
  --val_batch_size 1 \
  --num_workers 0
```

If a checkpoint is available, run:

```bash
python train.py --test --method MyModel --dataname VisualGenome \
  --ckpt_path path/to/checkpoint.ckpt \
  --test_dataset_size 20 \
  --val_batch_size 1 \
  --num_workers 0
```

## 9. Update Documentation

Update the supported-method table in `README.md` with:

- Paper or project link
- Venue, if applicable
- Method type
- CLI name
- One-line description

For reproduction results, update `guides/published_method_reproduction.md` only
for methods from published papers, and only when the metric can be traced to a
checkpoint, command, dataset split, and metric artifact.

## PR Checklist

- [ ] Model code is in `src/models/`.
- [ ] Lightning wrapper is in `src/methods/`.
- [ ] Method is registered in `src/methods/__init__.py`.
- [ ] CLI choice and config file naming are consistent.
- [ ] Config exists under `configs/<Dataset>/`.
- [ ] Loss construction works.
- [ ] Evaluation output convention is documented.
- [ ] Smoke train/eval command succeeds.
- [ ] README and published-method reproduction report are updated when applicable.
