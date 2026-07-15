from functools import partial
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset
import utils.misc as utils
from data.dataloaders import build_dataset, get_coco_api_from_dataset

# Fixed max image size for val/test collation: [C, H, W] = [3, 1344, 1344]
# Using a fixed size ensures all batches have identical tensor shapes,
# eliminating CUDA allocator fragmentation from per-batch variable sizes.
# 1344 covers the val transform's max_size=1333 plus rounding slack.
_VAL_FIXED_MAX_SIZE = [3, 1344, 1344]

def load_data(args=None, **kwargs):
    if args is None:
        args = SimpleNamespace(**kwargs)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # Optional small subsets for smoke tests. In eval mode dataset_val points to
    # the test split, so test_dataset_size/val_dataset_size controls test size.
    train_size = getattr(args, 'dataset_size', None)
    eval_size = getattr(args, 'test_dataset_size', None)
    if eval_size is None:
        eval_size = getattr(args, 'val_dataset_size', None)

    train_start = int(getattr(args, 'dataset_start_index', 0) or 0)
    eval_start = getattr(args, 'test_dataset_start_index', None)
    if eval_start is None:
        eval_start = getattr(args, 'val_dataset_start_index', 0)
    eval_start = int(eval_start or 0)

    if train_size is not None:
        train_start = min(max(train_start, 0), len(dataset_train))
        train_n = min(int(train_size), max(len(dataset_train) - train_start, 0))
        dataset_train = Subset(dataset_train, list(range(train_start, train_start + train_n)))
    if eval_size is not None:
        eval_start = min(max(eval_start, 0), len(dataset_val))
        val_n = min(int(eval_size), max(len(dataset_val) - eval_start, 0))
        dataset_val = Subset(dataset_val, list(range(eval_start, eval_start + val_n)))

    if args.distributed:
        # Lightning injects DistributedSampler after process-group
        # initialization. Creating it here fails in the parent process because
        # torch.distributed is not initialized yet.
        sampler_train = None
        sampler_val = None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # NOTE: Do NOT pass an explicit batch_sampler to DataLoader.
    # Lightning's _dataloader_init_kwargs_resolve_sampler() has a bug (as of 2.6.1)
    # where the batch_sampler code path hardcodes batch_size=1/drop_last=False in
    # the reconstructed kwargs, corrupting DDP training. Using batch_size+sampler
    # instead lets PyTorch auto-create the BatchSampler internally, which makes
    # Lightning use the safe sampler-only code path.
    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        drop_last=True,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    if sampler_train is None:
        train_loader_kwargs["shuffle"] = True
    else:
        train_loader_kwargs["sampler"] = sampler_train
    data_loader_train = DataLoader(dataset_train, **train_loader_kwargs)

    val_batch_size = getattr(args, 'val_batch_size', None) or args.batch_size
    val_collate = partial(utils.collate_fn, fixed_max_size=_VAL_FIXED_MAX_SIZE)

    val_loader_kwargs = dict(
        batch_size=val_batch_size,
        drop_last=False,
        collate_fn=val_collate,
        num_workers=args.num_workers,
    )
    if sampler_val is None:
        val_loader_kwargs["shuffle"] = False
    else:
        val_loader_kwargs["sampler"] = sampler_val
    data_loader_val = DataLoader(dataset_val, **val_loader_kwargs)

    # 可选：保存评估用对象，挂到 dataset 上而非 loader
    base_ds = get_coco_api_from_dataset(dataset_val)
    data_loader_val.dataset.base_ds = base_ds

    return data_loader_train, data_loader_val, data_loader_val
