# ssgoos/evaluation/orchestrator.py
"""
Evaluation orchestrator for SGG tasks.

Extracted from Base_method._run_epoch_eval() to provide a reusable,
DDP-safe evaluation pipeline. No algorithm logic is changed — this is
purely an engineering reorganization.
"""

import os
import shutil
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist


def gather_epoch_outputs_to_rank0(
    outputs: List[Any],
    save_dir: str,
    stage: str,
    epoch: int,
) -> Optional[List[Any]]:
    """
    Gather per-rank epoch outputs to rank 0 using filesystem-based sharing.

    This avoids OOM from dist.all_gather_object() by using temporary files
    that each rank writes independently. Rank 0 reads all shards and merges.

    Args:
        outputs: List of per-batch outputs accumulated by this rank.
        save_dir: Base directory for temporary cache files.
        stage: 'val' or 'test'.
        epoch: Current epoch number.

    Returns:
        Merged outputs on rank 0, or None on other ranks.
    """
    if not dist.is_available() or not dist.is_initialized():
        return outputs

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size == 1:
        return outputs

    # Create temp directory for this stage
    tmp_dir = os.path.join(save_dir, '.ddp_eval_cache', f'{stage}_epoch_{epoch}')
    if rank == 0:
        os.makedirs(tmp_dir, exist_ok=True)

    dist.barrier()

    # Each rank writes its outputs to a file
    rank_file = os.path.join(tmp_dir, f'rank_{rank}.pt')
    torch.save(outputs, rank_file)

    dist.barrier()

    if rank == 0:
        # Read all shards and merge
        all_outputs = []
        for r in range(world_size):
            rf = os.path.join(tmp_dir, f'rank_{r}.pt')
            all_outputs.append(torch.load(rf, map_location='cpu'))
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # Flatten: each rank's outputs is a list of batch outputs
        merged = []
        for rank_outputs in all_outputs:
            merged.extend(rank_outputs)
        return merged

    dist.barrier()
    return None


def merge_pred_outputs(outputs: List[Dict]) -> Dict[str, Any]:
    """
    Merge per-batch prediction outputs into a single dict.

    Tensors are concatenated along dim 0; numpy arrays are
    concatenated along axis 0.

    Args:
        outputs: List of per-batch output dicts.

    Returns:
        Merged output dict.
    """
    if not outputs:
        return {}

    merged = {}
    example = outputs[0]

    for key in example:
        values = [o[key] for o in outputs if key in o]
        if not values:
            continue

        if torch.is_tensor(values[0]):
            merged[key] = torch.cat([v.cpu() for v in values], dim=0)
        elif hasattr(values[0], 'numpy'):
            import numpy as np
            merged[key] = np.concatenate([v for v in values], axis=0)
        else:
            merged[key] = [v for v in values]

    return merged


def merge_targets(targets: List[Dict]) -> List[Dict]:
    """
    Merge per-batch targets into a flat list.

    Args:
        targets: List of per-batch target lists.

    Returns:
        Flat list of target dicts.
    """
    merged = []
    for batch_targets in targets:
        if isinstance(batch_targets, list):
            merged.extend(batch_targets)
        elif isinstance(batch_targets, dict):
            merged.append(batch_targets)
    return merged


def average_losses(losses: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Average per-batch losses across all batches.

    Args:
        losses: List of per-batch loss dicts.

    Returns:
        Dict of averaged loss values.
    """
    if not losses:
        return {}

    result = {}
    for key in losses[0]:
        result[key] = sum(l[key] for l in losses if key in l) / len(losses)
    return result
