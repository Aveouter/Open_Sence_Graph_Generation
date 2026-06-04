# ssgoos/evaluation/metric_adapters.py
"""
Metric adapter: bridges model outputs to the SceneGraphEvaluator.

This is a thin wrapper over the existing src/core/metrics.py module,
providing a clean interface for the task layer.
The underlying evaluation logic is completely unchanged.
"""

from typing import Any, Dict, List, Optional


def metric(
    predictions: Dict[str, Any],
    targets: List[Dict],
    metric_names: List[str],
    rel_nums: int = 51,
    entity_nums: int = 151,
    triplet_indices: Optional[List] = None,
) -> Dict[str, float]:
    """
    Compute scene graph evaluation metrics.

    Delegates to the existing metric() function in src/core/metrics.py.
    No algorithm logic is changed.

    Args:
        predictions: Merged model output dict.
        targets: Flat list of ground-truth target dicts.
        metric_names: List of metric specifiers, e.g. ['sgdet_R@50', 'sgdet_mR@20'].
        rel_nums: Number of predicate classes (including background).
        entity_nums: Number of entity classes (including background).
        triplet_indices: Optional Hungarian matching indices for predcls/sgcls.

    Returns:
        Dict mapping metric name to float value.
    """
    # Delegate to existing implementation (unchanged logic)
    from src.core.metrics import metric as _original_metric
    return _original_metric(
        predictions, targets, metric_names,
        rel_nums=rel_nums,
        entity_nums=entity_nums,
        triplet_indices=triplet_indices,
    )
