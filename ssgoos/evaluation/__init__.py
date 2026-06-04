# ssgoos/evaluation/__init__.py
from ssgoos.evaluation.orchestrator import (
    gather_epoch_outputs_to_rank0,
    merge_pred_outputs,
    merge_targets,
    average_losses,
)
from ssgoos.evaluation.metric_adapters import metric
