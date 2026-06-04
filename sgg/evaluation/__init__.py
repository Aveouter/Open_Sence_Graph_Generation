# sgg/evaluation/__init__.py
from sgg.evaluation.orchestrator import (
    gather_epoch_outputs_to_rank0,
    merge_pred_outputs,
    merge_targets,
    average_losses,
)
from sgg.evaluation.metric_adapters import metric
