# sgg/utils/main_utils.py
"""
Re-export shim for project-level utilities.

Delegates to utils/main_utils.py for backward compatibility.
"""

from utils.main_utils import (
    check_dir,
    collect_env,
    print_log,
    output_namespace,
    get_dataset,
    load_config,
    update_config,
    get_dist_info,
    measure_throughput,
    weights_to_cpu,
)

__all__ = [
    'check_dir', 'collect_env', 'print_log', 'output_namespace',
    'get_dataset', 'load_config', 'update_config', 'get_dist_info',
    'measure_throughput', 'weights_to_cpu',
]
