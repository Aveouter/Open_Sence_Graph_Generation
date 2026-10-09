# Copyright (c) CIGIT HPC Lab. All rights reserved

from importlib import import_module

# Parsing a method must not import the unrelated Lightning runtime. Preserve
# every public export, loading its owning module only when requested.
_EXPORTS = {
    "config_utils": ("Config",),
    "main_utils": ("print_log", "output_namespace", "collect_env", "check_dir",
                   "get_dataset", "measure_throughput", "load_config", "update_config", "get_dist_info"),
    "parser": ("create_parser", "default_parser"),
    "callbacks": ("SetupCallback", "EpochEndCallback", "BestCheckpointCallback"),
    "path_utils": ("resolve_output_paths", "normalize_ex_name", "ensure_unique_run_dir",
                   "collect_metadata", "write_metadata", "save_eval_results"),
}


def __getattr__(name):
    for module, names in _EXPORTS.items():
        if name in names:
            value = getattr(import_module(f".{module}", __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'Config', 'create_parser', 'default_parser',
    'print_log', 'output_namespace', 'collect_env', 'check_dir',
    'get_dataset', 'measure_throughput', 'load_config', 'update_config',
    'get_dist_info',
    'SetupCallback', 'EpochEndCallback', 'BestCheckpointCallback',
    'resolve_output_paths', 'normalize_ex_name', 'ensure_unique_run_dir',
    'collect_metadata', 'write_metadata', 'save_eval_results',
]

