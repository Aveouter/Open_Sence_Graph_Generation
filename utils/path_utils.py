"""
Central path resolution and run lifecycle utilities.

All output paths are resolved here — single source of truth shared by
src/exp.py, standalone scripts, and the migration tool.
"""

import json
import os
import os.path as osp
import socket
import subprocess
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_output_paths(args, run_dir_override: Optional[str] = None) -> dict:
    """Compute all output paths from args.

    Parameters
    ----------
    args : object
        Must expose ``method``, ``ex_name``, and ``output_dir`` attributes.
    run_dir_override : str or None
        When provided, use this path directly instead of constructing it.

    Returns
    -------
    dict with keys: root, run_dir, ckpt_dir, eval_dir, log_file,
                    config_dir, args_file, config_file, resolved_config_file,
                    lightning_dir, profiler_dir, pretrained_dir
    """
    method = args.method.lower()
    root = getattr(args, 'output_dir', None) or './outputs'

    if run_dir_override is not None:
        run_dir = run_dir_override
    else:
        run_dir = osp.join(root, 'runs', method, args.ex_name)

    return {
        'root':                root,
        'run_dir':             run_dir,
        'ckpt_dir':            osp.join(run_dir, 'checkpoints'),
        'eval_dir':            osp.join(run_dir, 'eval'),
        'log_file':            osp.join(run_dir, 'run.log'),
        'config_dir':          osp.join(run_dir, 'config'),
        'args_file':           osp.join(run_dir, 'config', 'args.json'),
        'config_file':         osp.join(run_dir, 'config', 'config.json'),
        'resolved_config_file': osp.join(run_dir, 'config', 'resolved_config.yaml'),
        'lightning_dir':       osp.join(run_dir, 'lightning'),
        'profiler_dir':        osp.join(run_dir, 'profiler'),
        'pretrained_dir':      osp.join(root, 'pretrained'),
    }


# ---------------------------------------------------------------------------
# Experiment name helpers
# ---------------------------------------------------------------------------

def normalize_ex_name(raw_name: str) -> str:
    """Ensure the experiment name starts with a ``YYYY-MM-DD_`` prefix.

    >>> normalize_ex_name('test')
    '2026-06-09_test'
    >>> normalize_ex_name('2026-06-01_baseline')
    '2026-06-01_baseline'
    """
    # Already has date prefix?  (len=11 covers "YYYY-MM-DD_")
    if len(raw_name) >= 11 and raw_name[4] == '-' and raw_name[7] == '-' and raw_name[10] == '_':
        # Quick sanity: first 10 chars look like a date
        try:
            datetime.strptime(raw_name[:10], '%Y-%m-%d')
            return raw_name
        except ValueError:
            pass

    today = datetime.now().strftime('%Y-%m-%d')
    return f'{today}_{raw_name}'


def _extract_ex_name_from_run_dir(run_dir: str) -> str:
    """Extract the experiment name (last path component) from a run directory."""
    return osp.basename(run_dir.rstrip('/'))


def ensure_unique_run_dir(run_dir: str, overwrite: bool = False) -> Tuple[str, str]:
    """Guarantee a unique, non-colliding run directory.

    Parameters
    ----------
    run_dir : str
        Desired run directory path.
    overwrite : bool
        If True, return *run_dir* even if it already exists.

    Returns
    -------
    (actual_run_dir, actual_ex_name) : Tuple[str, str]
        The guaranteed-unique directory path and the corresponding experiment
        name extracted from it.
    """
    if overwrite:
        ex_name = _extract_ex_name_from_run_dir(run_dir)
        return run_dir, ex_name

    if not osp.exists(run_dir):
        ex_name = _extract_ex_name_from_run_dir(run_dir)
        return run_dir, ex_name

    # Collision — append numeric suffix
    base_dir = osp.dirname(run_dir)
    base_name = osp.basename(run_dir)
    counter = 1
    while True:
        candidate_name = f'{base_name}_{counter:03d}'
        candidate = osp.join(base_dir, candidate_name)
        if not osp.exists(candidate):
            return candidate, candidate_name
        counter += 1


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _get_git_info() -> Tuple[str, bool]:
    """Return (commit_hash, is_dirty) for the current repository."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode('utf-8').strip()
    except Exception:
        commit = 'unknown'

    try:
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL,
        ).decode('utf-8').strip()
        dirty = len(status) > 0
    except Exception:
        dirty = False

    return commit, dirty


def collect_metadata(args, run_dir: str, start_time: str) -> dict:
    """Gather run identity information for ``metadata.json``.

    Parameters
    ----------
    args : object
        CLI / config namespace.
    run_dir : str
        Final run directory (after uniqueness resolution).
    start_time : str
        ISO-format start timestamp.
    """
    git_commit, git_dirty = _get_git_info()

    # GPU info — guard against driver / init errors
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    gpu_count = 0
    gpu_names = []
    if cuda_ok:
        try:
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                try:
                    gpu_names.append(torch.cuda.get_device_name(i))
                except Exception:
                    gpu_names.append(f'device_{i}')
        except Exception:
            gpu_count = -1
            gpu_names = ['<error retrieving GPU info>']

    return {
        'method':           getattr(args, 'method', 'unknown').lower(),
        'ex_name':          _extract_ex_name_from_run_dir(run_dir),
        'run_dir':          run_dir,
        'created_at':       start_time,
        'command':          ' '.join(sys.argv),
        'git_commit':       git_commit,
        'git_dirty':        git_dirty,
        'hostname':         socket.gethostname(),
        'python_version':   sys.version.split()[0],
        'torch_version':    torch.__version__,
        'cuda_available':   cuda_ok,
        'gpu_count':        gpu_count,
        'gpu_names':        gpu_names,
        'seed':             getattr(args, 'seed', None),
    }


def write_metadata(run_dir: str, metadata: dict) -> None:
    """Persist ``metadata.json`` into *run_dir*."""
    os.makedirs(run_dir, exist_ok=True)
    path = osp.join(run_dir, 'metadata.json')
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------

_VALID_EVAL_MODES = {'predcls', 'sgcls', 'sgdet'}


def save_eval_results(eval_dir: str, eval_mode: str, metrics: dict,
                      predictions=None) -> None:
    """Save per-mode evaluation results and synchronise the summary.

    Parameters
    ----------
    eval_dir : str
        Parent ``eval/`` directory for the run.
    eval_mode : str
        One of ``predcls``, ``sgcls``, ``sgdet``.
    metrics : dict
        Metric name → float value.
    predictions : optional
        Arbitrary prediction object to serialise via :func:`torch.save`.
    """
    if eval_mode not in _VALID_EVAL_MODES:
        raise ValueError(
            f'eval_mode must be one of {sorted(_VALID_EVAL_MODES)}, got {eval_mode!r}'
        )

    mode_dir = osp.join(eval_dir, eval_mode)
    os.makedirs(mode_dir, exist_ok=True)

    metrics_file = osp.join(mode_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=float)

    if predictions is not None:
        pred_file = osp.join(mode_dir, 'predictions.pt')
        torch.save(predictions, pred_file)

    _update_eval_summary(eval_dir)


def _update_eval_summary(eval_dir: str) -> None:
    """Aggregate all existing ``<mode>/metrics.json`` into ``summary.json``."""
    summary = {}
    for mode in sorted(_VALID_EVAL_MODES):
        metrics_file = osp.join(eval_dir, mode, 'metrics.json')
        if osp.isfile(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    summary[mode] = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

    if summary:
        os.makedirs(eval_dir, exist_ok=True)
        summary_file = osp.join(eval_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
