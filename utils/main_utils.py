# Copyright (c) CIGIT HPC Lab. All rights reserved
import traceback
import cv2
import os
import logging
import subprocess
import sys
from collections import defaultdict, OrderedDict
from typing import Tuple

import torch
import torchvision
from torch import distributed as dist

from .config_utils import Config

def collect_env():
    """Collect the information of the running environments."""
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        from torch.utils.cpp_extension import CUDA_HOME
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
            try:
                nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc.exe' if os.name == 'nt' else 'nvcc')
                nvcc = subprocess.check_output([nvcc, '-V'], stderr=subprocess.STDOUT)
                nvcc = nvcc.decode('utf-8', errors='ignore').strip().splitlines()[-1]
            except Exception:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

        devices = defaultdict(list)
        try:
            for k in range(torch.cuda.device_count()):
                devices[torch.cuda.get_device_name(k)].append(str(k))
            for name, devids in devices.items():
                env_info['GPU ' + ','.join(devids)] = name
        except RuntimeError as exc:
            env_info['CUDA device query error'] = str(exc)

    try:
        gcc = subprocess.check_output(['gcc', '--version'], stderr=subprocess.STDOUT)
        gcc = gcc.decode('utf-8', errors='ignore').strip().splitlines()[0]
    except Exception:
        gcc = 'Not Available'
    env_info['GCC'] = gcc

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = torch.__config__.show()
    env_info['TorchVision'] = torchvision.__version__
    env_info['OpenCV'] = cv2.__version__

    return env_info

def print_log(message):
    print(message)
    logging.info(message)


def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    return path


def get_dataset(dataname, config):
    from data.dataloaders.dataset_constant import dataset_parameters
    from data.dataloaders.dataloader import load_data
    # from data.dataloaders.dataloader_weather import load_data
    # Preserve model-specific keys (set by config file) from being
    # overwritten by dataset-wide defaults.
    saved = {k: config[k] for k in ('entity_nums', 'rel_nums', 'metrics', 'distributed') if k in config}
    config.update(dataset_parameters[dataname])
    config.update(saved)  # config file values take priority over dataset defaults
    return load_data(**config)


def measure_throughput(model, input_dummy):

    def get_batch_size(H, W):
        max_side = max(H, W)
        if max_side >= 128:
            bs = 10
            repetitions = 1000
        else:
            bs = 100
            repetitions = 100
        return bs, repetitions

    if isinstance(input_dummy, tuple):
        input_dummy = list(input_dummy)
        _, T, C, H, W = input_dummy[0].shape
        bs, repetitions = get_batch_size(H, W)
        _input = torch.rand(bs, T, C, H, W).to(input_dummy[0].device)
        input_dummy[0] = _input
        input_dummy = tuple(input_dummy)
    else:
        _, T, C, H, W = input_dummy.shape
        bs, repetitions = get_batch_size(H, W)
        input_dummy = torch.rand(bs, T, C, H, W).to(input_dummy.device)
    total_time = 0
    with torch.no_grad():
        for _ in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            if isinstance(input_dummy, tuple):
                _ = model(*input_dummy)
            else:
                _ = model(input_dummy)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * bs) / total_time
    return Throughput


def load_config(filename:str = None, allow_missing: bool = False):
    """load and print config.

    Args:
        filename: path to config .py file.
        allow_missing: if True, return empty dict on missing file (for special
            scripts).  The main training entry (train.py) should NOT use this
            so that a missing config fails immediately rather than running with
            unintended defaults.
    """
    print('loading config from ' + filename + ' ...')
    try:
        configfile = Config(filename=filename)
        config = configfile._cfg_dict
    except (FileNotFoundError, IOError):
        if allow_missing:
            config = dict()
            print('warning: fail to load the config!')
        else:
            raise FileNotFoundError(
                f'Config file not found: {filename}. '
                f'Check that the method name matches a config file under '
                f'configs/<dataset>/, or use --config_file to specify a path.'
            ) from None
    return config

def update_config(args, config, exclude_keys=None):
    """Update the args dict with a new config dict.

    CLI args (``args``) take priority: a config key is only applied when
    the corresponding CLI arg is missing (not in args) or explicitly None.
    Keys in ``exclude_keys`` are never overwritten by the config file.
    """
    if exclude_keys is None:
        exclude_keys = []
    assert isinstance(args, dict) and isinstance(config, dict)
    for k in config.keys():
        if k in exclude_keys:
            continue  # keep command-line value, don't touch
        if k in args and args[k] is not None:
            if args[k] != config[k]:
                print(f'overwrite config key -- {k}: {config[k]} -> {args[k]}')
        else:
            args[k] = config[k]
    return args


def weights_to_cpu(state_dict: OrderedDict) -> OrderedDict:
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(  # type: ignore
        state_dict, '_metadata', OrderedDict())
    return state_dict_cpu


def get_dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size
