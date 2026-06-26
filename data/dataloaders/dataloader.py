# Copyright (c) CIGIT HPC Lab. All rights reserved
""" 通用数据加载路由 """
from __future__ import annotations
import inspect
from types import SimpleNamespace
from typing import Any, Dict

def _safe_call(fn, prefer: Dict[str, Any], extra: Dict[str, Any], *, verbose: bool = False):
    """按底层函数签名做白名单过滤并去重调用。

    prefer 中的 None 值不会覆盖 extra 中的有效值，
    确保用户通过 kwargs 传入的参数优先于框架默认的 None 值。
    """
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())

    # prefer 中值为 None 的 key 不参与覆盖
    prefer_keys = {k for k, v in prefer.items() if v is not None}
    extra_keys = set(extra.keys())

    conflict = sorted(list(prefer_keys & extra_keys))
    invalid = sorted(list(extra_keys - allowed))

    if verbose:
        if conflict:
            print(f"[safe_call] prefer keys override extra: {conflict}")
        if invalid:
            print(f"[safe_call] drop invalid keys (not in signature): {invalid}")

    filtered_extra = {k: v for k, v in extra.items() if k in allowed and k not in prefer_keys}
    filtered_prefer = {k: v for k, v in prefer.items() if k in allowed and v is not None}

    merged = {**filtered_extra, **filtered_prefer}
    return fn(**merged)

def load_data(
    dataname: str,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    data_root: str,
    dist: bool = False,
    **kwargs,
):
    """路由到具体数据集的 load_data 实现。"""
    # —— 统一读取公共/默认配置 ——
    cfg_dataloader = dict(
        pre_seq_length=kwargs.get('pre_seq_length', 10),
        aft_seq_length=kwargs.get('aft_seq_length', 10),
        in_shape=kwargs.get('in_shape', None),
        distributed=dist,
        use_augment=kwargs.get('use_augment', False),
        use_prefetcher=kwargs.get('use_prefetcher', False),
        drop_last=kwargs.get('drop_last', False),
    )

    if 'VisualGenome' in dataname or 'OpenImageV6' in dataname:
        from .dataloader_VisualGenome import load_data as load_VG
        dataname = 'vg' if 'VisualGenome' in dataname else ('oi' if 'OpenImageV6' in dataname else dataname)
        merged = dict(
            dataset=dataname,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            data_root=data_root,

            # 其他参数
            image_size=kwargs.get('image_size', (224, 224)),
            transform=kwargs.get('transform', None),
            dataset_size=kwargs.get('dataset_size', None),
            val_dataset_size=kwargs.get('val_dataset_size', None),
            seed=kwargs.get('seed', 0),
        )

        # 把剩余 kwargs 也并进去，避免丢参
        for key, value in kwargs.items():
            if key not in merged:
                merged[key] = value

        # 封装成 args，适配 RelTR 风格 load_data(args)
        args = SimpleNamespace(**merged)

        return load_VG(args=args)
    

    else:
        raise ValueError(f'Dataname {dataname} is unsupported')