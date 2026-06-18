#!/usr/bin/env python
"""Inspect SGB/Kaihua Motifs checkpoints against the OpenSGG implementation.

The downloaded public Motifs checkpoints are full maskrcnn-benchmark models.
This utility remaps ``roi_heads.relation.predictor.*`` keys to OpenSGG's
SGB-compatible Motifs relation head and reports exact shape matches.

Example:
    python tools/checkpoints/inspect_motifs_checkpoint.py \
      --checkpoint outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.motifs import build_motifs


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "module"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return _strip_module_prefix(ckpt[key])
        if all(torch.is_tensor(v) for v in ckpt.values()):
            return _strip_module_prefix(ckpt)
    raise TypeError(f"Could not find a state_dict in {path}")


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        (name[7:] if name.startswith("module.") else name): tensor
        for name, tensor in state.items()
    }


def _build_default_motifs(preset: str):
    if preset == "sgb-vg":
        args = SimpleNamespace(
            entity_nums=151,
            rel_nums=51,
            # Build the checkpoint-native relation head: SGB produces 4096D
            # ROI and union features before MotifPredictor.
            visual_dim=4096,
            motifs_obj_feat_dim=4096,
            hidden_dim=512,
            pooling_dim=4096,
            embed_dim=200,
            obj_lstm_layers=1,
            edge_lstm_layers=1,
            dropout=0.2,
            motifs_order="leftright",
            use_freq_bias=True,
            freq_bias_eps=1e-3,
            use_vision=True,
            use_tanh=False,
            motifs_pos_embed_dim=128,
            motifs_pos_batchnorm=True,
            motifs_obj_feat_to_edge=True,
            motifs_effect_analysis=True,
            motifs_include_bg_predicate=True,
            motifs_predicate_bg_index="first",
            data_root=None,
        )
    else:
        args = SimpleNamespace(
            entity_nums=151,
            rel_nums=51,
            visual_dim=2048,
            hidden_dim=512,
            pooling_dim=4096,
            motifs_obj_feat_dim=4096,
            embed_dim=200,
            obj_lstm_layers=1,
            edge_lstm_layers=1,
            dropout=0.2,
            motifs_order="leftright",
            use_freq_bias=True,
            freq_bias_eps=1e-3,
            use_vision=True,
            use_tanh=False,
            motifs_pos_embed_dim=128,
            motifs_pos_batchnorm=True,
            motifs_obj_feat_to_edge=True,
            motifs_effect_analysis=True,
            motifs_include_bg_predicate=True,
            motifs_predicate_bg_index="first",
            data_root=None,
        )
    return build_motifs(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--preset",
        choices=("sgb-vg", "opensgg"),
        default="sgb-vg",
        help="Model shape to compare against.",
    )
    parser.add_argument("--show-missing", action="store_true")
    parser.add_argument("--show-unexpected", action="store_true")
    args = parser.parse_args()

    ckpt_state = _load_state_dict(args.checkpoint)
    model = _build_default_motifs(args.preset)
    remapped_state = model.remap_external_state_dict(ckpt_state)
    model_state = model.state_dict()

    exact = []
    shape_mismatch = []
    unexpected = []
    for name, tensor in remapped_state.items():
        if name not in model_state:
            unexpected.append(name)
            continue
        if tuple(tensor.shape) == tuple(model_state[name].shape):
            exact.append(name)
        else:
            shape_mismatch.append((name, tuple(tensor.shape), tuple(model_state[name].shape)))

    missing = sorted(set(model_state) - set(remapped_state))
    prefixes = Counter(name.split(".", 1)[0] for name in ckpt_state)

    print(f"checkpoint: {args.checkpoint}")
    print(f"preset:     {args.preset}")
    print(f"checkpoint keys: {len(ckpt_state)}")
    print(f"remapped keys:   {len(remapped_state)}")
    print(f"model keys:      {len(model_state)}")
    print(f"exact matches:   {len(exact)}")
    print(f"shape mismatch:  {len(shape_mismatch)}")
    print(f"unexpected:      {len(unexpected)}")
    print(f"missing:         {len(missing)}")
    print("")
    print("checkpoint key prefixes:")
    for prefix, count in prefixes.most_common():
        print(f"  {prefix}: {count}")

    if shape_mismatch:
        print("")
        print("shape mismatches:")
        for name, ckpt_shape, model_shape in shape_mismatch[:50]:
            print(f"  {name}: ckpt={ckpt_shape}, model={model_shape}")
        if len(shape_mismatch) > 50:
            print(f"  ... {len(shape_mismatch) - 50} more")

    if args.show_unexpected and unexpected:
        print("")
        print("unexpected keys:")
        for name in unexpected:
            print(f"  {name}")

    if args.show_missing and missing:
        print("")
        print("missing model keys:")
        for name in missing:
            print(f"  {name}")


if __name__ == "__main__":
    main()
