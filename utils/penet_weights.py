"""
Pretrained weight loading for PENet backbone + FPN + box-head.

Matches the official VL-Group/PENET pretraining chain:

  1. Backbone: ImageNet-pretrained ResNeXt-101-32×8d via torchvision
  2. FPN: COCO-pretrained via torchvision Faster R-CNN ResNet50 FPN
     (same architecture: 4× lateral 1×1 + 4× smoothing 3×3, all 256-dim)
  3. Box head fc6/fc7 (4096-dim): requires official pretrained detector
     checkpoint; kaiming_init when unavailable

  The PENet relation predictor is trained from scratch — NOT loaded.
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision


# =========================================================================
# Key-mapping tables
# =========================================================================

def _fpn_key_map_official_to_ours() -> Dict[str, str]:
    """Map official maskrcnn-benchmark FPN keys → our ``FPNNeck`` keys.

    Official FPN keys (from ``maskrcnn_benchmark.modeling.backbone.fpn.FPN``):
      backbone.fpn.fpn_inner1.weight  →  _fpn.inner_blocks.0.conv.weight
      backbone.fpn.fpn_inner2.weight  →  _fpn.inner_blocks.1.conv.weight
      ...
      backbone.fpn.fpn_layer1.weight  →  _fpn.layer_blocks.0.conv.weight
      ...
    """
    mapping = {}
    for i in range(4):
        mapping[f"backbone.fpn.fpn_inner{i+1}.weight"] = f"_fpn.inner_blocks.{i}.conv.weight"
        mapping[f"backbone.fpn.fpn_inner{i+1}.bias"] = f"_fpn.inner_blocks.{i}.conv.bias"
        mapping[f"backbone.fpn.fpn_layer{i+1}.weight"] = f"_fpn.layer_blocks.{i}.conv.weight"
        mapping[f"backbone.fpn.fpn_layer{i+1}.bias"] = f"_fpn.layer_blocks.{i}.conv.bias"
    return mapping


def _box_head_key_map_official_to_ours() -> Dict[str, str]:
    """Map official box-head keys → our ``PENetBoxFeatureExtractor`` keys.

    Official (maskrcnn-benchmark):
      roi_heads.box_head.fc6.weight  →  _box_extractor.fc6.weight
    """
    return {
        "roi_heads.box_head.fc6.weight": "_box_extractor.fc6.weight",
        "roi_heads.box_head.fc6.bias": "_box_extractor.fc6.bias",
        "roi_heads.box_head.fc7.weight": "_box_extractor.fc7.weight",
        "roi_heads.box_head.fc7.bias": "_box_extractor.fc7.bias",
    }




# =========================================================================
# Helpers
# =========================================================================

def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove leading ``module.`` prefix added by DataParallel / DDP."""
    stripped = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            stripped[k[7:]] = v
        else:
            stripped[k] = v
    return stripped


def _transfer_weights(
    src: Dict[str, torch.Tensor],
    dst_modules: Dict[str, nn.Module],
    key_map: Dict[str, str],
    strict: bool = False,
) -> int:
    """Copy weights from *src* to *dst_modules* using *key_map*.

    Returns the number of parameters transferred.
    """
    transferred = 0
    dst_params = {}
    for scope, mod in dst_modules.items():
        for name, param in mod.named_parameters():
            full = f"{scope}.{name}" if scope else name
            dst_params[full] = param

    for src_key, dst_key in key_map.items():
        if src_key not in src:
            if strict:
                raise KeyError(f"Missing source key: {src_key}")
            continue
        if dst_key not in dst_params:
            if strict:
                raise KeyError(f"Missing destination key: {dst_key}")
            continue

        src_tensor = src[src_key]
        dst_tensor = dst_params[dst_key]
        if src_tensor.shape != dst_tensor.shape:
            warnings.warn(
                f"Shape mismatch for {src_key} → {dst_key}: "
                f"{tuple(src_tensor.shape)} vs {tuple(dst_tensor.shape)} — skipping"
            )
            continue

        dst_tensor.data.copy_(src_tensor)
        transferred += 1

    return transferred


# =========================================================================
# Public API
# =========================================================================


def load_backbone_from_torchvision(
    backbone: nn.Module,
    arch: str = "resnet50",
) -> int:
    """Load ImageNet-pretrained backbone weights from torchvision.

    Returns number of parameters loaded.
    """
    if arch == "resnext101_32x8d":
        model = torchvision.models.resnext101_32x8d(
            weights=torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2
        )
    elif arch == "resnet50":
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
    elif arch == "resnet101":
        model = torchvision.models.resnet101(
            weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1
        )
    else:
        raise ValueError(f"Unknown backbone arch: {arch}")

    # Build key map: torchvision resnet keys → our ResNetBackbone keys
    tv_sd = model.state_dict()
    our_params = dict(backbone.named_parameters())

    transferred = 0
    for tv_key, tv_val in tv_sd.items():
        if tv_key in our_params and tv_val.shape == our_params[tv_key].shape:
            our_params[tv_key].data.copy_(tv_val)
            transferred += 1

    return transferred


def load_detector_checkpoint(
    backbone: nn.Module,
    fpn: nn.Module,
    box_extractor: nn.Module,
    checkpoint_path: str,
) -> Dict[str, int]:
    """Load pretrained detector weights from an official ``model_final.pth``.

    This covers the official ``pretrained_faster_rcnn/model_final.pth``
    which contains ResNeXt-101-32×8d backbone + FPN + box head (4096-dim).

    Returns dict with counts per component.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model" in ckpt:
        sd = ckpt["model"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    sd = _strip_module_prefix(sd)

    counts = {}

    # 1) Backbone: direct key match (official uses same torchvision resnet keys)
    counts["backbone"] = _transfer_weights(
        sd, {"": backbone}, {}, strict=False,
    )

    # 2) FPN: official naming → ours
    fpn_map = _fpn_key_map_official_to_ours()
    counts["fpn"] = _transfer_weights(sd, {"_fpn": fpn}, fpn_map, strict=False)

    # 3) Box head fc6/fc7
    box_map = _box_head_key_map_official_to_ours()
    counts["box_extractor"] = _transfer_weights(
        sd, {"_box_extractor": box_extractor}, box_map, strict=False,
    )

    return counts


def load_all_pretrained(
    backbone: nn.Module,
    fpn: nn.Module,
    box_extractor: nn.Module,
    arch: str = "resnext101_32x8d",
    detector_ckpt: Optional[str] = None,
) -> Dict[str, int]:
    """Load pretrained backbone weights.  FPN and box-head require the official
    detector checkpoint for COCO pretraining (torchvision does not provide
    ResNeXt-101-FPN detection models).

    Loading chain:
      1. torchvision ImageNet → backbone (always available)
      2. Official detector ckpt → backbone + FPN + box_extractor fc6/fc7
         (overrides step 1 for backbone; loads FPN and box-head which
          otherwise use kaiming_init)

    The PENet relation predictor is always trained from scratch.
    """
    counts: Dict[str, int] = {}

    # Step 1: ImageNet backbone — always loaded from torchvision
    n = load_backbone_from_torchvision(backbone, arch)
    counts["backbone_imagenet"] = n
    print(f"[weights] Backbone (ImageNet, {arch}): {n} params loaded")

    # Step 2: Official detector checkpoint — loads backbone + FPN + box-head
    # torchvision does NOT provide ResNeXt-101/ResNet-101/ResNet-50 FPN
    # weights with 4096-dim box-head, so without this file FPN and
    # box-head start from kaiming_init.
    if detector_ckpt is not None and os.path.isfile(detector_ckpt):
        print(f"[weights] Loading official detector checkpoint: {detector_ckpt}")
        det_counts = load_detector_checkpoint(
            backbone, fpn, box_extractor, detector_ckpt,
        )
        for k, v in det_counts.items():
            counts[f"detector_{k}"] = v
            print(f"[weights]   {k}: {v} params loaded (overrides previous)")
    elif detector_ckpt is not None:
        print(f"[weights] WARNING: detector checkpoint not found: {detector_ckpt}")
        print(f"[weights]   FPN and box-head will use kaiming_init.")

    # ── Summary ──
    have_detector = detector_ckpt is not None and os.path.isfile(detector_ckpt)
    print("[weights] ─────────────────────────────────────────────")
    print("[weights] Weight initialization summary:")
    print(f"[weights]   Backbone   ← ImageNet (torchvision{' + detector ckpt' if have_detector else ''})")
    print(f"[weights]   FPN        ← {'COCO (detector ckpt)' if have_detector else 'kaiming_init (needs detector ckpt)'}")
    print(f"[weights]   Box fc6/fc7← {'COCO (detector ckpt)' if have_detector else 'kaiming_init (needs detector ckpt)'}")
    print("[weights]   Union ext  ← kaiming_init (trained from scratch)")
    print("[weights]   PENet model← kaiming_init (trained from scratch)")
    if not have_detector:
        print("[weights] ─────────────────────────────────────────────")
        print("[weights] To load FPN + box-head COCO pretrained weights:")
        print("[weights]   1. Download pretrained_faster_rcnn/model_final.pth")
        print("[weights]      from the official PENet repo (Google Drive link in README)")
        print("[weights]   2. Set penet_detector_ckpt to the file path in PE_NET.py")
    print("[weights] ─────────────────────────────────────────────")

    return counts
