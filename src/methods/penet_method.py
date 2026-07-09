"""
PE-NET method wrapper for PyTorch Lightning.

Fully aligned with official VL-Group/PENET pipeline:
  Image → ResNeXt-101-32×8d → FPN(P2-P6) → {BoxExtractor, UnionExtractor}
    → PrototypeEmbeddingNetwork → predicate logits + proto losses

Weight loading (matching official pretraining chain):
  1. Backbone  ← torchvision ImageNet
  2. FPN + Box head ← official detector checkpoint (if provided), else kaiming_init
  3. PENet model ← trained from scratch (kaiming_init)
"""

from __future__ import annotations


import torch
import torch.nn.functional as F

from .motifs_method import Motifs_Method, MotifsCriterion
from src.models.penet_detector import PENetSGDetProposalGenerator
from src.models.backbone import (
    PENetBoxFeatureExtractor,
    PENetUnionFeatureExtractor,
    ResNetBackbone,
    FPNNeck,
)
from src.models.penet import build_penet
from utils.penet_weights import load_all_pretrained


class PENetCriterion(MotifsCriterion):
    """PE-NET criterion with prototype loss terms."""

    def forward(self, outputs: dict, targets: list) -> dict:
        loss_dict = super().forward(outputs, targets)
        loss_dict.pop("loss_total", None)
        add_losses = outputs.get("add_losses", {})
        if isinstance(add_losses, dict):
            for name, value in add_losses.items():
                if isinstance(value, torch.Tensor):
                    loss_dict[f"proto_{name}"] = value
        loss_dict["loss_total"] = sum(loss_dict.values())
        return loss_dict


class PENet_Method(Motifs_Method):
    """PE-NET Lightning method — full official pipeline with pretrained weights."""

    def __init__(self, **args):
        super().__init__(**args)

        use_backbone = args.get("use_backbone", True)
        if not use_backbone:
            return

        arch = args.get("backbone_arch", "resnext101_32x8d")
        frozen = args.get("backbone_frozen", True)
        roi_size = args.get("roi_output_size", 7)
        detector_ckpt = args.get("penet_detector_ckpt", None)

        # 1) Backbone (random init first, then load pretrained below)
        self._backbone = ResNetBackbone(arch=arch, pretrained=False, frozen=frozen)

        # 2) FPN neck
        self._fpn = FPNNeck(in_channels_list=(256, 512, 1024, 2048), out_channels=256)

        # 3) Box extractor
        self._relation_box_extractor = PENetBoxFeatureExtractor(
            roi_output_size=roi_size,
            representation_size=4096,
        )
        # Backward-compatible alias used by existing PredCls/SGCls paths.
        self._box_extractor = self._relation_box_extractor

        self._detector_box_extractor = PENetBoxFeatureExtractor(
            roi_output_size=roi_size,
            representation_size=4096,
        )
        self._proposal_generator = PENetSGDetProposalGenerator(
            num_classes=args.get("entity_nums", 151),
        )

        # 4) Union extractor (always trained from scratch)
        self._union_extractor = PENetUnionFeatureExtractor(
            roi_output_size=roi_size,
            representation_size=4096,
        )

        # Load detector-side weights. Official relation predictor tensors are
        # remapped later by ``PENetContext.remap_external_state_dict`` when
        # ``ckpt_path`` is an official SGDet checkpoint.
        counts = load_all_pretrained(
            backbone=self._backbone,
            fpn=self._fpn,
            box_extractor=self._relation_box_extractor,
            arch=arch,
            detector_ckpt=detector_ckpt,
            detector_box_extractor=self._detector_box_extractor,
            proposal_generator=self._proposal_generator,
        )
        self._weight_load_counts = counts
        self._has_official_sgdet_detector = (
            detector_ckpt is not None
            and counts.get("detector_rpn_head", 0) > 0
            and counts.get("detector_box_predictor", 0) > 0
            and counts.get("detector_box_extractor_eval", 0) > 0
        )

        # ── Image normalisation adaptor ──
        # maskrcnn_benchmark: BGR pixel-mean subtraction
        #   x_mb = pixel_BGR - [102.98, 115.95, 122.77]
        #
        # OpenSGG: RGB ImageNet nomalisation
        #   x_in[RGB] = (pixel/255 - mean)/std
        #
        # Conversion: undo ImageNet → flip RGB→BGR → subtract pixel_mean
        scale = torch.tensor([57.375, 57.12, 58.395]).view(3, 1, 1)
        bias  = torch.tensor([0.55, 0.33, 0.905]).view(3, 1, 1)
        self.register_buffer("_norm_scale", scale, persistent=False)
        self.register_buffer("_norm_bias", bias, persistent=False)

    # ── External checkpoint loading ────────────────────────────────

    def _load_extractors(self, state_dict: dict) -> dict:
        """Load backbone, FPN, box-head and union-extractor weights from an
        official maskrcnn-benchmark state dict (e.g. ``model_final.pth``).

        Called automatically by exp.py before ``_adapt_state_dict`` when a
        ``.pth`` / ``.pt`` checkpoint is detected.
        """
        from typing import Dict
        from utils.penet_weights import (
            _box_head_key_map_official_to_ours,
            _box_predictor_key_map_official_to_ours,
            _relation_box_head_key_map_official_to_ours,
            _rpn_head_key_map_official_to_ours,
            _strip_module_prefix,
            _transfer_weights,
        )

        sd = _strip_module_prefix(state_dict)
        counts: Dict[str, int] = {}

        # 1) Backbone: official uses backbone.body.* → ours omits prefix
        #    Also handles stem. prefix (official: stem.conv1, ours: conv1)
        #    and BN buffers (running_mean/running_var) that are in named_buffers().
        bb_params = dict(self._backbone.named_parameters())
        bb_buffers = dict(self._backbone.named_buffers())
        bb_all = {**bb_params, **bb_buffers}
        n_bb = 0
        for k, v in sd.items():
            if k.startswith("backbone.body."):
                our_k = k[len("backbone.body."):]  # → layer1.0.conv1...
                # Official uses stem.conv1 / stem.bn1, we use conv1 / bn1
                our_k = our_k.replace("stem.", "")
                if our_k in bb_all and bb_all[our_k].shape == v.shape:
                    bb_all[our_k].data.copy_(v)
                    n_bb += 1
        counts["backbone"] = n_bb

        # 2) FPN: backbone.fpn.fpn_innerN → _fpn.inner_blocks.N-1
        fpn_map = {}
        for i in range(4):
            fpn_map[f"backbone.fpn.fpn_inner{i + 1}.weight"] = (
                f"_fpn.inner_blocks.{i}.conv.weight"
            )
            fpn_map[f"backbone.fpn.fpn_inner{i + 1}.bias"] = (
                f"_fpn.inner_blocks.{i}.conv.bias"
            )
            fpn_map[f"backbone.fpn.fpn_layer{i + 1}.weight"] = (
                f"_fpn.layer_blocks.{i}.conv.weight"
            )
            fpn_map[f"backbone.fpn.fpn_layer{i + 1}.bias"] = (
                f"_fpn.layer_blocks.{i}.conv.bias"
            )
        counts["fpn"] = _transfer_weights(
            sd, {"_fpn": self._fpn}, fpn_map, strict=False
        )

        # 3) Box extractor — MUST use relation.box_feature_extractor, NOT
        #    roi_heads.box.feature_extractor (detector).  The two are
        #    different tensors (max abs diff ≈ 0.01).
        box_map = _relation_box_head_key_map_official_to_ours()
        counts["box_extractor"] = _transfer_weights(
            sd,
            {"_box_extractor": self._relation_box_extractor},
            box_map,
            strict=False,
        )

        if hasattr(self, "_detector_box_extractor"):
            counts["detector_box_extractor"] = _transfer_weights(
                sd,
                {"_box_extractor": self._detector_box_extractor},
                _box_head_key_map_official_to_ours(),
                strict=False,
            )
        if hasattr(self, "_proposal_generator"):
            counts["detector_rpn_head"] = _transfer_weights(
                sd,
                {"": self._proposal_generator},
                _rpn_head_key_map_official_to_ours(),
                strict=False,
            )
            counts["detector_box_predictor"] = _transfer_weights(
                sd,
                {"": self._proposal_generator},
                _box_predictor_key_map_official_to_ours(),
                strict=False,
            )
            self._has_official_sgdet_detector = (
                counts.get("detector_box_extractor", 0) > 0
                and counts.get("detector_rpn_head", 0) > 0
                and counts.get("detector_box_predictor", 0) > 0
            )

        # 4) Union feature extractor
        ufe_pfx = "roi_heads.relation.union_feature_extractor"
        P = "_union_extractor"
        union_map = {
            f"{ufe_pfx}.feature_extractor.fc6.weight":  f"{P}.fc6.weight",
            f"{ufe_pfx}.feature_extractor.fc6.bias":    f"{P}.fc6.bias",
            f"{ufe_pfx}.feature_extractor.fc7.weight":  f"{P}.fc7.weight",
            f"{ufe_pfx}.feature_extractor.fc7.bias":    f"{P}.fc7.bias",
            f"{ufe_pfx}.feature_extractor.pooler.reduce_channel.0.weight":
                f"{P}.pooler.reduce_channel.0.weight",
            f"{ufe_pfx}.feature_extractor.pooler.reduce_channel.0.bias":
                f"{P}.pooler.reduce_channel.0.bias",
        }
        # rect_conv: same Sequential indices but our index 1,3 are ReLU,BN
        for j in (0, 2, 4, 6):
            for suffix in ("weight", "bias"):
                union_map[f"{ufe_pfx}.rect_conv.{j}.{suffix}"] = f"{P}.rect_conv.{j}.{suffix}"
            if j in (2, 6):
                for stat in ("running_mean", "running_var", "num_batches_tracked"):
                    union_map[f"{ufe_pfx}.rect_conv.{j}.{stat}"] = f"{P}.rect_conv.{j}.{stat}"
        counts["union_extractor"] = _transfer_weights(
            sd, {"_union_extractor": self._union_extractor}, union_map, strict=False
        )

        # Also copy BN buffers missed by _transfer_weights (named_buffers).
        ue_bufs = dict(self._union_extractor.named_buffers())
        n_buf = 0
        for src_key, dst_key in union_map.items():
            # Strip _union_extractor. prefix from dst_key for buffer lookup
            local_key = dst_key[len(P) + 1:] if dst_key.startswith(P + ".") else dst_key
            if local_key in ue_bufs and src_key in sd and ue_bufs[local_key].shape == sd[src_key].shape:
                ue_bufs[local_key].data.copy_(sd[src_key])
                n_buf += 1
        counts["union_extractor"] += n_buf

        total = sum(counts.values())
        if total > 0:
            print(
                "[weights] Loaded extractors from official ckpt: "
                + ", ".join(f"{k}={v}" for k, v in counts.items() if v > 0)
            )
        return counts

    def _build_model(self, **args):
        # Built locally first; official relation predictor tensors are loaded
        # by the experiment checkpoint adapter during evaluation.
        return build_penet(self.hparams)

    def _build_criterion(self, **args):
        return PENetCriterion(num_predicates=args.get("rel_nums", 51))

    # ── Feature extraction ─────────────────────────────────────────

    def _to_official_padded_image(self, img, size, size_divisible=32):
        """Convert OpenSGG RGB-normalized image to official padded input.

        Official maskrcnn_benchmark applies BGR255 mean subtraction before
        ImageList padding; padded pixels are therefore zeros in that official
        normalized space.
        """
        device = img.device
        if size is None:
            size = torch.as_tensor(img.shape[-2:], dtype=torch.float32, device=device)
        elif torch.is_tensor(size):
            size = size.to(device)
        else:
            size = torch.as_tensor(size, dtype=torch.float32, device=device)

        h, w = int(size[0].item()), int(size[1].item())
        cropped = img[..., :h, :w]
        official = cropped.flip(0) * self._norm_scale + self._norm_bias

        if size_divisible > 0:
            padded_h = ((h + size_divisible - 1) // size_divisible) * size_divisible
            padded_w = ((w + size_divisible - 1) // size_divisible) * size_divisible
            pad_h = padded_h - h
            pad_w = padded_w - w
            if pad_h > 0 or pad_w > 0:
                official = F.pad(official, (0, pad_w, 0, pad_h), value=0.0)
        return official, size

    def _extract_features(self, images, boxes_list, labels_list, image_sizes):
        """Per-image: backbone → FPN → box features + FPN tuple."""

        if not hasattr(self, "_backbone") or self._backbone is None:
            return [
                {
                    "roi_feats": self._visual_extractor(lab.to(self.device)),
                    "fpn_features": None,
                }
                for lab in labels_list
            ]

        images = self._image_list_from_batch(images)
        if len(images) != len(boxes_list):
            raise ValueError(
                f"PENet: {len(images)} images vs {len(boxes_list)} targets"
            )
        device = next(self._backbone.parameters()).device
        images = [img.to(device) for img in images]
        box_dev = [b.to(device) for b in boxes_list]
        sz_dev = []
        official_images = []
        for img, size in zip(images, image_sizes):
            official_img, size = self._to_official_padded_image(img, size)
            sz_dev.append(size)
            official_images.append(official_img)

        results = []
        for img, boxes, sz in zip(official_images, box_dev, sz_dev):
            boxes = boxes.to(device)
            sz = sz.to(device)

            with torch.set_grad_enabled(
                not all(not p.requires_grad for p in self._backbone.parameters())
            ):
                raw = self._backbone(img, return_all_scales=True)
                # raw = {4: C2, 8: C3, 16: C4, 32: C5}

            # FPN: (C2, C3, C4, C5) → (P2, P3, P4, P5, P6)
            fpn_in = (raw[4], raw[8], raw[16], raw[32])
            fpn_features = self._fpn(fpn_in)  # tuple of 5 tensors

            # Box features via FPN pooler
            roi_feats = self._relation_box_extractor(fpn_features, boxes, sz)

            results.append(
                {
                    "roi_feats": roi_feats,
                    "fpn_features": fpn_features,
                }
            )

        return results

    def _compute_union_features(self, fpn_features, boxes, pairs, img_size):
        if fpn_features is None or pairs.numel() == 0:
            return None
        return self._union_extractor(fpn_features, boxes, pairs, img_size)

    def _detect_sgdet_features(self, images, image_sizes):
        """Run the official-checkpoint detector path for SGDet proposals."""
        if not getattr(self, "_has_official_sgdet_detector", False):
            raise ValueError(
                "PENet SGDet requires official detector RPN, detector box "
                "extractor and detector box predictor weights. Use the "
                "official SGDet model_final.pth as --ckpt_path, or provide "
                "--penet_detector_ckpt for the frozen detector components."
            )
        images = self._image_list_from_batch(images)
        device = next(self._backbone.parameters()).device
        images = [img.to(device) for img in images]
        results = []
        for img, size in zip(images, image_sizes):
            official_img, size = self._to_official_padded_image(img, size)
            with torch.no_grad():
                raw = self._backbone(official_img, return_all_scales=True)
                fpn_features = self._fpn((raw[4], raw[8], raw[16], raw[32]))
                proposals = self._proposal_generator(
                    fpn_features,
                    self._detector_box_extractor,
                    size,
                )
                roi_feats = self._relation_box_extractor(
                    fpn_features,
                    proposals.boxes,
                    size,
                )
            results.append(
                {
                    "roi_feats": roi_feats,
                    "boxes": proposals.boxes,
                    "labels": proposals.labels,
                    "obj_dists": proposals.obj_dists,
                    "boxes_per_cls": proposals.boxes_per_cls,
                    "fpn_features": fpn_features,
                    "image_size": size,
                }
            )
        return results

    def _extra_model_kwargs(self, target, boxes, labels, return_obj_preds):
        extra = super()._extra_model_kwargs(target, boxes, labels, return_obj_preds)
        if "rel_annotations" in target:
            extra["rel_annotations"] = target["rel_annotations"]
        for key in ("union_feats", "edge_visual_feats"):
            if key in target:
                extra["precomputed_union_feats"] = target[key]
                break
        if "boxes_per_cls" in target:
            extra["boxes_per_cls"] = target["boxes_per_cls"]
        for key in ("obj_dists", "scores_all"):
            if key in target:
                extra["obj_dists"] = target[key]
                break
        return extra

    # ── Forward ──────────────────────────────────────────────────

    def forward(self, images, targets=None, **kwargs):
        is_training = self.training
        eval_mode = getattr(self.hparams, "eval_mode", "predcls")
        return_obj_preds = getattr(self.hparams, "eval_mode", "predcls") in ("sgcls", "sgdet")

        if is_training or targets is not None:
            all_outputs = []
            if targets is None:
                targets = [{}] * (
                    len(images) if isinstance(images, list) else images.size(0)
                )

            boxes_list = [t["boxes"] for t in targets]
            labels_list = [t["labels"] for t in targets]
            image_sizes = [t.get("size", t.get("orig_size")) for t in targets]

            if eval_mode == "sgdet" and not self.training:
                vis_results = self._detect_sgdet_features(images, image_sizes)
                boxes_list = [r["boxes"] for r in vis_results]
                labels_list = [r["labels"] for r in vis_results]
            else:
                vis_results = self._extract_features(
                    images, boxes_list, labels_list, image_sizes
                )

            for i, (box, lab, sz) in enumerate(
                zip(boxes_list, labels_list, image_sizes)
            ):
                r = vis_results[i]
                roi_feats, fpn_feats = r["roi_feats"], r["fpn_features"]

                extra = self._extra_model_kwargs(targets[i], box, lab, return_obj_preds)
                if eval_mode == "sgdet" and not self.training:
                    extra["obj_dists"] = r["obj_dists"]
                    extra["boxes_per_cls"] = r["boxes_per_cls"]
                    sz = r["image_size"]
                if eval_mode == "sgdet":
                    missing = [
                        key
                        for key in ("boxes_per_cls", "obj_dists")
                        if key not in extra
                    ]
                    if missing:
                        raise ValueError(
                            "PENet SGDet requires official detector proposal "
                            f"fields {missing}; otherwise the run is a "
                            "protocol_mismatch, not an official SGDet eval."
                        )

                # Pass FPN features + extraction callback to model
                extra["_compute_union_fn"] = self._compute_union_features
                extra["_fpn_features"] = fpn_feats
                extra["_image_size"] = sz

                out = self.model(
                    roi_feats, box, lab, return_obj_preds=return_obj_preds, **extra
                )
                all_outputs.append(out)

            batched = {
                "model_family": "penet_sgdet" if eval_mode == "sgdet" else "motifs",
                "rel_logits": [o["rel_logits"] for o in all_outputs],
                "pair_indices": [o["pair_indices"] for o in all_outputs],
                "sub_boxes": [o["sub_boxes"] for o in all_outputs],
                "obj_boxes": [o["obj_boxes"] for o in all_outputs],
                "obj_labels": [o["obj_labels"] for o in all_outputs],
                "predicate_bg_index": all_outputs[0].get("predicate_bg_index"),
                "relation_softmax_scope": all_outputs[0].get(
                    "relation_softmax_scope", "all"
                ),
            }
            if return_obj_preds:
                batched["obj_logits"] = [o.get("obj_logits") for o in all_outputs]
            sgdet_keys = (
                "sgdet_rel_scores",
                "sgdet_sub_boxes",
                "sgdet_obj_boxes",
                "sgdet_sub_scores",
                "sgdet_obj_scores",
                "sgdet_sub_classes",
                "sgdet_obj_classes",
            )
            if eval_mode == "sgdet":
                for key in sgdet_keys:
                    if all(key in o for o in all_outputs):
                        batched[key] = [o[key] for o in all_outputs]
                if all("sgdet_box_space" in o for o in all_outputs):
                    batched["sgdet_box_space"] = [o["sgdet_box_space"] for o in all_outputs]

            add_losses = {}
            for o in all_outputs:
                for name, value in o.get("add_losses", {}).items():
                    add_losses.setdefault(name, []).append(value)
            if add_losses:
                batched["add_losses"] = {
                    name: torch.stack(values).mean()
                    if all(torch.is_tensor(v) for v in values)
                    else values
                    for name, values in add_losses.items()
                }

            out = {"outputs": batched}
            if is_training and self.criterion is not None:
                loss_dict = self.criterion(batched, targets)
                out["loss_dict"] = loss_dict
                out["loss"] = loss_dict["loss_total"]
            return out

        return {"outputs": {}}
