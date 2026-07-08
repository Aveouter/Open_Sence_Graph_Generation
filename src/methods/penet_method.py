"""
PE-NET method wrapper for PyTorch Lightning.

Fully aligned with official VL-Group/PENET pipeline:
  Image → ResNeXt-101-32×8d → FPN(P2-P6) → {BoxExtractor, UnionExtractor}
    → PrototypeEmbeddingNetwork → predicate logits + proto losses

Weight loading for checkpoint-backed evaluation:
  1. Backbone / FPN / Box head ← official detector checkpoint when provided
  2. PrototypeEmbeddingNetwork ← official PE-NET checkpoint via --ckpt_path
"""

from __future__ import annotations


import torch

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
        self._box_extractor = PENetBoxFeatureExtractor(
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

        # ── Load pretrained backbone + FPN + box-head (detector) weights ──
        # The relation predictor is loaded later from --ckpt_path by BaseExperiment.
        counts = load_all_pretrained(
            backbone=self._backbone,
            fpn=self._fpn,
            box_extractor=self._box_extractor,
            arch=arch,
            detector_ckpt=detector_ckpt,
            proposal_generator=self._proposal_generator,
        )
        self._weight_load_counts = counts
        self._has_official_sgdet_detector = (
            detector_ckpt is not None
            and counts.get("detector_rpn_head", 0) > 0
            and counts.get("detector_box_predictor", 0) > 0
        )

    def _build_model(self, **args):
        return build_penet(self.hparams)

    def _build_criterion(self, **args):
        return PENetCriterion(num_predicates=args.get("rel_nums", 51))

    # ── Feature extraction ─────────────────────────────────────────

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
        cropped = []
        for img, size in zip(images, image_sizes):
            if size is None:
                size = torch.as_tensor(
                    img.shape[-2:], dtype=torch.float32, device=device
                )
            elif torch.is_tensor(size):
                size = size.to(device)
            else:
                size = torch.as_tensor(size, dtype=torch.float32, device=device)
            h, w = int(size[0].item()), int(size[1].item())
            sz_dev.append(size)
            cropped.append(img[..., :h, :w])

        results = []
        for img, boxes, sz in zip(cropped, box_dev, sz_dev):
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
            roi_feats = self._box_extractor(fpn_features, boxes, sz)

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
                "PENet SGDet requires the official pretrained Faster R-CNN "
                "detector checkpoint with RPN and box predictor weights. "
                "Provide --penet_detector_ckpt; otherwise the run is a "
                "protocol_mismatch, not an official SGDet evaluation."
            )
        images = self._image_list_from_batch(images)
        device = next(self._backbone.parameters()).device
        images = [img.to(device) for img in images]
        results = []
        for img, size in zip(images, image_sizes):
            if size is None:
                size = torch.as_tensor(img.shape[-2:], dtype=torch.float32, device=device)
            elif torch.is_tensor(size):
                size = size.to(device)
            else:
                size = torch.as_tensor(size, dtype=torch.float32, device=device)
            h, w = int(size[0].item()), int(size[1].item())
            cropped = img[..., :h, :w]
            with torch.no_grad():
                raw = self._backbone(cropped, return_all_scales=True)
                fpn_features = self._fpn((raw[4], raw[8], raw[16], raw[32]))
                proposals = self._proposal_generator(
                    fpn_features,
                    self._box_extractor,
                    size,
                )
            results.append(
                {
                    "roi_feats": proposals.roi_feats,
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
        return_obj_preds = eval_mode in {"sgcls", "sgdet"}

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
                            f"fields {missing}. Provide boxes_per_cls and "
                            "predict_logits/scores_all from the detector; "
                            "otherwise the run is a protocol_mismatch, not an "
                            "official SGDet evaluation."
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
