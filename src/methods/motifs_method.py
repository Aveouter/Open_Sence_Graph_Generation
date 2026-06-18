"""
Motifs method wrapper for PyTorch Lightning.

Integrates the Neural Motifs model into the OpenSGG training framework.

Supports:
  - PredCLS: GT boxes + GT labels → predicate prediction
  - SGCLS:  GT boxes → object label prediction + predicate prediction

Training batch convention:
    batch = (images, targets)
      images: Tensor[B, 3, H, W] or list[Tensor(3, H, W)]
      targets: list[dict], each contains:
        - "labels": LongTensor[num_obj] — object class labels
        - "boxes":  FloatTensor[num_obj, 4] — (cx, cy, w, h) normalized
        - "rel_annotations": LongTensor[num_rel, 3] — (sub_idx, obj_idx, rel_label)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_method import Base_method
from src.models.motifs import build_motifs
from utils.misc import NestedTensor


class MotifsCriterion(nn.Module):
    """Loss function for Motifs model.

    Computes cross-entropy loss on predicate predictions
    for the GT-annotated object pairs.
    """

    def __init__(self, num_predicates: int):
        super().__init__()
        self.num_predicates = num_predicates - 1 if num_predicates > 50 else num_predicates
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

    def _target_predicate_index(self, pred_id: int, logits_dim: int, bg_index) -> int:
        """Map VG predicate ids (1..50) to the model's predicate-logit layout."""
        if logits_dim == self.num_predicates:
            return pred_id - 1
        if logits_dim == self.num_predicates + 1:
            return pred_id if bg_index == "first" else pred_id - 1
        return -1

    def forward(self, outputs: dict, targets: list) -> dict:
        total_pred_loss = None
        total_obj_loss = None
        total_pairs = 0
        total_objs = 0

        # Determine if outputs are batched (list of per-image tensors)
        rel_logits_list = outputs.get("rel_logits")
        pair_indices_list = outputs.get("pair_indices")
        obj_logits_list = outputs.get("obj_logits")
        bg_index = outputs.get("predicate_bg_index")
        if isinstance(bg_index, (list, tuple)):
            bg_index = bg_index[0] if bg_index else None

        if not isinstance(rel_logits_list, (list, tuple)):
            rel_logits_list = [rel_logits_list]
            pair_indices_list = [pair_indices_list] if pair_indices_list is not None else None
            obj_logits_list = [obj_logits_list] if obj_logits_list is not None else None

        for i, t in enumerate(targets):
            if i >= len(rel_logits_list):
                break

            rel_logits = rel_logits_list[i]
            pair_indices = pair_indices_list[i] if pair_indices_list is not None else None

            if rel_logits is None or pair_indices is None:
                continue

            if isinstance(rel_logits, torch.Tensor) and rel_logits.dim() == 3:
                rel_logits = rel_logits.squeeze(0)
            if isinstance(pair_indices, torch.Tensor) and pair_indices.dim() == 3:
                pair_indices = pair_indices.squeeze(0)

            rel_anns = t.get("rel_annotations")
            if rel_anns is None or rel_anns.numel() == 0:
                continue

            # Build GT predicate labels for each pair. OpenSGG/VG stores
            # predicates as 1..50 with 0 reserved for background. OpenSGG's
            # original local Motifs emitted 50 pure predicate logits; the
            # SGB/Kaihua layout emits 51 logits with __no_relation__ at index 0.
            P = rel_logits.size(0)
            gt_preds = torch.full((P,), -1, dtype=torch.long, device=rel_logits.device)

            # Create lookup from (s,o) → predicate
            pair_to_pred = {}
            for ann in rel_anns:
                s, o, p = int(ann[0]), int(ann[1]), int(ann[2])
                p = self._target_predicate_index(p, rel_logits.size(-1), bg_index)
                if 0 <= p < rel_logits.size(-1):
                    pair_to_pred[(s, o)] = p

            for p_idx in range(P):
                s = int(pair_indices[p_idx, 0])
                o = int(pair_indices[p_idx, 1])
                if (s, o) in pair_to_pred:
                    gt_preds[p_idx] = pair_to_pred[(s, o)]

            # Cross-entropy on annotated pairs
            valid_mask = gt_preds >= 0
            if valid_mask.any():
                pred_loss = self.ce_loss(rel_logits[valid_mask], gt_preds[valid_mask])
                total_pred_loss = pred_loss if total_pred_loss is None else total_pred_loss + pred_loss
                total_pairs += int(valid_mask.sum().item())

            # Object loss (if predicted)
            if obj_logits_list is not None and i < len(obj_logits_list):
                obj_logits = obj_logits_list[i]
                if obj_logits is not None:
                    gt_labels = t.get("labels")
                    if gt_labels is not None:
                        if obj_logits.dim() == 2:
                            obj_loss = F.cross_entropy(obj_logits, gt_labels, reduction='sum')
                            total_obj_loss = obj_loss if total_obj_loss is None else total_obj_loss + obj_loss
                            total_objs += gt_labels.numel()

        if total_pred_loss is None:
            device = None
            if isinstance(rel_logits_list, (list, tuple)) and len(rel_logits_list) > 0 and torch.is_tensor(rel_logits_list[0]):
                device = rel_logits_list[0].device
            total_pred_loss = torch.tensor(0.0, device=device)
        if total_obj_loss is None:
            total_obj_loss = total_pred_loss.new_tensor(0.0)

        loss_dict = {
            "loss_predicate": total_pred_loss / max(total_pairs, 1),
        }
        total_loss = loss_dict["loss_predicate"]

        if total_objs > 0:
            loss_dict["loss_object"] = total_obj_loss / total_objs
            total_loss = total_loss + loss_dict["loss_object"]

        loss_dict["loss_total"] = total_loss
        return loss_dict


class Motifs_Method(Base_method):
    """Neural Motifs Lightning method.

    Training batch convention:
        batch = (images, targets)
          images: Tensor[B, 3, H, W] or list[Tensor(3, H, W)]
          targets: list[dict], each contains:
            - "labels": LongTensor[num_obj]
            - "boxes": FloatTensor[num_obj, 4] (cx, cy, w, h)
            - "rel_annotations": LongTensor[num_rel, 3]
    """

    def __init__(self, **args):
        super().__init__(**args)
        # Build visual feature extractor (backbone + ROI Align)
        use_backbone = args.get('use_backbone', True)
        if use_backbone:
            from src.models.backbone import build_visual_extractor
            self._visual_extractor = build_visual_extractor(self.hparams)
        else:
            # Fallback: per-class embedding placeholder
            visual_dim = args.get('visual_dim', 2048)
            num_classes = args.get('entity_nums', 151)
            self._visual_extractor = nn.Embedding(num_classes, visual_dim)

    def _build_criterion(self, **args):
        return MotifsCriterion(
            num_predicates=args.get('rel_nums', 51),
        )

    def _build_model(self, **args):
        return build_motifs(self.hparams)

    def _extra_model_kwargs(self, target, boxes, labels, return_obj_preds):
        extra = {}
        for key in ("obj_dists", "scores_all"):
            if key in target:
                extra["obj_dists"] = target[key]
                break
        for key in ("union_feats", "edge_visual_feats"):
            if key in target:
                extra["union_feats"] = target[key]
                break
        return extra

    @property
    def _use_backbone(self):
        return isinstance(self._visual_extractor, nn.Module) and \
               not isinstance(self._visual_extractor, nn.Embedding)

    # ---------- visual feature extraction ----------

    def _image_list_from_batch(self, images):
        if isinstance(images, NestedTensor) or (
            hasattr(images, "tensors") and torch.is_tensor(images.tensors)
        ):
            images = images.tensors

        if torch.is_tensor(images):
            if images.dim() == 3:
                return [images]
            if images.dim() == 4:
                return [img for img in images]
            raise ValueError(f"images tensor must be [C,H,W] or [B,C,H,W], got {tuple(images.shape)}")

        if isinstance(images, (list, tuple)):
            return list(images)

        raise TypeError(f"Unsupported images type for Motifs: {type(images)}")

    def _extract_visual_features(self, images, boxes_list, labels_list, image_sizes):
        """Extract per-object visual features.

        Routes to backbone+ROI or per-class embedding depending on config.
        """
        if self._use_backbone:
            # Full backbone + ROI Align pipeline
            images = self._image_list_from_batch(images)
            if len(images) != len(boxes_list):
                raise ValueError(
                    f"Motifs got {len(images)} image(s) but {len(boxes_list)} target(s)."
                )

            # Move images to device
            device = next(self._visual_extractor.parameters()).device
            images = [img.to(device) for img in images]
            box_dev = [b.to(device) for b in boxes_list]
            sz_dev = []
            cropped_images = []
            for img, size in zip(images, image_sizes):
                if size is None:
                    size = torch.as_tensor(img.shape[-2:], dtype=torch.float32, device=device)
                elif torch.is_tensor(size):
                    size = size.to(device)
                else:
                    size = torch.as_tensor(size, dtype=torch.float32, device=device)
                if size.numel() != 2:
                    raise ValueError(f"image size must have 2 elements, got {tuple(size.shape)}")
                height, width = int(size[0].item()), int(size[1].item())
                img = img[..., :height, :width]
                cropped_images.append(img)
                sz_dev.append(size)
            images = cropped_images
            return self._visual_extractor(images, box_dev, sz_dev)
        else:
            # Per-class embedding placeholder
            return [self._visual_extractor(lab.to(self.device))
                    for lab in labels_list]

    # ---------- forward ----------

    def forward(self, images, targets=None, **kwargs):
        """Forward pass for Motifs."""
        is_training = targets is not None
        return_obj_preds = getattr(self.hparams, 'eval_mode', 'predcls') == 'sgcls'

        if is_training or targets is not None:
            all_outputs = []
            if targets is None:
                targets = [{}] * (len(images) if isinstance(images, list) else images.size(0))

            boxes_list = [t["boxes"] for t in targets]
            labels_list = [t["labels"] for t in targets]
            image_sizes = [t.get("size", t.get("orig_size")) for t in targets]

            # Extract visual features
            visual_feats_list = self._extract_visual_features(
                images, boxes_list, labels_list, image_sizes)

            for i, (box, lab, vis) in enumerate(zip(boxes_list, labels_list, visual_feats_list)):
                extra_kwargs = self._extra_model_kwargs(
                    targets[i], box, lab, return_obj_preds)
                out = self.model(
                    vis, box, lab, return_obj_preds=return_obj_preds,
                    **extra_kwargs)
                all_outputs.append(out)

            # Batch outputs
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

    # ---------- training / eval steps ----------

    def training_step(self, batch, batch_idx):
        images, targets = self._split_batch(batch)
        targets = self._move_targets_to_device(targets)

        result = self.forward(images, targets)
        loss_dict = result.get("loss_dict", {})
        total_loss = result.get("loss", torch.tensor(0.0, device=self.device))

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f"train_{k}", v, on_step=True, on_epoch=False)

        return total_loss

    @torch.no_grad()
    def _eval_step(self, batch, prefix: str):
        images, targets = self._split_batch(batch)
        targets = self._move_targets_to_device(targets)

        result = self.forward(images, targets)
        outputs = result.get("outputs", {})
        loss_dict = result.get("loss_dict", {})
        total_loss = result.get("loss", torch.tensor(0.0, device=self.device))

        self.log(f"{prefix}_loss", total_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f"{prefix}_{k}", v, on_step=False, on_epoch=True, sync_dist=True)

        return outputs, targets, loss_dict, total_loss

    def validation_step(self, batch, batch_idx):
        outputs, targets, loss_dict, total_loss = self._eval_step(batch, "val")
        self._cache_step_output(self.val_outputs, outputs, targets, loss_dict, total_loss)
        return total_loss

    def test_step(self, batch, batch_idx):
        outputs, targets, loss_dict, total_loss = self._eval_step(batch, "test")
        self._cache_step_output(self.test_outputs, outputs, targets, loss_dict, total_loss)
        return total_loss
