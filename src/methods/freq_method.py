"""Frequency-prior baseline for Visual Genome scene graph generation."""

import torch
import torch.nn as nn

from .base_method import Base_method
from .motifs_method import MotifsCriterion
from src.models.motifs import PairFrequencyBias, generate_object_pairs


class FREQModel(nn.Module):
    """Predict predicates from P(predicate | subject class, object class)."""

    def __init__(self, args):
        super().__init__()
        rel_nums = getattr(args, "rel_nums", 51)
        entity_nums = getattr(args, "entity_nums", 151)
        self.predicate_bg_index = getattr(args, "freq_predicate_bg_index", "first")
        self.num_predicates = rel_nums
        self.freq_bias = PairFrequencyBias(
            num_objects=entity_nums,
            num_predicates=rel_nums,
            eps=getattr(args, "freq_bias_eps", 1e-3),
            data_root=getattr(args, "data_root", None),
            predicate_bg_index=self.predicate_bg_index,
        )
        # Keep optimizer construction valid while preserving the frozen prior.
        self.dummy_param = nn.Parameter(torch.zeros(()))

    def forward(self, boxes, labels):
        pairs = generate_object_pairs(labels.numel(), labels.device)
        if pairs.numel() == 0:
            rel_logits = labels.new_zeros((0, self.num_predicates), dtype=torch.float)
        else:
            pair_labels = torch.stack((labels[pairs[:, 0]], labels[pairs[:, 1]]), dim=1)
            rel_logits = self.freq_bias.index_with_labels(pair_labels.long())
            rel_logits = rel_logits + self.dummy_param * 0.0

        return {
            "rel_logits": rel_logits,
            "pair_indices": pairs,
            "sub_boxes": boxes[pairs[:, 0]] if pairs.numel() else boxes.new_zeros((0, 4)),
            "obj_boxes": boxes[pairs[:, 1]] if pairs.numel() else boxes.new_zeros((0, 4)),
            "obj_labels": labels,
            "predicate_bg_index": self.predicate_bg_index,
            "relation_softmax_scope": "all",
        }


class FREQ_Method(Base_method):
    """Lightning wrapper for the non-visual FREQ prior baseline."""

    def _build_criterion(self, **args):
        return MotifsCriterion(num_predicates=args.get("rel_nums", 51))

    def _build_model(self, **args):
        return FREQModel(self.hparams)

    def forward(self, images, targets=None, **kwargs):
        if targets is None:
            return {"outputs": {}}

        all_outputs = [
            self.model(t["boxes"].to(self.device), t["labels"].to(self.device))
            for t in targets
        ]
        batched = {
            "rel_logits": [o["rel_logits"] for o in all_outputs],
            "pair_indices": [o["pair_indices"] for o in all_outputs],
            "sub_boxes": [o["sub_boxes"] for o in all_outputs],
            "obj_boxes": [o["obj_boxes"] for o in all_outputs],
            "obj_labels": [o["obj_labels"] for o in all_outputs],
            "predicate_bg_index": self.model.predicate_bg_index,
            "relation_softmax_scope": "all",
        }
        out = {"outputs": batched}
        if self.criterion is not None:
            loss_dict = self.criterion(batched, targets)
            out["loss_dict"] = loss_dict
            out["loss"] = loss_dict["loss_total"]
        return out

    def training_step(self, batch, batch_idx):
        images, targets = self._split_batch(batch)
        targets = self._move_targets_to_device(targets)
        result = self.forward(images, targets)
        loss = result.get("loss", torch.tensor(0.0, device=self.device))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def _eval_step(self, batch, prefix: str):
        images, targets = self._split_batch(batch)
        targets = self._move_targets_to_device(targets)
        result = self.forward(images, targets)
        outputs = result.get("outputs", {})
        loss_dict = result.get("loss_dict", {})
        total_loss = result.get("loss", torch.tensor(0.0, device=self.device))
        self.log(f"{prefix}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return outputs, targets, loss_dict, total_loss
