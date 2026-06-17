"""
SHA-GCL method wrapper for PyTorch Lightning.
"""
import torch
from .motifs_method import Motifs_Method, MotifsCriterion
from src.models.shagcl import build_shagcl


class SHAGCLCriterion(MotifsCriterion):
    """Extended criterion for SHA-GCL with GCL loss."""

    def __init__(self, num_predicates: int, gcl_weight: float = 0.1):
        super().__init__(num_predicates)
        self.gcl_weight = gcl_weight

    def forward(self, outputs: dict, targets: list) -> dict:
        loss_dict = super().forward(outputs, targets)

        add_losses = outputs.get("add_losses", {})
        if isinstance(add_losses, dict) and "gcl_loss" in add_losses:
            gcl = add_losses["gcl_loss"]
            if torch.is_tensor(gcl):
                loss_dict["loss_gcl"] = gcl * self.gcl_weight
                loss_dict["loss_total"] = loss_dict["loss_total"] + loss_dict["loss_gcl"]

        return loss_dict


class SHAGCL_Method(Motifs_Method):
    """SHA-GCL Lightning method."""

    def _build_model(self, **args):
        return build_shagcl(self.hparams)

    def _build_criterion(self, **args):
        return SHAGCLCriterion(
            num_predicates=args.get('rel_nums', 51),
            gcl_weight=args.get('shagcl_gcl_weight', 0.1),
        )

    def _extra_model_kwargs(self, target, boxes, labels, return_obj_preds):
        device = boxes.device
        num_obj = int(boxes.size(0))
        idx = torch.arange(num_obj, device=device)
        grid_i, grid_j = torch.meshgrid(idx, idx, indexing='ij')
        pair_mask = grid_i != grid_j
        pairs = torch.stack([grid_i[pair_mask], grid_j[pair_mask]], dim=-1)

        gt_rel_labels = torch.full(
            (pairs.size(0),), -1, dtype=torch.long, device=device)
        rel_anns = target.get("rel_annotations")
        if rel_anns is not None and rel_anns.numel() > 0:
            rel_anns = rel_anns.to(device)
            pair_to_idx = {
                (int(pair[0]), int(pair[1])): pair_idx
                for pair_idx, pair in enumerate(pairs)
            }
            for ann in rel_anns:
                key = (int(ann[0]), int(ann[1]))
                if key in pair_to_idx:
                    gt_rel_labels[pair_to_idx[key]] = int(ann[2])

        num_predicates = int(getattr(self.hparams, 'rel_nums', 51))
        num_groups = max(1, int(getattr(self.hparams, 'shagcl_num_groups', 3)))
        group_assignments = torch.arange(
            num_predicates, device=device, dtype=torch.long)
        if num_predicates > 1:
            group_assignments = torch.div(
                group_assignments * num_groups,
                num_predicates,
                rounding_mode='trunc',
            ).clamp(max=num_groups - 1)

        return {
            "gt_rel_labels": gt_rel_labels,
            "group_assignments": group_assignments,
        }
