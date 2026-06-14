"""
REACT method wrapper for PyTorch Lightning.

Extends Motifs_Method with prototype regularization losses (L21 + distance).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .motifs_method import Motifs_Method, MotifsCriterion
from src.models.react_sgg import build_react


class REACTCriterion(MotifsCriterion):
    """Extended criterion for REACT with prototype regularization losses.

    Adds L21 (semantic matrix sparsity) and distance loss (prototype separation)
    on top of the standard cross-entropy predicate loss.
    """

    def __init__(self, num_predicates: int,
                 l21_weight: float = 1.0,
                 dist_weight: float = 1.0):
        super().__init__(num_predicates)
        self.l21_weight = l21_weight
        self.dist_weight = dist_weight

    def forward(self, outputs: dict, targets: list) -> dict:
        # Standard predicate loss
        loss_dict = super().forward(outputs, targets)

        # Add prototype regularization from model outputs
        add_losses = outputs.get("add_losses", {})
        if isinstance(add_losses, dict):
            if "l21_loss" in add_losses and self.l21_weight > 0:
                l21 = add_losses["l21_loss"]
                if torch.is_tensor(l21):
                    loss_dict["loss_l21"] = l21 * self.l21_weight
                    loss_dict["loss_total"] = loss_dict["loss_total"] + loss_dict["loss_l21"]

            if "dist_loss" in add_losses and self.dist_weight > 0:
                dist = add_losses["dist_loss"]
                if torch.is_tensor(dist):
                    loss_dict["loss_dist"] = dist * self.dist_weight
                    loss_dict["loss_total"] = loss_dict["loss_total"] + loss_dict["loss_dist"]

        return loss_dict


class REACT_Method(Motifs_Method):
    """REACT Lightning method.

    Extends Motifs with prototype-based cosine similarity classification
    and prototype regularization losses.
    """

    def _build_model(self, **args):
        return build_react(self.hparams)

    def _build_criterion(self, **args):
        return REACTCriterion(
            num_predicates=args.get('rel_nums', 51),
            l21_weight=args.get('react_l21_weight', 1.0),
            dist_weight=args.get('react_dist_weight', 1.0),
        )

    def forward(self, images, targets=None, **kwargs):
        """Forward pass with add_losses propagated by Motifs_Method."""
        return super().forward(images, targets, **kwargs)
