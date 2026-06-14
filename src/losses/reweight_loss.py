"""
Reweighting and class-balanced loss functions for SGG.

Ported from SGG-Benchmark (Maelic/SGG-Benchmark).
These address the severe long-tail distribution in VisualGenome predicates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReweightingCE(nn.Module):
    """Cross-entropy loss with inverse-frequency reweighting.

    Args:
        pred_weight: [num_classes] tensor of per-class weights.
                     Typically 1/freq for each class.
    """

    def __init__(self, pred_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pred_weight", pred_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute reweighted CE loss.

        Args:
            logits: [N, C] or [N, C] predicate logits.
            labels: [N] ground truth labels.

        Returns:
            scalar loss.
        """
        return F.cross_entropy(logits, labels, weight=self.pred_weight.to(logits.device))


class ClassBalancedCELoss(nn.Module):
    """Class-balanced cross-entropy loss.

    Reference: "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)

    Weight = (1 - beta) / (1 - beta^n_i) where n_i is the sample count for class i.

    Args:
        samples_per_class: [num_classes] tensor of sample counts.
        num_classes: Number of classes.
        beta: Hyperparameter controlling balance (0 = no reweighting, 1 = full inverse).
        loss_type: 'ce' for cross-entropy.
    """

    def __init__(self, samples_per_class: torch.Tensor, num_classes: int,
                 beta: float = 0.9999, loss_type: str = 'ce'):
        super().__init__()
        self.beta = beta
        self.num_classes = num_classes

        effective_num = 1.0 - beta ** samples_per_class.float()
        weights = (1.0 - beta) / (effective_num + 1e-8)
        # Normalize
        weights = weights / weights.sum() * num_classes

        self.register_buffer("weights", weights)
        self.loss_type = loss_type

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'ce':
            return F.cross_entropy(logits, labels, weight=self.weights.to(logits.device))
        else:
            return F.cross_entropy(logits, labels, weight=self.weights.to(logits.device))


class CBFocalLoss(nn.Module):
    """Class-balanced focal loss.

    Combines focal loss (for hard example mining) with class-balanced reweighting.

    Args:
        samples_per_class: [num_classes] tensor of sample counts.
        num_classes: Number of classes.
        beta: Class-balancing hyperparameter.
        gamma: Focal loss gamma (focus on hard examples).
    """

    def __init__(self, samples_per_class: torch.Tensor, num_classes: int,
                 beta: float = 0.9999, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes

        effective_num = 1.0 - beta ** samples_per_class.float()
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * num_classes

        self.register_buffer("weights", weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute class-balanced focal loss.

        Args:
            logits: [N, C] logits.
            labels: [N] integer labels.

        Returns:
            scalar loss.
        """
        ce_loss = F.cross_entropy(logits, labels, weight=self.weights.to(logits.device),
                                   reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class FocalLoss(nn.Module):
    """Standard focal loss.

    Args:
        gamma: Focusing parameter (default 2.0).
        alpha: Class balancing weight tensor (optional).
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, labels, weight=self.alpha.to(logits.device) if self.alpha is not None else None,
            reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class EdgeDensityLoss(nn.Module):
    """Edge density prior loss.

    Penalizes the model for predicting too few or too many edges,
    encouraging a target edge density.

    Args:
        target_density: Target fraction of edges that should be active (non-bg).
    """

    def __init__(self, target_density: float = 0.1):
        super().__init__()
        self.target_density = target_density

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute edge density loss.

        Args:
            logits: [N, C] predicate logits (C includes background).

        Returns:
            scalar loss penalizing deviation from target density.
        """
        probs = F.softmax(logits, dim=-1)
        # Probability of being a non-background predicate
        if logits.size(-1) > 1:
            non_bg_prob = 1 - probs[:, 0]  # class 0 = background
        else:
            non_bg_prob = probs[:, 0]
        density = non_bg_prob.mean()
        return F.mse_loss(density, torch.tensor(self.target_density, device=logits.device))


class HierarchicalLoss(nn.Module):
    """Hierarchical relation loss with predicate grouping.

    Groups predicates into semantic categories (geo, pos, sem) and
    computes both per-group and overall cross-entropy losses.

    Args:
        num_predicates: Total number of predicate classes.
        group_assignments: [num_predicates] tensor of group indices.
        group_weights: [num_groups] tensor of per-group loss weights.
    """

    def __init__(self, num_predicates: int, group_assignments: torch.Tensor,
                 group_weights: torch.Tensor = None):
        super().__init__()
        self.num_predicates = num_predicates
        self.register_buffer("group_assignments", group_assignments)
        num_groups = int(group_assignments.max().item()) + 1
        if group_weights is None:
            group_weights = torch.ones(num_groups)
        self.register_buffer("group_weights", group_weights)
        self.num_groups = num_groups

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple:
        """Compute hierarchical loss.

        Args:
            logits: [N, C] predicate logits.
            labels: [N] GT predicate labels.

        Returns:
            (total_loss, per_group_losses dict)
        """
        # Overall CE loss
        total_loss = F.cross_entropy(logits, labels, reduction='mean')
        per_group = {"total": total_loss}

        # Per-group CE loss
        for g in range(self.num_groups):
            group_mask = self.group_assignments[labels] == g
            if group_mask.sum() < 2:
                per_group[f"group_{g}"] = logits.new_tensor(0.0)
                continue

            # Only compute over classes in this group
            group_logits = logits[group_mask]
            group_labels = labels[group_mask]
            per_group[f"group_{g}"] = F.cross_entropy(
                group_logits, group_labels, reduction='mean')

        return total_loss, per_group
