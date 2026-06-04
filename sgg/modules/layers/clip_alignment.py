"""
Hierarchical semantic alignment loss for scene graph generation.

Takes visual relation features and aligns them with frozen CLIP text prototypes
at multiple semantic granularities (coarse -> fine), using InfoNCE-style
contrastive learning.

Only used during training. Fully removed at inference time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalAlignmentLoss(nn.Module):
    """Multi-level InfoNCE loss between visual relation features and frozen CLIP text prototypes.

    For each hierarchy level L:
      - Each GT predicate p is mapped to cluster g_L(p) via precomputed assignments
      - For each positive relation (i,j) with GT predicate p:
          L_L = -log( exp(cosine(h_ij, prototype[g_L(p)])/tau) /
                      sum_{all g} exp(cosine(h_ij, prototype[g])/tau) )
      - Total = sum_L w_L * L_L

    Parameters
    ----------
    prototypes : list[Tensor]
        List of prototype tensors, one per hierarchy level.
        prototypes[L] shape: [C_L, clip_dim], L2-normalized.
    assignments : list[Tensor]
        assignments[L] shape: [num_predicates], dtype long.
        assignments[L][p] = the cluster index that predicate p belongs to at level L.
    hierarchy_weights : list[float]
        Weight for each level. Usually fine levels get higher weight.
    temperature : float
        Temperature for InfoNCE softmax.
    """

    def __init__(
        self,
        prototypes: list,
        assignments: list,
        hierarchy_weights: list = None,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.num_levels = len(prototypes)
        assert self.num_levels == len(assignments), \
            f"Mismatched lengths: {len(prototypes)} prototypes vs {len(assignments)} assignments"

        # Register prototypes and assignments as buffers
        for level, (proto, assign) in enumerate(zip(prototypes, assignments)):
            self.register_buffer(f"prototypes_L{level}", proto.clone())
            self.register_buffer(f"assignments_L{level}", assign.clone())

        if hierarchy_weights is None:
            # Default: increasing weight from coarse to fine
            raw = torch.linspace(1.0, float(self.num_levels), self.num_levels)
            hierarchy_weights = (raw / raw.sum()).tolist()
        assert len(hierarchy_weights) == self.num_levels
        self.register_buffer(
            "hierarchy_weights",
            torch.tensor(hierarchy_weights, dtype=torch.float32)
        )

        self.temperature = temperature

    def forward(self, rel_features, gt_predicate_labels):
        """Compute multi-level alignment loss.

        Parameters
        ----------
        rel_features : Tensor [N_pos, D]
            L2-normalized visual relation features for positive pairs.
            N_pos is the total number of matched relations in the batch.
        gt_predicate_labels : Tensor [N_pos]
            GT predicate labels (1-indexed, i.e., 1..num_predicates).

        Returns
        -------
        loss : Tensor scalar
        """
        if rel_features.numel() == 0 or gt_predicate_labels.numel() == 0:
            return rel_features.new_tensor(0.0)

        total_loss = 0.0
        for level in range(self.num_levels):
            prototypes = getattr(self, f"prototypes_L{level}")  # [C_L, D]
            assignments = getattr(self, f"assignments_L{level}")  # [num_preds]

            # Map predicate to cluster: pred_label is 1-indexed
            target_clusters = assignments[gt_predicate_labels - 1]  # [N_pos]

            # Cosine similarity: rel_features @ prototypes.T (both already L2-normed)
            sim = rel_features @ prototypes.T / self.temperature  # [N_pos, C_L]
            loss_level = F.cross_entropy(sim, target_clusters)

            weight = self.hierarchy_weights[level]
            total_loss = total_loss + weight * loss_level

        return (total_loss / self.num_levels)


def load_prototypes(prototype_path: str, device: str = "cpu") -> dict:
    """Load precomputed prototypes from a .pth file.

    Returns a dict with keys:
        prototypes: list[Tensor]
        assignments: list[Tensor]
        predicate_names: list[str]
        clip_dim: int
    """
    data = torch.load(prototype_path, map_location=device, weights_only=True)
    prototypes = [h["prototypes"] for h in data["hierarchy"]]
    assignments = [h["assignments"] for h in data["hierarchy"]]
    return {
        "prototypes": prototypes,
        "assignments": assignments,
        "predicate_names": data["predicate_names"],
        "clip_dim": data["clip_dim"],
        "num_levels": data["num_levels"],
    }
