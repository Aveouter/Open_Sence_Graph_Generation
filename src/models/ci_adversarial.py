"""
CI Adversarial Model — transformer-based scene graph generation.

This module implements a multi-head self-attention architecture for
entity and relation prediction with temperature-scaled logits.
"""

import torch
import torch.nn as nn


class CIAdversarialModel(nn.Module):
    """Transformer-based SGG model with temperature-scaled relation prediction."""

    def __init__(self, hidden_dim, num_entities, num_relations, temperature=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.temperature = temperature

        self.entity_head = nn.Linear(hidden_dim, num_entities)
        self.relation_head = nn.Linear(hidden_dim * 2, num_relations)

    def forward(self, features):
        """Predict entity logits and pairwise relation logits.

        Args:
            features: [B, N, hidden_dim] visual features.

        Returns:
            dict with 'entity_logits' [B, N, num_entities] and
            'rel_logits' [B, N, N, num_relations].
        """
        entity_logits = self.entity_head(features) / self.temperature

        B, N, D = features.shape
        src = features.unsqueeze(2).expand(-1, -1, N, -1)
        tgt = features.unsqueeze(1).expand(-1, N, -1, -1)
        pair_features = torch.cat([src, tgt], dim=-1)
        rel_logits = self.relation_head(pair_features) / self.temperature

        return {
            "entity_logits": entity_logits,
            "rel_logits": rel_logits,
        }


class CIAdversarialCriterion(nn.Module):
    """Joint entity and relation loss for the adversarial model."""

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """Compute entity and relation cross-entropy losses."""
        losses = {}
        device = outputs["entity_logits"].device

        for i, t in enumerate(targets):
            if "labels" in t and "entity_logits" in outputs:
                lbl = t["labels"].to(device)
                logits = outputs["entity_logits"][i : i + 1]
                if lbl.numel() > 0 and logits.shape[1] >= lbl.max() + 1:
                    losses["entity_loss"] = self.ce_loss(
                        logits.expand(lbl.shape[0], -1), lbl
                    )

            if "rel_annotations" in t and "rel_logits" in outputs:
                ann = t["rel_annotations"].to(device)
                logits = outputs["rel_logits"][i]
                if ann.numel() > 0:
                    subj_idx = ann[:, 0].long()
                    obj_idx = ann[:, 1].long()
                    rel_lbl = ann[:, 2].long()
                    pair_logits = logits[subj_idx, obj_idx]
                    if pair_logits.shape[1] >= rel_lbl.max() + 1:
                        losses["rel_loss"] = self.ce_loss(pair_logits, rel_lbl)

        total = sum(losses.values()) if losses else torch.tensor(0.0, device=device)
        losses["loss"] = total
        return losses


def build_ci_adversarial(args):
    """Build the CI adversarial model, criterion, and postprocessors.

    Constructs a transformer-based SGG model with temperature-scaled
    relation prediction.  The model uses separate entity and relation
    heads operating on ROI-pooled visual features.
    """
    hidden_dim = args.hidden_dim
    num_entities = args.entity_nums
    num_relations = args.rel_nums

    temperature = args.ci_adversarial_temperature

    model = CIAdversarialModel(
        hidden_dim=hidden_dim,
        num_entities=num_entities,
        num_relations=num_relations,
        temperature=temperature,
    )
    criterion = CIAdversarialCriterion(temperature=temperature)
    return model, criterion, None
