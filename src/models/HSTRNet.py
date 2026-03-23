import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.layers.hstr_blocks import (
    QueryObjectEncoder,
    RelationProposalNetwork,
    TemporalRelationEncoder,
    HierarchicalPrototypeModule,
    PredicateClassifier,
    TripletCompatibilityHead,
    HSTRCriterion,
    batched_gather_queries,
)


class HSTRNetModel(nn.Module):
    """
    主模型层：
    object encoder
      -> relation proposal
      -> temporal relation encoder
      -> hierarchical prototype learning
      -> predicate head
      -> compatibility head
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.hidden_dim = getattr(args, "hidden_dim", 256)
        self.num_object_queries = getattr(args, "num_object_queries", 16)
        self.num_object_classes = getattr(args, "num_object_classes", 23)
        self.num_predicates = getattr(args, "num_predicates", 29)
        self.max_relation_pairs = getattr(args, "max_relation_pairs", 32)
        self.prototype_levels = getattr(args, "prototype_levels", [8, 16])
        self.in_channels = getattr(args, "in_channels", 3)
        self.temporal_num_layers = getattr(args, "temporal_num_layers", 2)
        self.temporal_num_heads = getattr(args, "temporal_num_heads", 8)
        self.temporal_dropout = getattr(args, "temporal_dropout", 0.1)
        self.prototype_dim = getattr(args, "prototype_dim", self.hidden_dim)

        self.object_encoder = QueryObjectEncoder(
            in_channels=self.in_channels,
            hidden_dim=self.hidden_dim,
            num_queries=self.num_object_queries,
            num_object_classes=self.num_object_classes,
        )

        self.relation_proposal = RelationProposalNetwork(
            hidden_dim=self.hidden_dim,
            max_relation_pairs=self.max_relation_pairs,
        )

        self.temporal_relation_encoder = TemporalRelationEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=self.temporal_num_layers,
            num_heads=self.temporal_num_heads,
            dropout=self.temporal_dropout,
        )

        self.prototype_module = HierarchicalPrototypeModule(
            hidden_dim=self.hidden_dim,
            prototype_dim=self.prototype_dim,
            level_sizes=self.prototype_levels,
        )

        self.predicate_head = PredicateClassifier(
            hidden_dim=self.hidden_dim,
            num_predicates=self.num_predicates,
        )

        self.compatibility_head = TripletCompatibilityHead(
            hidden_dim=self.hidden_dim,
            num_predicates=self.num_predicates,
        )

        self.criterion = HSTRCriterion(args)

    def forward(self, images: torch.Tensor, targets: Optional[Dict] = None) -> Dict:
        """
        images: [B, T, C, H, W]
        """
        # assert images.dim() == 5, f"Expected [B, T, C, H, W], got {images.shape}"

        # 1) query-based object representation
        obj_out = self.object_encoder(images)
        object_queries = obj_out["object_queries"]   # [B, T, Nq, D]
        object_logits = obj_out["object_logits"]     # [B, T, Nq, No]

        # 2) learned relation proposal
        rel_prop_out = self.relation_proposal(object_queries)
        relationness_logits_dense = rel_prop_out["relationness_logits_dense"]  # [B,T,Nq,Nq]
        pair_indices = rel_prop_out["relation_pair_indices"]                   # [B,T,K,2]
        pair_features = rel_prop_out["pair_features"]                          # [B,T,K,D]

        # 3) temporal relation state modeling
        temp_out = self.temporal_relation_encoder(pair_features)
        relation_features = temp_out["relation_features"]                      # [B,K,D]

        # 4) vision-adaptive hierarchical prototype learning
        proto_out = self.prototype_module(relation_features)
        refined_relation_features = proto_out["refined_features"]              # [B,K,D]

        # 5) predicate classification
        predicate_logits = self.predicate_head(refined_relation_features)      # [B,K,P]

        # 6) compatibility / energy scoring
        subj_q, obj_q = batched_gather_queries(object_queries, pair_indices)   # [B,T,K,D], [B,T,K,D]
        subj_feat = subj_q.mean(dim=1)                                         # [B,K,D]
        obj_feat = obj_q.mean(dim=1)                                           # [B,K,D]

        energy_scores = self.compatibility_head(
            subj_feat=subj_feat,
            obj_feat=obj_feat,
            predicate_logits=predicate_logits,
        )                                                                      # [B,K]

        final_predicate_logits = predicate_logits + energy_scores.unsqueeze(-1)

        outputs = {
            "object_queries": object_queries,
            "object_logits": object_logits,
            "relationness_logits_dense": relationness_logits_dense,
            "relation_pair_indices": pair_indices,
            "pair_features": pair_features,
            "temporal_relation_features": temp_out["temporal_relation_features"],
            "relation_features": refined_relation_features,
            "predicate_logits": predicate_logits,
            "energy_scores": energy_scores,
            "final_predicate_logits": final_predicate_logits,
            "prototype_outputs": proto_out,
        }

        if targets is not None:
            losses = self.criterion(outputs, targets)
            outputs["losses"] = losses

        return outputs