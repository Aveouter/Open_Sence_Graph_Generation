import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Utility
# =========================================================

def batched_gather_queries(
    queries: torch.Tensor,   # [B, T, Nq, D]
    pair_idx: torch.Tensor   # [B, T, K, 2]
):
    B, T, Nq, D = queries.shape
    K = pair_idx.shape[2]

    subj_idx = pair_idx[..., 0]
    obj_idx = pair_idx[..., 1]

    subj = torch.gather(
        queries,
        dim=2,
        index=subj_idx.unsqueeze(-1).expand(B, T, K, D)
    )
    obj = torch.gather(
        queries,
        dim=2,
        index=obj_idx.unsqueeze(-1).expand(B, T, K, D)
    )
    return subj, obj


# =========================================================
# Backbone + Query Object Encoder
# =========================================================

class SimpleCNNBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0

    def forward(self, feat):
        B, D, H, W = feat.shape
        device = feat.device

        y_embed = torch.linspace(0, 1, H, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.linspace(0, 1, W, device=device).unsqueeze(0).repeat(H, 1)

        d = D // 4
        omega = torch.arange(d, device=device).float()
        omega = 1.0 / (10000 ** (omega / d))

        y = y_embed.reshape(-1, 1) * omega.reshape(1, -1)
        x = x_embed.reshape(-1, 1) * omega.reshape(1, -1)

        pe = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)
        pe = pe.unsqueeze(0).repeat(B, 1, 1)  # [B, HW, D]
        return pe


class QueryObjectEncoder(nn.Module):
    """
    输入:
        images: [B, T, C, H, W]
    输出:
        object_queries: [B, T, Nq, D]
        object_logits : [B, T, Nq, No]
    """

    def __init__(self, in_channels, hidden_dim, num_queries, num_object_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_object_classes = num_object_classes

        self.backbone = SimpleCNNBackbone(in_channels, hidden_dim)
        self.pos2d = PositionalEncoding2D(hidden_dim)

        self.object_queries = nn.Parameter(
            torch.randn(num_queries, hidden_dim) * 0.02
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.object_cls_head = nn.Linear(hidden_dim, num_object_classes)

    def forward(self, images):
        B, T, C, H, W = images.shape
        x = images.reshape(B * T, C, H, W)

        feat = self.backbone(x)                        # [B*T, D, H', W']
        BT, D, Hf, Wf = feat.shape

        tokens = feat.flatten(2).transpose(1, 2)      # [B*T, HW, D]
        tokens = tokens + self.pos2d(feat)

        q = self.object_queries.unsqueeze(0).repeat(BT, 1, 1)  # [B*T, Nq, D]

        attn_out, _ = self.cross_attn(q, tokens, tokens)
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ffn(q))

        object_logits = self.object_cls_head(q)       # [B*T, Nq, No]

        return {
            "object_queries": q.reshape(B, T, self.num_queries, D),
            "object_logits": object_logits.reshape(B, T, self.num_queries, self.num_object_classes),
        }


# =========================================================
# Relation Proposal
# =========================================================

class PairFeatureBuilder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, queries):
        B, T, Nq, D = queries.shape

        qi = queries.unsqueeze(3).expand(B, T, Nq, Nq, D)
        qj = queries.unsqueeze(2).expand(B, T, Nq, Nq, D)

        pair = torch.cat([qi, qj, qi - qj, qi * qj], dim=-1)
        return self.proj(pair)


class RelationProposalNetwork(nn.Module):
    """
    输出:
        relationness_logits_dense: [B, T, Nq, Nq]
        relation_pair_indices: [B, T, K, 2]
        pair_features: [B, T, K, D]
    """

    def __init__(self, hidden_dim, max_relation_pairs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_relation_pairs = max_relation_pairs

        self.builder = PairFeatureBuilder(hidden_dim)
        self.relationness_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, object_queries):
        B, T, Nq, D = object_queries.shape
        device = object_queries.device

        dense_pair_features = self.builder(object_queries)          # [B,T,Nq,Nq,D]
        relationness_logits = self.relationness_head(dense_pair_features).squeeze(-1)

        eye = torch.eye(Nq, device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        relationness_logits = relationness_logits.masked_fill(eye, -1e9)

        K = min(self.max_relation_pairs, Nq * (Nq - 1))

        flat_logits = relationness_logits.reshape(B, T, -1)        # [B,T,Nq*Nq]
        topk_vals, topk_idx = torch.topk(flat_logits, k=K, dim=-1)

        pair_i = topk_idx // Nq
        pair_j = topk_idx % Nq
        pair_indices = torch.stack([pair_i, pair_j], dim=-1)       # [B,T,K,2]

        pair_flat_feat = dense_pair_features.reshape(B, T, Nq * Nq, D)
        pair_features = torch.gather(
            pair_flat_feat,
            dim=2,
            index=topk_idx.unsqueeze(-1).expand(B, T, K, D)
        )

        return {
            "relationness_logits_dense": relationness_logits,
            "relation_pair_indices": pair_indices,
            "pair_features": pair_features,
            "topk_relationness_logits": topk_vals,
        }


# =========================================================
# Temporal Relation Encoder
# =========================================================

class LearnableTemporalPositionalEncoding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TemporalRelationEncoder(nn.Module):
    """
    输入:
        pair_features: [B, T, K, D]
    输出:
        temporal_relation_features: [B, T, K, D]
        relation_features: [B, K, D]
    """

    def __init__(self, hidden_dim, num_layers=2, num_heads=8, dropout=0.1, max_time=64):
        super().__init__()
        self.temporal_pe = LearnableTemporalPositionalEncoding(max_time, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, pair_features):
        B, T, K, D = pair_features.shape

        x = pair_features.permute(0, 2, 1, 3).reshape(B * K, T, D)  # [B*K,T,D]
        x = self.temporal_pe(x)
        x = self.encoder(x)

        score = self.pool_proj(x).mean(dim=-1)                      # [B*K,T]
        weight = F.softmax(score, dim=-1)
        rel_feat = torch.sum(x * weight.unsqueeze(-1), dim=1)       # [B*K,D]

        temporal_features = x.reshape(B, K, T, D).permute(0, 2, 1, 3)
        relation_features = rel_feat.reshape(B, K, D)

        return {
            "temporal_relation_features": temporal_features,
            "relation_features": relation_features,
        }


# =========================================================
# Hierarchical Prototype Learning
# =========================================================

class HierarchicalPrototypeModule(nn.Module):
    """
    输入:
        relation_features: [B, K, D]
    输出:
        level_logits / level_probs / refined_features
    """

    def __init__(self, hidden_dim, prototype_dim, level_sizes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prototype_dim = prototype_dim
        self.level_sizes = level_sizes
        self.num_levels = len(level_sizes)

        self.feature_proj = nn.Linear(hidden_dim, prototype_dim)

        self.prototypes = nn.ParameterList()
        for p in level_sizes:
            self.prototypes.append(nn.Parameter(torch.randn(p, prototype_dim) * 0.02))

        self.refine_proj = nn.Linear(prototype_dim * self.num_levels, hidden_dim)

        self.parent_child_logits = nn.ParameterList()
        for i in range(self.num_levels - 1):
            p1 = level_sizes[i]
            p2 = level_sizes[i + 1]
            self.parent_child_logits.append(
                nn.Parameter(torch.randn(p1, p2) * 0.02)
            )

    def forward(self, relation_features):
        B, K, D = relation_features.shape

        feat = self.feature_proj(relation_features)
        feat = F.normalize(feat, dim=-1)

        level_logits = []
        level_probs = []
        level_contexts = []

        for proto in self.prototypes:
            proto_norm = F.normalize(proto, dim=-1)
            logits = torch.matmul(feat, proto_norm.t())             # [B,K,P_l]
            probs = F.softmax(logits, dim=-1)
            context = torch.matmul(probs, proto_norm)               # [B,K,Dp]

            level_logits.append(logits)
            level_probs.append(probs)
            level_contexts.append(context)

        refined = relation_features + self.refine_proj(torch.cat(level_contexts, dim=-1))

        return {
            "level_logits": level_logits,
            "level_probs": level_probs,
            "level_prototypes": [p for p in self.prototypes],
            "parent_child_logits": [p for p in self.parent_child_logits],
            "refined_features": refined,
        }


# =========================================================
# Predicate Head
# =========================================================

class PredicateClassifier(nn.Module):
    def __init__(self, hidden_dim, num_predicates):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_predicates),
        )

    def forward(self, relation_features):
        return self.head(relation_features)


# =========================================================
# Compatibility Head
# =========================================================

class TripletCompatibilityHead(nn.Module):
    """
    输入:
        subj_feat: [B,K,D]
        obj_feat: [B,K,D]
        predicate_logits: [B,K,P]
    输出:
        energy_scores: [B,K]
    """

    def __init__(self, hidden_dim, num_predicates):
        super().__init__()
        self.predicate_embed = nn.Linear(num_predicates, hidden_dim)
        self.energy_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, subj_feat, obj_feat, predicate_logits):
        pred_prob = F.softmax(predicate_logits, dim=-1)
        pred_feat = self.predicate_embed(pred_prob)
        fused = torch.cat([subj_feat, pred_feat, obj_feat], dim=-1)
        energy = self.energy_mlp(fused).squeeze(-1)
        return energy


# =========================================================
# Criterion
# =========================================================

class HSTRCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_object_weight = getattr(args, "loss_object_weight", 1.0)
        self.loss_relationness_weight = getattr(args, "loss_relationness_weight", 1.0)
        self.loss_predicate_weight = getattr(args, "loss_predicate_weight", 1.0)
        self.loss_proto_align_weight = getattr(args, "loss_proto_align_weight", 0.2)
        self.loss_hier_consistency_weight = getattr(args, "loss_hier_consistency_weight", 0.2)
        self.loss_energy_weight = getattr(args, "loss_energy_weight", 0.2)

    def object_loss(self, object_logits, targets):
        if "object_labels" not in targets:
            return object_logits.new_tensor(0.0)
        gt = targets["object_labels"]  # [B,T,Nq]
        B, T, Nq, No = object_logits.shape
        return F.cross_entropy(
            object_logits.reshape(B * T * Nq, No),
            gt.reshape(B * T * Nq),
            ignore_index=-100
        )

    def relationness_loss(self, relationness_logits_dense, targets):
        if "relation_labels_dense" not in targets:
            return relationness_logits_dense.new_tensor(0.0)
        gt = targets["relation_labels_dense"].float()
        loss = F.binary_cross_entropy_with_logits(relationness_logits_dense, gt, reduction="mean")
        return loss

    def predicate_loss(self, final_predicate_logits, targets):
        if "pair_predicate_labels" not in targets:
            return final_predicate_logits.new_tensor(0.0)

        gt = targets["pair_predicate_labels"]  # [B,K]
        B, K, P = final_predicate_logits.shape

        loss = F.cross_entropy(
            final_predicate_logits.reshape(B * K, P),
            gt.reshape(B * K),
            ignore_index=-100
        )
        return loss

    def prototype_alignment_loss(self, prototype_outputs):
        probs_list = prototype_outputs["level_probs"]
        loss = 0.0
        for probs in probs_list:
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            loss += entropy
        return loss / max(len(probs_list), 1)

    def hierarchical_consistency_loss(self, prototype_outputs):
        if len(prototype_outputs["parent_child_logits"]) == 0:
            return prototype_outputs["refined_features"].new_tensor(0.0)

        total_loss = 0.0
        count = 0
        level_probs = prototype_outputs["level_probs"]
        trans_list = prototype_outputs["parent_child_logits"]

        for i, trans_logits in enumerate(trans_list):
            coarse_prob = level_probs[i]           # [B,K,Pc]
            fine_prob = level_probs[i + 1]         # [B,K,Pf]
            trans = F.softmax(trans_logits, dim=-1)
            fine_to_coarse = torch.matmul(fine_prob, trans.t())
            total_loss += F.mse_loss(fine_to_coarse, coarse_prob)
            count += 1

        return total_loss / max(count, 1)

    def compatibility_loss(self, energy_scores, targets):
        if "pair_compatibility_labels" not in targets:
            return energy_scores.new_tensor(0.0)
        gt = targets["pair_compatibility_labels"].float()
        return F.binary_cross_entropy_with_logits(energy_scores, gt)

    def forward(self, outputs, targets):
        l_obj = self.object_loss(outputs["object_logits"], targets)
        l_rel = self.relationness_loss(outputs["relationness_logits_dense"], targets)
        l_pred = self.predicate_loss(outputs["final_predicate_logits"], targets)
        l_proto = self.prototype_alignment_loss(outputs["prototype_outputs"])
        l_hier = self.hierarchical_consistency_loss(outputs["prototype_outputs"])
        l_energy = self.compatibility_loss(outputs["energy_scores"], targets)

        total = (
            self.loss_object_weight * l_obj
            + self.loss_relationness_weight * l_rel
            + self.loss_predicate_weight * l_pred
            + self.loss_proto_align_weight * l_proto
            + self.loss_hier_consistency_weight * l_hier
            + self.loss_energy_weight * l_energy
        )

        return {
            "loss_total": total,
            "loss_object": l_obj,
            "loss_relationness": l_rel,
            "loss_predicate": l_pred,
            "loss_proto_align": l_proto,
            "loss_hier_consistency": l_hier,
            "loss_energy": l_energy,
        }