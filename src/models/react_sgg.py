"""
REACT: Real-time Efficiency and Accuracy Compromise for Tradeoffs in SGG.

Reference: "REACT: Real-time Efficiency and Accuracy Compromise for Tradeoffs
in Scene Graph Generation" (Neau et al., BMVC 2025)

Architecture:
  1. Visual-Semantic Fusion: gated combination of GloVe embeddings + ROI features
     s = Ws*ts + gs · h(xs)
     o = Wo*to + go · h(xo)
  2. Union Feature Debiasing:
     rel_rep = F(s, o) - gp · h(xu)   (subtract union bias)
  3. Prototype-based Classification:
     predicate_logits = cos_sim(rel_rep_norm, prototype_norm) / τ
  4. Prototype Regularization:
     - L21: ||S||_{2,1}  (semantic matrix sparsity)
     - Distance: max(0, -d_min + γ)  (prototype separation)

Key innovations:
  - Cosine similarity instead of linear classifier (better for few-shot)
  - Union debiasing removes spatial/visual shortcuts
  - Prototype regularization keeps predicate embeddings well-separated
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .motifs import FrequencyBias, generate_object_pairs


# ──────────────────────────────────────────────
# MLP helper
# ──────────────────────────────────────────────

class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        current_dim = in_dim
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else out_dim
            layers.append(nn.Linear(current_dim, next_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
            current_dim = next_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# REACT Model
# ──────────────────────────────────────────────

class REACTModel(nn.Module):
    """REACT: Real-time scene graph generation with prototype learning.

    Args:
        num_classes: Number of object classes.
        num_predicates: Number of predicate classes.
        visual_dim: Input visual feature dimension.
        hidden_dim: Internal hidden dimension (mlp_dim in paper).
        embed_dim: GloVe word embedding dimension.
        use_union: Whether to use union feature for debiasing.
        text_only: Use only text features (ablation).
        use_freq_bias: Whether to use frequency bias.
        dropout: Dropout probability.
    """

    def __init__(self, num_classes: int = 151, num_predicates: int = 51,
                 visual_dim: int = 2048, hidden_dim: int = 512,
                 embed_dim: int = 200,
                 use_union: bool = True,
                 text_only: bool = False,
                 use_freq_bias: bool = True, freq_bias_eps: float = 1e-12,
                 dropout: float = 0.1):
        super().__init__()

        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.use_union = use_union
        self.text_only = text_only
        self.use_freq_bias = use_freq_bias

        # Visual feature → subject/object split
        self.post_emb = nn.Linear(visual_dim, hidden_dim * 2)

        # Semantic projections
        self.W_sub = MLP(embed_dim, hidden_dim // 2, hidden_dim, 2)
        self.W_obj = MLP(embed_dim, hidden_dim // 2, hidden_dim, 2)
        self.W_pred = MLP(embed_dim, hidden_dim // 2, hidden_dim, 2)

        # Gating
        self.gate_sub = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_obj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Visual-to-semantic
        self.vis2sem = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Residual + normalization
        self.linear_sub = nn.Linear(hidden_dim, hidden_dim)
        self.linear_obj = nn.Linear(hidden_dim, hidden_dim)
        self.linear_pred = nn.Linear(hidden_dim, hidden_dim)
        self.linear_rel_rep = nn.Linear(hidden_dim, hidden_dim)

        self.norm_sub = nn.LayerNorm(hidden_dim)
        self.norm_obj = nn.LayerNorm(hidden_dim)
        self.norm_rel_rep = nn.LayerNorm(hidden_dim)

        self.dropout_sub = nn.Dropout(dropout)
        self.dropout_obj = nn.Dropout(dropout)
        self.dropout_rel_rep = nn.Dropout(dropout)
        self.dropout_rel = nn.Dropout(dropout)
        self.dropout_pred = nn.Dropout(dropout)

        # Union gate
        if use_union:
            self.gate_pred = nn.Linear(hidden_dim * 2, hidden_dim)

        # Fusion + projection
        self.so_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.project_head = MLP(hidden_dim, hidden_dim, hidden_dim * 2, 2)

        # Learnable temperature for cosine similarity
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        # Embeddings (initialized later from GloVe if available)
        self.obj_embed = nn.Embedding(num_classes, embed_dim)
        self.rel_embed = nn.Embedding(num_predicates, embed_dim)

        # Frequency bias
        self.freq_bias = None
        if use_freq_bias:
            self.freq_bias = FrequencyBias(num_predicates, freq_bias_eps)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _fusion(self, sub: torch.Tensor, obj: torch.Tensor) -> torch.Tensor:
        """Fuse subject and object: F(s, o)."""
        return F.relu(self.so_linear(torch.cat([sub, obj], dim=-1)))

    def forward(self, visual_feats: torch.Tensor, boxes: torch.Tensor,
                labels: torch.Tensor, return_obj_preds: bool = False):
        """Forward pass — same interface as all two-stage models.

        Args:
            visual_feats: [N, visual_dim] ROI features.
            boxes: [N, 4] (cx, cy, w, h) normalized.
            labels: [N] object class labels.
            return_obj_preds: If True, also return object predictions.

        Returns:
            dict with rel_logits (cosine similarity), pair_indices, add_losses.
        """
        if self.text_only:
            return self._text_only_forward(visual_feats, boxes, labels)

        N = visual_feats.size(0)
        device = visual_feats.device
        add_losses = {}

        entity_preds = labels  # PredCLS: use GT labels

        # Split visual features: [subject_rep, object_rep]
        entity_rep = self.post_emb(visual_feats)  # [N, H*2]
        entity_rep = entity_rep.view(N, 2, self.hidden_dim)
        sub_rep = entity_rep[:, 1]  # [N, H]
        obj_rep = entity_rep[:, 0]  # [N, H]

        # Word embeddings
        entity_embeds = self.obj_embed(entity_preds)  # [N, embed_dim]

        # Generate pairs
        pairs = generate_object_pairs(N, device)
        if pairs.numel() == 0:
            return {
                "rel_logits": visual_feats.new_zeros(0, self.num_predicates),
                "pair_indices": pairs,
                "sub_boxes": boxes.new_zeros(0, 4),
                "obj_boxes": boxes.new_zeros(0, 4),
                "obj_labels": entity_preds,
                "add_losses": {},
            }

        P = pairs.size(0)
        s_idx = pairs[:, 0]
        o_idx = pairs[:, 1]

        # Semantic prototypes for subject/object
        s_embed = self.W_sub(entity_embeds[s_idx])  # [P, H]
        o_embed = self.W_obj(entity_embeds[o_idx])  # [P, H]

        # Visual to semantic
        sem_sub = self.vis2sem(sub_rep[s_idx])  # [P, H]
        sem_obj = self.vis2sem(obj_rep[o_idx])  # [P, H]

        # Gated fusion
        gate_sem_sub = torch.sigmoid(
            self.gate_sub(torch.cat([s_embed, sem_sub], dim=-1)))
        gate_sem_obj = torch.sigmoid(
            self.gate_obj(torch.cat([o_embed, sem_obj], dim=-1)))

        sub = s_embed + sem_sub * gate_sem_sub
        obj = o_embed + sem_obj * gate_sem_obj

        # Residual + norm for convergence
        sub = self.norm_sub(self.dropout_sub(F.relu(self.linear_sub(sub))) + sub)
        obj = self.norm_obj(self.dropout_obj(F.relu(self.linear_obj(obj))) + obj)

        # Fusion F(s, o)
        fusion_so = self._fusion(sub, obj)  # [P, H]

        # Union feature debiasing (if enabled)
        if self.use_union:
            # Union visual feature: mean of subject and object projected features
            union_vis = (sub_rep[s_idx] + obj_rep[o_idx]) / 2.0  # [P, H]
            sem_pred = self.vis2sem(union_vis)  # h(xu)
            gate_sem_pred = torch.sigmoid(
                self.gate_pred(torch.cat([fusion_so, sem_pred], dim=-1)))
            rel_rep = fusion_so - sem_pred * gate_sem_pred
        else:
            rel_rep = fusion_so

        # Predicate prototypes
        predicate_proto = self.W_pred(self.rel_embed.weight)  # [C, H]

        # Residual + norm
        rel_rep = self.norm_rel_rep(
            self.dropout_rel_rep(F.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        # Project to metric space
        rel_rep = self.project_head(self.dropout_rel(F.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(F.relu(predicate_proto)))

        # Cosine similarity classification
        rel_rep_norm = rel_rep / (rel_rep.norm(dim=1, keepdim=True) + 1e-8)
        pred_proto_norm = predicate_proto / (predicate_proto.norm(dim=1, keepdim=True) + 1e-8)

        rel_logits = rel_rep_norm @ pred_proto_norm.t() * self.logit_scale.exp().clamp(max=100.0)

        # Prototype regularization (training only)
        if self.training:
            # L21: semantic matrix sparsity
            simil_mat = pred_proto_norm @ pred_proto_norm.t()
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_predicates * self.num_predicates)
            add_losses["l21_loss"] = l21

            # Distance loss: keep prototypes well-separated
            gamma2 = 7.0
            proto_a = predicate_proto.unsqueeze(1).expand(-1, self.num_predicates, -1)
            proto_b = predicate_proto.unsqueeze(0).expand(self.num_predicates, -1, -1)
            proto_dis = (proto_a - proto_b).norm(dim=2) ** 2
            sorted_dis, _ = torch.sort(proto_dis, dim=1)
            topK_dis = sorted_dis[:, :2].sum(dim=1) / 1.0
            dist_loss = torch.clamp(-topK_dis + gamma2, min=0).mean()
            add_losses["dist_loss"] = dist_loss

        if self.freq_bias is not None:
            rel_logits = self.freq_bias(rel_logits)

        return {
            "rel_logits": rel_logits,
            "pair_indices": pairs,
            "obj_labels": entity_preds,
            "obj_logits": None,
            "sub_boxes": boxes[pairs[:, 0]],
            "obj_boxes": boxes[pairs[:, 1]],
            "add_losses": add_losses,
        }

    def _text_only_forward(self, visual_feats, boxes, labels):
        """Text-only ablation: no visual features."""
        N = visual_feats.size(0)
        device = visual_feats.device
        entity_preds = labels

        pairs = generate_object_pairs(N, device)
        if pairs.numel() == 0:
            return {
                "rel_logits": visual_feats.new_zeros(0, self.num_predicates),
                "pair_indices": pairs,
                "sub_boxes": boxes.new_zeros(0, 4),
                "obj_boxes": boxes.new_zeros(0, 4),
                "obj_labels": entity_preds,
                "add_losses": {},
            }

        P = pairs.size(0)
        s_idx, o_idx = pairs[:, 0], pairs[:, 1]

        entity_embeds = self.obj_embed(entity_preds)
        s_embed = self.W_sub(entity_embeds[s_idx])
        o_embed = self.W_obj(entity_embeds[o_idx])

        sub = self.norm_sub(self.dropout_sub(F.relu(self.linear_sub(s_embed))) + s_embed)
        obj = self.norm_obj(self.dropout_obj(F.relu(self.linear_obj(o_embed))) + o_embed)

        fusion_so = self._fusion(sub, obj)
        rel_rep = fusion_so

        predicate_proto = self.W_pred(self.rel_embed.weight)

        rel_rep = self.norm_rel_rep(
            self.dropout_rel_rep(F.relu(self.linear_rel_rep(rel_rep))) + rel_rep)
        rel_rep = self.project_head(self.dropout_rel(F.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(F.relu(predicate_proto)))

        rel_rep_norm = rel_rep / (rel_rep.norm(dim=1, keepdim=True) + 1e-8)
        pred_proto_norm = predicate_proto / (predicate_proto.norm(dim=1, keepdim=True) + 1e-8)

        rel_logits = rel_rep_norm @ pred_proto_norm.t() * self.logit_scale.exp().clamp(max=100.0)

        if self.freq_bias is not None:
            rel_logits = self.freq_bias(rel_logits)

        return {
            "rel_logits": rel_logits,
            "pair_indices": pairs,
            "obj_labels": entity_preds,
            "obj_logits": None,
            "sub_boxes": boxes[pairs[:, 0]],
            "obj_boxes": boxes[pairs[:, 1]],
            "add_losses": {},
        }


# ──────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────

def build_react(args) -> REACTModel:
    """Build REACT model from config args."""
    model = REACTModel(
        num_classes=getattr(args, 'entity_nums', 151),
        num_predicates=getattr(args, 'rel_nums', 51),
        visual_dim=getattr(args, 'visual_dim', 2048),
        hidden_dim=getattr(args, 'hidden_dim', 512),
        embed_dim=getattr(args, 'react_embed_dim', 200),
        use_union=getattr(args, 'react_use_union', True),
        text_only=getattr(args, 'react_text_only', False),
        use_freq_bias=getattr(args, 'use_freq_bias', True),
        freq_bias_eps=getattr(args, 'freq_bias_eps', 1e-12),
        dropout=getattr(args, 'dropout', 0.1),
    )
    return model
