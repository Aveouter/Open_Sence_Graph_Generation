"""
SHA-GCL: Stacked Hybrid-Attention and Group Collaborative Learning for SGG.

Reference: "Stacked Hybrid-Attention and Group Collaborative Learning for
Unbiased Scene Graph Generation" (Dong et al., CVPR 2022)

Architecture:
  1. Object Encoding: label embedding + visual projection
  2. Hybrid Attention (stacked layers):
     a. Co-Attention: subject ↔ object cross-modal interaction
     b. Self-Attention: within-modality refinement
  3. Group Collaborative Learning:
     - Predicates split into groups based on frequency
     - Within-group contrastive loss: pull same-group predicates together
     - Cross-group discrimination loss: push different groups apart
  4. Predicate Classification: MLP on refined pair features

Key innovation: hybrid attention combines co-attention and self-attention
for richer relation representation, plus group-based learning for debiasing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .motifs import FrequencyBias, ObjectEncoder, generate_object_pairs


# ──────────────────────────────────────────────
# Co-Attention Layer
# ──────────────────────────────────────────────

class CoAttentionLayer(nn.Module):
    """Co-attention between subject and object streams."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # QKV projections
        self.W_q_sub = nn.Linear(d_model, d_model)
        self.W_k_obj = nn.Linear(d_model, d_model)
        self.W_v_obj = nn.Linear(d_model, d_model)

        self.W_q_obj = nn.Linear(d_model, d_model)
        self.W_k_sub = nn.Linear(d_model, d_model)
        self.W_v_sub = nn.Linear(d_model, d_model)

        self.out_proj_sub = nn.Linear(d_model, d_model)
        self.out_proj_obj = nn.Linear(d_model, d_model)

        self.norm_sub = nn.LayerNorm(d_model)
        self.norm_obj = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.scale = d_model ** 0.5

    def forward(self, sub_feats: torch.Tensor, obj_feats: torch.Tensor):
        """
        Args:
            sub_feats: [P, D] subject features.
            obj_feats: [P, D] object features.

        Returns:
            updated_sub_feats: [P, D]
            updated_obj_feats: [P, D]
        """
        # Subject attends to Object
        q_s = self.W_q_sub(sub_feats)
        k_o = self.W_k_obj(obj_feats)
        v_o = self.W_v_obj(obj_feats)
        attn_s2o = F.softmax((q_s * k_o).sum(-1, keepdim=True) / self.scale, dim=0)
        updated_sub = self.norm_sub(
            sub_feats + self.dropout(self.out_proj_sub(attn_s2o * v_o)))

        # Object attends to Subject
        q_o = self.W_q_obj(obj_feats)
        k_s = self.W_k_sub(sub_feats)
        v_s = self.W_v_sub(sub_feats)
        attn_o2s = F.softmax((q_o * k_s).sum(-1, keepdim=True) / self.scale, dim=0)
        updated_obj = self.norm_obj(
            obj_feats + self.dropout(self.out_proj_obj(attn_o2s * v_s)))

        return updated_sub, updated_obj


# ──────────────────────────────────────────────
# Hybrid Attention Stack
# ──────────────────────────────────────────────

class HybridAttentionStack(nn.Module):
    """Stack of co-attention + self-attention layers."""

    def __init__(self, d_model: int, num_layers: int = 2, nhead: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.co_attn_layers = nn.ModuleList([
            CoAttentionLayer(d_model, dropout) for _ in range(num_layers)
        ])
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, sub_feats: torch.Tensor, obj_feats: torch.Tensor):
        """
        Args:
            sub_feats: [P, D]
            obj_feats: [P, D]

        Returns:
            fused_pair_feats: [P, D]
        """
        for i in range(self.num_layers):
            # Co-attention
            sub_feats, obj_feats = self.co_attn_layers[i](sub_feats, obj_feats)

            # Self-attention on fused features
            fused = sub_feats + obj_feats  # [P, D]
            fused_seq = fused.unsqueeze(0)  # [1, P, D]
            sa_out, _ = self.self_attn_layers[i](fused_seq, fused_seq, fused_seq)
            fused = self.norms[i](fused + self.dropout(sa_out.squeeze(0)))

        return fused


# ──────────────────────────────────────────────
# SHA-GCL Model
# ──────────────────────────────────────────────

class SHAGCLModel(nn.Module):
    """Stacked Hybrid-Attention and Group Collaborative Learning for SGG.

    Args:
        num_classes: Number of object classes.
        num_predicates: Number of predicate classes.
        visual_dim: Input visual feature dimension.
        hidden_dim: Internal hidden dimension.
        num_hybrid_layers: Number of hybrid attention layers.
        nhead: Number of self-attention heads.
        num_groups: Number of predicate groups for GCL.
        gcl_weight: Weight for group collaborative learning loss.
        use_freq_bias: Whether to use frequency bias.
        dropout: Dropout probability.
    """

    def __init__(self, num_classes: int = 151, num_predicates: int = 51,
                 visual_dim: int = 2048, hidden_dim: int = 512,
                 num_hybrid_layers: int = 2, nhead: int = 4,
                 num_groups: int = 3, gcl_weight: float = 0.1,
                 use_freq_bias: bool = True, freq_bias_eps: float = 1e-12,
                 dropout: float = 0.1):
        super().__init__()

        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.gcl_weight = gcl_weight
        self.use_freq_bias = use_freq_bias

        # Object encoding
        self.obj_encoder = ObjectEncoder(visual_dim, num_classes, hidden_dim, dropout)

        # Split object features into subject/object streams
        self.stream_proj = nn.Linear(hidden_dim, hidden_dim)

        # Spatial feature encoding
        self.spatial_proj = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Hybrid attention stack
        self.hybrid_attn = HybridAttentionStack(
            hidden_dim, num_hybrid_layers, nhead, dropout)

        # Group prototypes for GCL
        self.group_prototypes = nn.Parameter(
            torch.randn(num_groups, hidden_dim))
        nn.init.xavier_uniform_(self.group_prototypes)

        # Predicate classifier
        self.pred_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_predicates),
        )

        # Object classifier (for SGCLS)
        self.obj_classifier = nn.Linear(hidden_dim, num_classes)

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

    def _compute_spatial_feat(self, sub_box, obj_box):
        """5-dim spatial features."""
        sx, sy, sw, sh = sub_box[..., 0], sub_box[..., 1], sub_box[..., 2], sub_box[..., 3]
        ox, oy, ow, oh = obj_box[..., 0], obj_box[..., 1], obj_box[..., 2], obj_box[..., 3]
        dx = (sx - ox) / (ow + 1e-6)
        dy = (sy - oy) / (oh + 1e-6)
        dist = torch.sqrt(dx ** 2 + dy ** 2)
        area_ratio = (sw * sh) / (ow * oh + 1e-6)
        s_x1, s_y1 = sx - sw / 2, sy - sh / 2
        s_x2, s_y2 = sx + sw / 2, sy + sh / 2
        o_x1, o_y1 = ox - ow / 2, oy - oh / 2
        o_x2, o_y2 = ox + ow / 2, oy + oh / 2
        inter_w = torch.clamp(torch.min(s_x2, o_x2) - torch.max(s_x1, o_x1), min=0)
        inter_h = torch.clamp(torch.min(s_y2, o_y2) - torch.max(s_y1, o_y1), min=0)
        inter = inter_w * inter_h
        union = sw * sh + ow * oh - inter
        iou = inter / (union + 1e-6)
        return torch.stack([dx, dy, dist, area_ratio, iou], dim=-1)

    def _gcl_loss(self, pair_feats: torch.Tensor, rel_labels: torch.Tensor,
                  group_assignments: torch.Tensor) -> dict:
        """Group Collaborative Learning loss.

        Args:
            pair_feats: [P_valid, D] pair features for annotated pairs.
            rel_labels: [P_valid] GT predicate labels.
            group_assignments: [C] group id for each predicate class (0-based).

        Returns:
            dict with gcl_loss.
        """
        if pair_feats.size(0) == 0:
            return {"gcl_loss": pair_feats.new_tensor(0.0)}

        # Within-group: pull features of same-group predicates together
        # Cross-group: push features of different-group predicates apart
        within_loss = pair_feats.new_tensor(0.0)
        cross_loss = pair_feats.new_tensor(0.0)
        count = 0

        for g in range(self.num_groups):
            group_mask = group_assignments[rel_labels] == g
            if group_mask.sum() < 2:
                continue

            group_feats = pair_feats[group_mask]  # [N_g, D]
            # Within-group: mean distance to group prototype
            proto = self.group_prototypes[g]  # [D]
            within_loss = within_loss + F.mse_loss(group_feats, proto.unsqueeze(0).expand_as(group_feats))
            count += 1

        within_loss = within_loss / max(count, 1)

        # Cross-group: push prototypes apart
        if self.num_groups > 1:
            proto_norm = F.normalize(self.group_prototypes, dim=1)
            cross_sim = proto_norm @ proto_norm.t()  # [G, G]
            # Minimize off-diagonal similarity
            eye = torch.eye(self.num_groups, device=cross_sim.device)
            cross_loss = (cross_sim * (1 - eye)).abs().mean()

        return {"gcl_loss": within_loss + cross_loss * 0.1}

    def forward(self, visual_feats: torch.Tensor, boxes: torch.Tensor,
                labels: torch.Tensor, return_obj_preds: bool = False,
                gt_rel_labels: torch.Tensor = None,
                group_assignments: torch.Tensor = None):
        """
        Args:
            visual_feats: [N, visual_dim] ROI features.
            boxes: [N, 4] (cx, cy, w, h) normalized.
            labels: [N] object class labels.
            return_obj_preds: If True, also return object predictions.
            gt_rel_labels: [P_valid] GT predicate labels for GCL loss.
            group_assignments: [C] group id per predicate class.

        Returns:
            dict with rel_logits, pair_indices, add_losses.
        """
        N = visual_feats.size(0)
        device = visual_feats.device
        add_losses = {}

        # Object encoding
        node_feats = self.obj_encoder(visual_feats, labels)  # [N, H]

        # Generate pairs
        pairs = generate_object_pairs(N, device)
        P = pairs.size(0)

        if P == 0:
            return {
                "rel_logits": visual_feats.new_zeros(0, self.num_predicates),
                "pair_indices": pairs,
                "sub_boxes": boxes.new_zeros(0, 4),
                "obj_boxes": boxes.new_zeros(0, 4),
                "obj_labels": labels,
                "add_losses": {},
            }

        s_idx = pairs[:, 0]
        o_idx = pairs[:, 1]

        # Subject/Object stream features
        sub_feats = self.stream_proj(node_feats[s_idx])  # [P, H]
        obj_feats = self.stream_proj(node_feats[o_idx])  # [P, H]

        # Add spatial encoding
        spatial = self._compute_spatial_feat(boxes[s_idx], boxes[o_idx])
        spatial_enc = self.spatial_proj(spatial)  # [P, H]
        sub_feats = sub_feats + spatial_enc
        obj_feats = obj_feats + spatial_enc

        # Hybrid attention
        pair_feats = self.hybrid_attn(sub_feats, obj_feats)  # [P, H]

        # GCL loss (training only). gt_rel_labels may be aligned to all pairs
        # with -1 for unannotated pairs; keep only annotated predicates.
        if self.training and gt_rel_labels is not None and group_assignments is not None:
            gt_rel_labels = gt_rel_labels.to(pair_feats.device).long()
            group_assignments = group_assignments.to(pair_feats.device).long()
            valid = (gt_rel_labels >= 0) & (gt_rel_labels < group_assignments.numel())
            if valid.any():
                gcl = self._gcl_loss(pair_feats[valid], gt_rel_labels[valid], group_assignments)
            else:
                gcl = {"gcl_loss": pair_feats.new_tensor(0.0)}
            add_losses.update(gcl)

        # Predicate classification
        rel_logits = self.pred_classifier(pair_feats)

        if self.freq_bias is not None:
            rel_logits = self.freq_bias(rel_logits)

        obj_logits = None
        if return_obj_preds:
            obj_logits = self.obj_classifier(node_feats)

        return {
            "rel_logits": rel_logits,
            "pair_indices": pairs,
            "obj_labels": obj_logits.argmax(-1) if obj_logits is not None else labels,
            "obj_logits": obj_logits,
            "sub_boxes": boxes[pairs[:, 0]],
            "obj_boxes": boxes[pairs[:, 1]],
            "add_losses": add_losses,
        }


# ──────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────

def build_shagcl(args) -> SHAGCLModel:
    """Build SHA-GCL model from config args."""
    model = SHAGCLModel(
        num_classes=getattr(args, 'entity_nums', 151),
        num_predicates=getattr(args, 'rel_nums', 51),
        visual_dim=getattr(args, 'visual_dim', 2048),
        hidden_dim=getattr(args, 'hidden_dim', 512),
        num_hybrid_layers=getattr(args, 'shagcl_num_layers', 2),
        nhead=getattr(args, 'shagcl_nhead', 4),
        num_groups=getattr(args, 'shagcl_num_groups', 3),
        gcl_weight=getattr(args, 'shagcl_gcl_weight', 0.1),
        use_freq_bias=getattr(args, 'use_freq_bias', True),
        freq_bias_eps=getattr(args, 'freq_bias_eps', 1e-12),
        dropout=getattr(args, 'dropout', 0.1),
    )
    return model
