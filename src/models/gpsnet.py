"""
GPS-Net: Graph Property Sensing Network for Scene Graph Generation.

Reference: "GPS-Net: Graph Property Sensing Network for Scene Graph Generation"
(Lin et al., CVPR 2020)

Architecture:
  1. Object Encoder: Linear projection + label embedding + position encoding
  2. Node Attention (property-aware gating): select which nodes to propagate
  3. Iterative Message Passing:
     a. Edge Gating: directional attention over subject+object+union features
     b. Bidirectional Message: subject→object and object→subject
     c. Node Update: accumulate messages with attention, update via GRU
  4. Predicate Classifier: MLP on edge features

Key innovation: directional edge gating + bidirectional message passing
that respects the asymmetric nature of visual relationships.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .motifs import FrequencyBias, ObjectEncoder, generate_object_pairs


# ──────────────────────────────────────────────
# Gating Model (edge attention)
# ──────────────────────────────────────────────

class GatingModel(nn.Module):
    """Directional attention gating for edge features.

    Computes attention scores based on subject, object, and relation features.
    """

    def __init__(self, entity_input_dim: int, union_input_dim: int,
                 hidden_dim: int, filter_dim: int = 32):
        super().__init__()
        self.ws = nn.Sequential(
            nn.Linear(entity_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.wo = nn.Sequential(
            nn.Linear(entity_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.wu = nn.Sequential(
            nn.Linear(union_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.w = nn.Sequential(
            nn.Linear(hidden_dim, filter_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, subj_feat: torch.Tensor, obj_feat: torch.Tensor,
                rel_feat: torch.Tensor) -> torch.Tensor:
        """Compute attention gate values.

        Args:
            subj_feat: [P, D_e] subject features.
            obj_feat: [P, D_e] object features.
            rel_feat: [P, D_u] relation/union features.

        Returns:
            [P] attention scores.
        """
        prod = self.ws(subj_feat) * self.wo(obj_feat) * self.wu(rel_feat)
        attn = self.w(prod)
        if attn.shape[1] > 1:
            attn = attn.mean(1)
        return attn.squeeze(-1)


# ──────────────────────────────────────────────
# Message Generator (bidirectional)
# ──────────────────────────────────────────────

class MessageGenerator(nn.Module):
    """Bidirectional message passing with attention-weighted aggregation."""

    def __init__(self, input_dims: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dims = input_dims

        self.output_fc = nn.Sequential(
            nn.Linear(input_dims, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, source_features: torch.Tensor,
                weighting_gate: torch.Tensor,
                rel_pair_idx: torch.Tensor) -> torch.Tensor:
        """Aggregate messages from neighbors with attention.

        Args:
            source_features: [N, D] node features.
            weighting_gate: [P] edge attention scores.
            rel_pair_idx: [P, 2] (subject, object) pair indices.

        Returns:
            [N, hidden_dim] aggregated messages per node.
        """
        N = source_features.size(0)
        device = source_features.device

        # Build attention matrix (masked softmax)
        attn_mat = torch.zeros(N, N, device=device)
        attn_mat[rel_pair_idx[:, 0], rel_pair_idx[:, 1]] = weighting_gate
        attn_max = attn_mat.max()
        attn_mat = (attn_mat - attn_max).exp()
        mask = torch.zeros(N, N, device=device)
        mask[rel_pair_idx[:, 0], rel_pair_idx[:, 1]] = 1.0
        attn_mat = attn_mat * mask
        row_sum = attn_mat.sum(1, keepdim=True) + 1e-6
        attn_mat = attn_mat / row_sum

        # Bidirectional: subject→object AND object→subject
        attn_mat_t = attn_mat.transpose(0, 1)
        attn_bidi = torch.stack([attn_mat, attn_mat_t], dim=-1)  # [N, N, 2]

        # Aggregate messages from both directions
        msg_fwd = torch.mm(attn_bidi[:, :, 0], source_features)  # [N, D]
        msg_bwd = torch.mm(attn_bidi[:, :, 1], source_features)  # [N, D]
        received = self.output_fc(msg_fwd + msg_bwd)  # [N, hidden_dim]

        return received


# ──────────────────────────────────────────────
# GPS-Net Context
# ──────────────────────────────────────────────

class GPSNetContext(nn.Module):
    """Graph Property Sensing Network for SGG.

    Args:
        num_classes: Number of object classes.
        num_predicates: Number of predicate classes.
        visual_dim: Input visual feature dimension.
        hidden_dim: Internal hidden dimension.
        num_iterations: Number of message passing rounds.
        use_freq_bias: Whether to use frequency bias.
        dropout: Dropout probability.
    """

    def __init__(self, num_classes: int = 151, num_predicates: int = 51,
                 visual_dim: int = 2048, hidden_dim: int = 512,
                 num_iterations: int = 2,
                 use_freq_bias: bool = True, freq_bias_eps: float = 1e-12,
                 dropout: float = 0.1):
        super().__init__()

        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.use_freq_bias = use_freq_bias

        # Position embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(9, 32),
            nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
        )

        # Object encoding
        self.obj_encoder = ObjectEncoder(
            visual_dim + 128, num_classes, hidden_dim, dropout)

        # Node-wise gating (which nodes participate)
        self.node_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Spatial feature dim for pairs
        spatial_dim = 5
        union_dim = 5

        # Pair feature projection
        self.pair_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + spatial_dim + union_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Edge gating
        self.edge_gate = GatingModel(hidden_dim, hidden_dim, hidden_dim)

        # Message generator
        self.msg_gen = MessageGenerator(hidden_dim, hidden_dim)

        # Node update GRU
        self.node_gru = nn.GRUCell(hidden_dim, hidden_dim)

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

    def _encode_box_info(self, boxes: torch.Tensor) -> torch.Tensor:
        """Encode box geometry: 9-dim feature.

        Args:
            boxes: [N, 4] (cx, cy, w, h) normalized.

        Returns:
            [N, 9] box geometry features.
        """
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        area = w * h
        return torch.stack([cx, cy, w, h, x1, y1, x2, y2, area], dim=-1)

    def _compute_spatial_feat(self, sub_box: torch.Tensor,
                               obj_box: torch.Tensor) -> torch.Tensor:
        """Compute 5-dim spatial features between box pairs."""
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

    def _compute_union_box_feat(self, sub_box: torch.Tensor,
                                  obj_box: torch.Tensor) -> torch.Tensor:
        """Compute 5-dim union box features."""
        sx, sy, sw, sh = sub_box[..., 0], sub_box[..., 1], sub_box[..., 2], sub_box[..., 3]
        ox, oy, ow, oh = obj_box[..., 0], obj_box[..., 1], obj_box[..., 2], obj_box[..., 3]
        s_x1, s_y1 = sx - sw / 2, sy - sh / 2
        s_x2, s_y2 = sx + sw / 2, sy + sh / 2
        o_x1, o_y1 = ox - ow / 2, oy - oh / 2
        o_x2, o_y2 = ox + ow / 2, oy + oh / 2
        u_x1 = torch.min(s_x1, o_x1)
        u_y1 = torch.min(s_y1, o_y1)
        u_x2 = torch.max(s_x2, o_x2)
        u_y2 = torch.max(s_y2, o_y2)
        return torch.stack([
            (u_x1 + u_x2) / 2, (u_y1 + u_y2) / 2,
            u_x2 - u_x1, u_y2 - u_y1,
            (u_x2 - u_x1) * (u_y2 - u_y1),
        ], dim=-1)

    def forward(self, visual_feats: torch.Tensor, boxes: torch.Tensor,
                labels: torch.Tensor, return_obj_preds: bool = False):
        """
        Args:
            visual_feats: [N, visual_dim] ROI features.
            boxes: [N, 4] (cx, cy, w, h) normalized.
            labels: [N] object class labels.
            return_obj_preds: If True, also return object predictions.

        Returns:
            dict with rel_logits, pair_indices, etc.
        """
        N = visual_feats.size(0)
        device = visual_feats.device

        # Position encoding
        box_info = self._encode_box_info(boxes)
        pos_emb = self.pos_embed(box_info)  # [N, 128]

        # Concatenate visual + position
        vis_with_pos = torch.cat([visual_feats, pos_emb], dim=-1)  # [N, D+128]

        # Object encoding
        node_feats = self.obj_encoder(vis_with_pos, labels)  # [N, H]

        # Node attention gating
        node_gates = self.node_gate(node_feats)  # [N, 1]

        # Generate all pairs
        pairs = generate_object_pairs(N, device)  # [P, 2]
        if pairs.numel() == 0:
            return {
                "rel_logits": visual_feats.new_zeros(0, self.num_predicates),
                "pair_indices": pairs,
                "sub_boxes": boxes.new_zeros(0, 4),
                "obj_boxes": boxes.new_zeros(0, 4),
                "obj_labels": labels,
            }

        P = pairs.size(0)

        # Pre-compute spatial features (reused across iterations)
        sub_box = boxes[pairs[:, 0]]
        obj_box = boxes[pairs[:, 1]]
        spatial_feat = self._compute_spatial_feat(sub_box, obj_box)
        union_box_feat = self._compute_union_box_feat(sub_box, obj_box)

        # Iterative message passing
        for it in range(self.num_iterations):
            # Build pair features
            sub_feat = node_feats[pairs[:, 0]]
            obj_feat = node_feats[pairs[:, 1]]
            pair_feat = self.pair_proj(
                torch.cat([sub_feat, obj_feat, spatial_feat, union_box_feat], dim=-1))

            # Edge gating
            edge_gates = self.edge_gate(sub_feat, obj_feat, pair_feat)  # [P]

            # Message passing
            messages = self.msg_gen(node_feats, edge_gates, pairs)  # [N, H]

            # Node update with gating
            node_input = messages * node_gates
            node_feats = self.node_gru(node_input, node_feats)

        # Final pair features for prediction
        sub_feat_final = node_feats[pairs[:, 0]]
        obj_feat_final = node_feats[pairs[:, 1]]
        pair_feat_final = self.pair_proj(
            torch.cat([sub_feat_final, obj_feat_final, spatial_feat, union_box_feat], dim=-1))

        # Predicate classification
        rel_logits = self.pred_classifier(pair_feat_final)

        if self.freq_bias is not None:
            rel_logits = self.freq_bias(rel_logits)

        # Object classification (for SGCLS)
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
        }


# ──────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────

def build_gpsnet(args) -> GPSNetContext:
    """Build GPS-Net model from config args."""
    model = GPSNetContext(
        num_classes=getattr(args, 'entity_nums', 151),
        num_predicates=getattr(args, 'rel_nums', 51),
        visual_dim=getattr(args, 'visual_dim', 2048),
        hidden_dim=getattr(args, 'hidden_dim', 512),
        num_iterations=getattr(args, 'gpsnet_iterations', 2),
        use_freq_bias=getattr(args, 'use_freq_bias', True),
        freq_bias_eps=getattr(args, 'freq_bias_eps', 1e-12),
        dropout=getattr(args, 'dropout', 0.1),
    )
    return model
