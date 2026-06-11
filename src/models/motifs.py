"""
Motifs Model for Scene Graph Generation.

Reference: "Neural Motifs: Scene Graph Parsing with Global Context" (Zellers et al., CVPR 2018)

Architecture:
  1. Object Encoder: Linear projection of visual features + label embeddings
  2. Object Context: BiLSTM over sorted object sequence
  3. Edge Context: BiLSTM over ordered pair sequence
  4. Predicate Classifier: MLP on pair features + context

Supports PredCLS (GT boxes + GT labels) and SGCLS (GT boxes only) modes.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ──────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────


def generate_object_pairs(N: int, device: torch.device) -> torch.Tensor:
    """Generate all directed object pairs (i → j, i ≠ j).

    Returns:
        [N*(N-1), 2] tensor of (subject_idx, object_idx) pairs.
    """
    idx = torch.arange(N, device=device)
    grid_i, grid_j = torch.meshgrid(idx, idx, indexing='ij')
    mask = grid_i != grid_j
    return torch.stack([grid_i[mask], grid_j[mask]], dim=-1)


# ──────────────────────────────────────────────
# Utility layers
# ──────────────────────────────────────────────

class FrequencyBias(nn.Module):
    """Learnable frequency prior for predicate prediction.

    The bias is initialized from training set co-occurrence statistics
    and added to the predicate logits before softmax.
    """

    def __init__(self, num_predicates: int, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(num_predicates), requires_grad=False)

    def load_freq_bias(self, distribution: torch.Tensor):
        """Load pre-computed frequency distribution.

        Args:
            distribution: [num_predicates] float tensor of marginal predicate frequencies.
        """
        bias = torch.log(distribution + self.eps)
        self.bias.copy_(bias)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Add frequency bias to logits."""
        return logits + self.bias.to(logits.device)


class ObjectEncoder(nn.Module):
    """Encode object-level features from visual features + semantic embeddings.

    Args:
        visual_dim: Dimension of visual features (ROI features from backbone).
        num_classes: Number of object classes (including bg, e.g. 151).
        hidden_dim: Output feature dimension.
        dropout: Dropout probability.
    """

    def __init__(self, visual_dim: int, num_classes: int, hidden_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, hidden_dim // 2)
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_feats: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feats: [N, visual_dim] ROI features.
            labels: [N] object class labels (int).

        Returns:
            [N, hidden_dim] encoded object features.
        """
        v = self.visual_proj(visual_feats)
        s = self.label_embed(labels)
        feat = torch.cat([v, s], dim=-1)
        return self.dropout(feat)


class BiLSTMContext(nn.Module):
    """Bidirectional LSTM for context encoding over a sequence of objects/pairs."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, input_dim] sequence features.

        Returns:
            [B, N, hidden_dim] context-encoded features.
        """
        out, _ = self.lstm(x)
        return out


class PairFeatureGenerator(nn.Module):
    """Generate features for object pairs from object-level features.

    For each pair (i, j):
      - Subject feature o_i
      - Object feature o_j
      - Union box feature: [cx, cy, w, h, area] of the union box
      - Spatial feature: [dx, dy, dist, area_ratio, iou]
    """

    def __init__(self, obj_feat_dim: int, hidden_dim: int = 512):
        super().__init__()
        spatial_dim = 5  # relative position features
        union_dim = 5     # union box features
        total_dim = obj_feat_dim * 2 + spatial_dim + union_dim
        self.proj = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.hidden_dim = hidden_dim

    def _compute_spatial_feat(self, sub_box: torch.Tensor,
                               obj_box: torch.Tensor) -> torch.Tensor:
        """Compute spatial features between two boxes.

        Args:
            sub_box, obj_box: [..., 4] in (cx, cy, w, h) normalized.

        Returns:
            [..., 5] spatial features.
        """
        sx, sy, sw, sh = sub_box[..., 0], sub_box[..., 1], sub_box[..., 2], sub_box[..., 3]
        ox, oy, ow, oh = obj_box[..., 0], obj_box[..., 1], obj_box[..., 2], obj_box[..., 3]

        dx = (sx - ox) / (ow + 1e-6)
        dy = (sy - oy) / (oh + 1e-6)
        dist = torch.sqrt(dx ** 2 + dy ** 2)
        area_ratio = (sw * sh) / (ow * oh + 1e-6)
        # Approximate IoU
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

    def _compute_union_feat(self, sub_box: torch.Tensor,
                             obj_box: torch.Tensor) -> torch.Tensor:
        """Compute union box features."""
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
        u_cx = (u_x1 + u_x2) / 2
        u_cy = (u_y1 + u_y2) / 2
        u_w = u_x2 - u_x1
        u_h = u_y2 - u_y1
        u_area = u_w * u_h
        return torch.stack([u_cx, u_cy, u_w, u_h, u_area], dim=-1)

    def forward(self, obj_feats: torch.Tensor, boxes: torch.Tensor,
                pair_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obj_feats: [N, D] object features.
            boxes: [N, 4] object boxes (cx, cy, w, h) normalized.
            pair_indices: [P, 2] indices of (subject, object) pairs.

        Returns:
            [P, hidden_dim] pair features.
        """
        s_idx = pair_indices[:, 0]
        o_idx = pair_indices[:, 1]

        sub_feat = obj_feats[s_idx]
        obj_feat = obj_feats[o_idx]
        sub_box = boxes[s_idx]
        obj_box = boxes[o_idx]

        spatial = self._compute_spatial_feat(sub_box, obj_box)
        union = self._compute_union_feat(sub_box, obj_box)

        pair_feat = torch.cat([sub_feat, obj_feat, spatial, union], dim=-1)
        return self.proj(pair_feat)


# ──────────────────────────────────────────────
# Motifs Model
# ──────────────────────────────────────────────

class MotifsModel(nn.Module):
    """Neural Motifs model for scene graph generation.

    Args:
        num_classes: Number of object classes (including bg).
        num_predicates: Number of predicate classes.
        visual_dim: Dimension of input visual features per object.
        hidden_dim: Hidden dimension for all internal representations.
        obj_lstm_layers: Number of BiLSTM layers for object context.
        edge_lstm_layers: Number of BiLSTM layers for edge context.
        use_freq_bias: Whether to use frequency bias.
        freq_bias_eps: Epsilon for frequency bias computation.
        dropout: Dropout probability.
    """

    def __init__(self, num_classes: int = 151, num_predicates: int = 51,
                 visual_dim: int = 2048, hidden_dim: int = 512,
                 obj_lstm_layers: int = 1, edge_lstm_layers: int = 1,
                 use_freq_bias: bool = True, freq_bias_eps: float = 1e-12,
                 dropout: float = 0.1):
        super().__init__()

        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.hidden_dim = hidden_dim
        self.use_freq_bias = use_freq_bias

        # Object encoding
        self.obj_encoder = ObjectEncoder(visual_dim, num_classes, hidden_dim, dropout)
        self.obj_context = BiLSTMContext(hidden_dim, hidden_dim, obj_lstm_layers, dropout)
        self.obj_post = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Pair feature generation
        self.pair_gen = PairFeatureGenerator(hidden_dim, hidden_dim)

        # Edge context
        self.edge_context = BiLSTMContext(hidden_dim, hidden_dim, edge_lstm_layers, dropout)
        self.edge_post = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Predicate classifier
        self.pred_classifier = nn.Linear(hidden_dim, num_predicates)

        # Frequency bias
        self.freq_bias = None
        if use_freq_bias:
            self.freq_bias = FrequencyBias(num_predicates, freq_bias_eps)

        # Object classifier (for SGCLS mode)
        self.obj_classifier = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def _get_motif_order(self, labels: torch.Tensor,
                          boxes: torch.Tensor) -> torch.Tensor:
        """Sort objects by label frequency (motif ordering).

        The paper sorts objects by category frequency to create
        a canonical ordering that exposes co-occurrence patterns.
        Here we sort by (label, cx) to get a deterministic order.
        """
        # Sort by label first, then by x-center position
        _, indices = torch.sort(labels.float() * 1000 + boxes[:, 0])
        return indices

    def _generate_pairs(self, N: int, device: torch.device) -> torch.Tensor:
        """Generate all directed object pairs (i → j, i ≠ j)."""
        return generate_object_pairs(N, device)

    def forward(self, visual_feats: torch.Tensor, boxes: torch.Tensor,
                labels: torch.Tensor, return_obj_preds: bool = False):
        """
        Args:
            visual_feats: [N, visual_dim] ROI features per object.
            boxes: [N, 4] object boxes (cx, cy, w, h) normalized.
            labels: [N] object class labels (int, 0=bg).
            return_obj_preds: If True, also return object label predictions.

        Returns:
            dict with keys:
                - rel_logits: [P, num_predicates] predicate logits.
                - obj_labels: [N] object predictions (if return_obj_preds).
                - pair_indices: [P, 2] subject/object indices for each pair.
        """
        N = visual_feats.size(0)
        device = visual_feats.device

        # ── Object encoding with context ──
        sort_idx = self._get_motif_order(labels, boxes)
        unsort_idx = torch.argsort(sort_idx)

        obj_feats = self.obj_encoder(visual_feats[sort_idx], labels[sort_idx])
        obj_feats = self.obj_context(obj_feats.unsqueeze(0)).squeeze(0)  # [N, H]
        obj_feats = self.obj_post(obj_feats)
        obj_feats = obj_feats[unsort_idx]  # restore original order

        # ── Object classification (for SGCLS) ──
        obj_logits = None
        if return_obj_preds:
            obj_logits = self.obj_classifier(obj_feats)

        # ── Generate pair features ──
        pairs = self._generate_pairs(N, device)  # [P, 2]
        if pairs.numel() == 0:
            return {
                "rel_logits": visual_feats.new_zeros(0, self.num_predicates),
                "obj_labels": obj_logits.argmax(-1) if obj_logits is not None else labels,
                "pair_indices": pairs,
                "sub_boxes": boxes.new_zeros(0, 4),
                "obj_boxes": boxes.new_zeros(0, 4),
            }

        pair_feats = self.pair_gen(obj_feats, boxes, pairs)

        # ── Edge context: sort pairs for canonical ordering ──
        # Sort by (subject_label, object_label) for context
        s_labels = labels[pairs[:, 0]]
        o_labels = labels[pairs[:, 1]]
        pair_sort_key = s_labels.float() * 1000 + o_labels.float()
        pair_sort_idx = torch.argsort(pair_sort_key)
        pair_unsort_idx = torch.argsort(pair_sort_idx)

        pair_feats_sorted = pair_feats[pair_sort_idx]
        edge_feats = self.edge_context(pair_feats_sorted.unsqueeze(0)).squeeze(0)
        edge_feats = self.edge_post(edge_feats)
        edge_feats = edge_feats[pair_unsort_idx]  # restore order

        # ── Predicate classification ──
        rel_logits = self.pred_classifier(edge_feats)

        if self.freq_bias is not None:
            rel_logits = self.freq_bias(rel_logits)

        return {
            "rel_logits": rel_logits,
            "obj_logits": obj_logits,
            "obj_labels": obj_logits.argmax(-1) if obj_logits is not None else labels,
            "pair_indices": pairs,
            "sub_boxes": boxes[pairs[:, 0]],
            "obj_boxes": boxes[pairs[:, 1]],
        }


# ──────────────────────────────────────────────
# TDE Model (Total Direct Effect)
# ──────────────────────────────────────────────

class TDEModel(MotifsModel):
    """TDE: Total Direct Effect for Scene Graph Generation.

    Reference: "Unbiased Scene Graph Generation from Biased Training" (Tang et al., CVPR 2020)

    Extends MotifsModel with causal intervention:
      P_final = softmax(logits_factual - logits_counterfactual)

    The counterfactual replaces visual features with their dataset mean,
    removing the "direct effect" of visual appearance and leaving only
    the context bias, which is then subtracted.
    """

    def __init__(self, num_classes: int = 151, num_predicates: int = 51,
                 visual_dim: int = 2048, hidden_dim: int = 512,
                 obj_lstm_layers: int = 1, edge_lstm_layers: int = 1,
                 use_freq_bias: bool = True, freq_bias_eps: float = 1e-12,
                 dropout: float = 0.1,
                 tde_fusion: str = "subtract"):
        super().__init__(num_classes, num_predicates, visual_dim, hidden_dim,
                         obj_lstm_layers, edge_lstm_layers,
                         use_freq_bias, freq_bias_eps, dropout)
        self.tde_fusion = tde_fusion
        self.register_buffer("mean_visual_feat", torch.zeros(visual_dim))

    def set_mean_visual_feat(self, mean_feat: torch.Tensor):
        """Set the mean visual feature for counterfactual intervention."""
        self.mean_visual_feat.copy_(mean_feat)

    def forward_counterfactual(self, boxes: torch.Tensor, labels: torch.Tensor):
        """Forward pass with mean visual features (counterfactual)."""
        N = boxes.size(0)
        device = boxes.device
        visual_feats = self.mean_visual_feat.unsqueeze(0).expand(N, -1).to(device)
        return super().forward(visual_feats, boxes, labels)

    def forward(self, visual_feats: torch.Tensor, boxes: torch.Tensor,
                labels: torch.Tensor, return_obj_preds: bool = False,
                apply_tde: bool = True):
        """Forward pass with optional TDE intervention.

        Args:
            apply_tde: If True, apply causal intervention (only during inference).
        """
        # Factual prediction
        factual = super().forward(visual_feats, boxes, labels, return_obj_preds)

        if not apply_tde or self.training:
            return factual

        # Counterfactual prediction
        counterfactual = self.forward_counterfactual(boxes, labels)

        # TDE fusion
        if self.tde_fusion == "subtract":
            # P = softmax(logits_f - logits_c)
            factual["rel_logits"] = factual["rel_logits"] - counterfactual["rel_logits"]
        elif self.tde_fusion == "softmax_subtract":
            # P = softmax(logits_f) - softmax(logits_c)
            p_f = F.softmax(factual["rel_logits"], dim=-1)
            p_c = F.softmax(counterfactual["rel_logits"], dim=-1)
            factual["rel_logits"] = torch.log(torch.clamp(p_f - p_c, min=1e-8))

        return factual


# ──────────────────────────────────────────────
# Builder functions
# ──────────────────────────────────────────────

def build_motifs(args) -> MotifsModel:
    """Build Motifs model from config args."""
    model = MotifsModel(
        num_classes=getattr(args, 'entity_nums', 151),
        num_predicates=getattr(args, 'rel_nums', 51),
        visual_dim=getattr(args, 'visual_dim', 2048),
        hidden_dim=getattr(args, 'hidden_dim', 512),
        obj_lstm_layers=getattr(args, 'obj_lstm_layers', 1),
        edge_lstm_layers=getattr(args, 'edge_lstm_layers', 1),
        use_freq_bias=getattr(args, 'use_freq_bias', True),
        freq_bias_eps=getattr(args, 'freq_bias_eps', 1e-12),
        dropout=getattr(args, 'dropout', 0.1),
    )
    return model


def build_tde(args) -> TDEModel:
    """Build TDE model from config args."""
    model = TDEModel(
        num_classes=getattr(args, 'entity_nums', 151),
        num_predicates=getattr(args, 'rel_nums', 51),
        visual_dim=getattr(args, 'visual_dim', 2048),
        hidden_dim=getattr(args, 'hidden_dim', 512),
        obj_lstm_layers=getattr(args, 'obj_lstm_layers', 1),
        edge_lstm_layers=getattr(args, 'edge_lstm_layers', 1),
        use_freq_bias=getattr(args, 'use_freq_bias', True),
        freq_bias_eps=getattr(args, 'freq_bias_eps', 1e-12),
        dropout=getattr(args, 'dropout', 0.1),
        tde_fusion=getattr(args, 'tde_fusion', 'subtract'),
    )
    return model
