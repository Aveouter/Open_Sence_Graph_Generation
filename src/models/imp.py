"""
IMP (Iterative Message Passing) Model for Scene Graph Generation.

Reference: "Scene Graph Generation by Iterative Message Passing" (Xu et al., CVPR 2017)

Architecture:
  1. Object Encoder: Linear projection of visual features + label embeddings
  2. Iterative Message Passing (N iterations):
     a. Edge GRU: fuse(subject_feat, object_feat, union_feat) → edge hidden
     b. Node GRU: aggregate incoming edge messages → node hidden
  3. Predicate Classifier: MLP on edge features
  4. Object Classifier: MLP on node features (for SGCLS)

Supports PredCLS and SGCLS modes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .motifs import FrequencyBias, ObjectEncoder, generate_object_pairs


# ──────────────────────────────────────────────
# IMP Model
# ──────────────────────────────────────────────

class IMPContext(nn.Module):
    """Iterative Message Passing for scene graph generation.

    Args:
        num_classes: Number of object classes (including bg).
        num_predicates: Number of predicate classes.
        visual_dim: Dimension of input visual features per object.
        hidden_dim: Hidden dimension for all internal representations.
        num_iterations: Number of message passing iterations.
        use_freq_bias: Whether to use frequency bias.
        freq_bias_eps: Epsilon for frequency bias computation.
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

        # Reduce visual dim to hidden dim
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Object encoder
        self.obj_encoder = ObjectEncoder(hidden_dim, num_classes, hidden_dim, dropout)

        # Edge GRU: fuses subject and object features
        edge_input_dim = hidden_dim * 2  # subject + object context
        self.edge_gru = nn.GRUCell(edge_input_dim, hidden_dim)

        # Node GRU: aggregates incoming edge messages
        self.node_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.node_input_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Predicate classifier
        self.pred_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_predicates),
        )

        # Object classifier (for SGCLS mode)
        self.obj_classifier = nn.Linear(hidden_dim, num_classes)

        # Frequency bias
        self.freq_bias = None
        if use_freq_bias:
            self.freq_bias = FrequencyBias(num_predicates, freq_bias_eps)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def _compute_pair_features(self, node_feats: torch.Tensor,
                                pair_indices: torch.Tensor) -> torch.Tensor:
        """Compute edge input features from subject+object node features.

        Args:
            node_feats: [N, hidden_dim] node features.
            pair_indices: [P, 2] (subject_idx, object_idx) pairs.

        Returns:
            [P, edge_input_dim] raw edge features (subject+object concat).
        """
        s_idx = pair_indices[:, 0]
        o_idx = pair_indices[:, 1]
        sub_feat = node_feats[s_idx]
        obj_feat = node_feats[o_idx]
        return torch.cat([sub_feat, obj_feat], dim=-1)

    def _aggregate_messages(self, edge_feats: torch.Tensor,
                            pair_indices: torch.Tensor,
                            num_nodes: int) -> torch.Tensor:
        """Aggregate edge messages to each node (mean pooling).

        Args:
            edge_feats: [P, hidden_dim] edge features.
            pair_indices: [P, 2] pair indices.
            num_nodes: N, number of nodes.

        Returns:
            [N, hidden_dim] aggregated messages per node.
        """
        device = edge_feats.device
        aggr = torch.zeros(num_nodes, self.hidden_dim, device=device)
        if edge_feats.numel() == 0:
            return aggr

        receivers = pair_indices[:, 1]
        aggr.index_add_(0, receivers, edge_feats)
        counts = torch.zeros(num_nodes, device=device, dtype=edge_feats.dtype)
        counts.index_add_(0, receivers, torch.ones_like(receivers, dtype=edge_feats.dtype))
        aggr = aggr / counts.clamp_min(1).unsqueeze(1)
        return aggr

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
                - pair_indices: [P, 2] subject/object indices for each pair.
        """
        N = visual_feats.size(0)
        device = visual_feats.device

        # Project visual features
        vis_proj = self.visual_proj(visual_feats)  # [N, hidden_dim]

        # Encode objects
        node_feats = self.obj_encoder(vis_proj, labels)  # [N, hidden_dim]

        # Generate all directed pairs
        pairs = generate_object_pairs(N, device)
        if pairs.numel() == 0:
            return {
                "rel_logits": visual_feats.new_zeros(0, self.num_predicates),
                "pair_indices": pairs,
                "sub_boxes": boxes.new_zeros(0, 4),
                "obj_boxes": boxes.new_zeros(0, 4),
                "obj_labels": labels,
            }

        # Iterative message passing
        edge_feats = None
        for it in range(self.num_iterations):
            # Edge update: GRU over subject+object features
            pair_input = self._compute_pair_features(node_feats, pairs)
            if edge_feats is None:
                edge_feats = self.edge_gru(pair_input)
            else:
                edge_feats = self.edge_gru(pair_input, edge_feats)

            # Node update: aggregate incoming messages
            messages = self._aggregate_messages(edge_feats, pairs, N)
            node_input = self.node_input_proj(
                torch.cat([node_feats, messages], dim=-1))
            node_feats = self.node_gru(node_input, node_feats)

        # Predicate classification
        rel_logits = self.pred_classifier(edge_feats)

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

def build_imp(args) -> IMPContext:
    """Build IMP model from config args."""
    model = IMPContext(
        num_classes=getattr(args, 'entity_nums', 151),
        num_predicates=getattr(args, 'rel_nums', 51),
        visual_dim=getattr(args, 'visual_dim', 2048),
        hidden_dim=getattr(args, 'hidden_dim', 512),
        num_iterations=getattr(args, 'imp_iterations', 2),
        use_freq_bias=getattr(args, 'use_freq_bias', True),
        freq_bias_eps=getattr(args, 'freq_bias_eps', 1e-12),
        dropout=getattr(args, 'dropout', 0.1),
    )
    return model
