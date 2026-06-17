"""
SQUAT: Selective Quad Attention for Scene Graph Generation.

Reference: "Devil's on the Edges: Selective Quad Attention for Scene Graph Generation"
(Jung et al., CVPR 2023)

Architecture:
  1. Object Encoding: visual projection + label embedding
  2. Mask Predictor: selects informative entity pairs (sparsification)
  3. Quad-Attention Decoder:
     a. Edge Self-Attention: edges attend to other edges
     b. Edge-to-Node Attention: edges attend to nodes
     c. Node Self-Attention: nodes attend to other nodes
     d. Node-to-Edge Attention: nodes attend to edges
  4. Predicate Classifier: MLP on refined edge features

Key innovation: four-directional attention with selective sparsification,
capturing both entity-level and pair-level interactions efficiently.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .motifs import FrequencyBias, ObjectEncoder, generate_object_pairs


# ──────────────────────────────────────────────
# Mask Predictor (sparsification)
# ──────────────────────────────────────────────

class MaskPredictor(nn.Module):
    """Predict importance scores for entity pairs to select top-k edges."""

    def __init__(self, in_dim: int, h_dim: int):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1),
        )

    def forward(self, x):
        """Predict importance scores.

        Args:
            x: [P, in_dim] pair features.

        Returns:
            [P] importance scores.
        """
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=0, keepdim=True).expand(z_local.size(0), -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out.squeeze(-1)


# ──────────────────────────────────────────────
# Quad-Attention Decoder Layer
# ──────────────────────────────────────────────

class QuadAttentionLayer(nn.Module):
    """Single quad-attention layer: e2e, e2n, n2e, n2n attention."""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Edge self-attention
        self.self_attn_edge = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        # Node self-attention
        self.self_attn_node = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        # Edge-to-node cross-attention
        self.cross_attn_e2n = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        # Node-to-edge cross-attention
        self.cross_attn_n2e = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)

        # FFN for edges
        self.ffn_edge = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        # FFN for nodes
        self.ffn_node = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        # Layer norms
        self.norm_edge_1 = nn.LayerNorm(d_model)
        self.norm_edge_2 = nn.LayerNorm(d_model)
        self.norm_edge_3 = nn.LayerNorm(d_model)
        self.norm_node_1 = nn.LayerNorm(d_model)
        self.norm_node_2 = nn.LayerNorm(d_model)
        self.norm_node_3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_feats, node_feats):
        """
        Args:
            edge_feats: [1, P, D] edge features.
            node_feats: [1, N, D] node features.

        Returns:
            updated_edge_feats: [1, P, D]
            updated_node_feats: [1, N, D]
        """
        # Edge self-attention
        e_self, _ = self.self_attn_edge(edge_feats, edge_feats, edge_feats)
        edge_feats = self.norm_edge_1(edge_feats + self.dropout(e_self))

        # Edge-to-node cross-attention
        e2n, _ = self.cross_attn_e2n(edge_feats, node_feats, node_feats)
        edge_feats = self.norm_edge_2(edge_feats + self.dropout(e2n))

        # Edge FFN
        edge_feats = self.norm_edge_3(edge_feats + self.dropout(self.ffn_edge(edge_feats)))

        # Node self-attention
        n_self, _ = self.self_attn_node(node_feats, node_feats, node_feats)
        node_feats = self.norm_node_1(node_feats + self.dropout(n_self))

        # Node-to-edge cross-attention
        n2e, _ = self.cross_attn_n2e(node_feats, edge_feats, edge_feats)
        node_feats = self.norm_node_2(node_feats + self.dropout(n2e))

        # Node FFN
        node_feats = self.norm_node_3(node_feats + self.dropout(self.ffn_node(node_feats)))

        return edge_feats, node_feats


# ──────────────────────────────────────────────
# SQUAT Model
# ──────────────────────────────────────────────

class SquatModel(nn.Module):
    """Selective Quad Attention for SGG.

    Args:
        num_classes: Number of object classes.
        num_predicates: Number of predicate classes.
        visual_dim: Input visual feature dimension.
        hidden_dim: Internal hidden dimension.
        num_layers: Number of quad-attention layers.
        nhead: Number of attention heads.
        topk_ratio: Fraction of edges to keep after mask prediction.
        use_freq_bias: Whether to use frequency bias.
        dropout: Dropout probability.
    """

    def __init__(self, num_classes: int = 151, num_predicates: int = 51,
                 visual_dim: int = 2048, hidden_dim: int = 512,
                 num_layers: int = 2, nhead: int = 4,
                 topk_ratio: float = 0.5,
                 use_freq_bias: bool = True, freq_bias_eps: float = 1e-12,
                 dropout: float = 0.1):
        super().__init__()

        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.hidden_dim = hidden_dim
        self.topk_ratio = topk_ratio
        self.use_freq_bias = use_freq_bias

        # Object encoding
        self.obj_encoder = ObjectEncoder(visual_dim, num_classes, hidden_dim, dropout)

        # Pair feature projection
        self.pair_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Mask predictor for edge sparsification
        self.mask_predictor = MaskPredictor(hidden_dim, hidden_dim)

        # Quad-attention layers
        self.attn_layers = nn.ModuleList([
            QuadAttentionLayer(hidden_dim, nhead, dropout)
            for _ in range(num_layers)
        ])

        # Output heads
        self.pred_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_predicates),
        )
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

    def _select_topk_edges(self, edge_feats: torch.Tensor,
                            scores: torch.Tensor,
                            pair_indices: torch.Tensor) -> tuple:
        """Select top-k edges based on mask predictor scores.

        Args:
            edge_feats: [P, D] all edge features.
            scores: [P] importance scores.
            pair_indices: [P, 2] pair indices.

        Returns:
            selected_feats: [K, D]
            selected_indices: [K, 2]
            selected_mask: [P] boolean mask of selected edges
        """
        P = edge_feats.size(0)
        K = max(1, int(P * self.topk_ratio))

        _, topk_idx = torch.topk(scores, K, dim=0)
        selected_feats = edge_feats[topk_idx]
        selected_indices = pair_indices[topk_idx]
        selected_mask = torch.zeros(P, dtype=torch.bool, device=edge_feats.device)
        selected_mask[topk_idx] = True

        return selected_feats, selected_indices, selected_mask

    def _scatter_edges(self, sparse_feats: torch.Tensor,
                        sparse_indices: torch.Tensor,
                        num_edges: int) -> torch.Tensor:
        """Scatter sparse edge features back to full edge tensor.

        Args:
            sparse_feats: [K, D] selected edge features.
            sparse_indices: [K] indices into full edge tensor.
            num_edges: P, total number of edges.

        Returns:
            [P, D] full edge tensor with selected features filled in.
        """
        full = torch.zeros(num_edges, self.hidden_dim,
                           device=sparse_feats.device, dtype=sparse_feats.dtype)
        full[sparse_indices] = sparse_feats
        return full

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

        # Object encoding
        node_feats = self.obj_encoder(visual_feats, labels)  # [N, H]

        # Generate all pairs
        pairs = generate_object_pairs(N, device)
        P = pairs.size(0)

        if P == 0:
            return {
                "rel_logits": visual_feats.new_zeros(0, self.num_predicates),
                "pair_indices": pairs,
                "sub_boxes": boxes.new_zeros(0, 4),
                "obj_boxes": boxes.new_zeros(0, 4),
                "obj_labels": labels,
            }

        # Initial edge features
        s_idx = pairs[:, 0]
        o_idx = pairs[:, 1]
        edge_feats = self.pair_proj(
            torch.cat([node_feats[s_idx], node_feats[o_idx]], dim=-1))  # [P, H]

        # Edge sparsification
        mask_scores = self.mask_predictor(edge_feats)
        sparse_edges, sparse_indices, selected_mask = self._select_topk_edges(
            edge_feats, mask_scores, pairs)

        # Get sparse indices for scattering back
        sparse_idx = selected_mask.nonzero(as_tuple=True)[0]

        # Quad-attention layers
        node_feats_seq = node_feats.unsqueeze(0)  # [1, N, H]
        sparse_edges_seq = sparse_edges.unsqueeze(0)  # [1, K, H]

        for layer in self.attn_layers:
            sparse_edges_seq, _ = layer(sparse_edges_seq, node_feats_seq)

        sparse_edges = sparse_edges_seq.squeeze(0)  # [K, H]

        # Scatter sparse-edge updates back into the original edge features.
        # Non-selected edges keep their original features.
        edge_feats[sparse_idx] = sparse_edges

        # Predicate classification
        rel_logits = self.pred_classifier(edge_feats)

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
        }


# ──────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────

def build_squat(args) -> SquatModel:
    """Build SQUAT model from config args."""
    model = SquatModel(
        num_classes=getattr(args, 'entity_nums', 151),
        num_predicates=getattr(args, 'rel_nums', 51),
        visual_dim=getattr(args, 'visual_dim', 2048),
        hidden_dim=getattr(args, 'hidden_dim', 512),
        num_layers=getattr(args, 'squat_num_layers', 2),
        nhead=getattr(args, 'squat_nhead', 4),
        topk_ratio=getattr(args, 'squat_topk_ratio', 0.5),
        use_freq_bias=getattr(args, 'use_freq_bias', True),
        freq_bias_eps=getattr(args, 'freq_bias_eps', 1e-12),
        dropout=getattr(args, 'dropout', 0.1),
    )
    return model
