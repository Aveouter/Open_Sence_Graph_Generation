"""
Transformer Predictor for Scene Graph Generation.

Reference: "Unbiased Scene Graph Generation from Biased Training" (Tang et al., CVPR 2020)

Architecture:
  1. Object Encoder: Linear projection + label embedding
  2. Object Transformer: Multi-head Self-Attention over objects
  3. Pair Generation: concat(subject, object, spatial, union) features
  4. Edge Transformer: Multi-head Self-Attention over pairs
  5. Predicate Classifier: MLP on edge features

Uses self-attention instead of BiLSTM for context modeling,
which captures longer-range dependencies.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .motifs import FrequencyBias, ObjectEncoder, PairFeatureGenerator, generate_object_pairs


# ──────────────────────────────────────────────
# Transformer utilities (custom, no external dep)
# ──────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        self._init_weights()

    def _init_weights(self):
        for m in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            query: [B, L_q, D]
            key: [B, L_k, D]
            value: [B, L_v, D]
            mask: [B, L_q, L_k] or None (True = masked positions)

        Returns:
            [B, L_q, D], [B, L_q, L_k] attention weights
        """
        B = query.size(0)
        L_q, L_k, L_v = query.size(1), key.size(1), value.size(1)

        q = self.w_q(query).view(B, L_q, self.nhead, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(B, L_k, self.nhead, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(B, L_v, self.nhead, self.d_k).transpose(1, 2)

        # [B, nhead, L_q, L_k]
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            # mask: [B, L_q, L_k] → [B, 1, L_q, L_k]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, nhead, L_q, d_k]
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        out = self.w_o(out)
        return out, attn_weights


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer: Self-Attn + FFN."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # FFN
        ffn_out = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout3(ffn_out))

        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(self, d_model: int, nhead: int, num_layers: int = 2,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ──────────────────────────────────────────────
# Transformer SGG Model
# ──────────────────────────────────────────────

class TransformerSGGModel(nn.Module):
    """Transformer-based scene graph generation model.

    Replaces BiLSTM context with multi-head self-attention.

    Args:
        num_classes: Number of object classes (including bg).
        num_predicates: Number of predicate classes.
        visual_dim: Dimension of input visual features per object.
        hidden_dim: Hidden dimension for all internal representations.
        nhead: Number of attention heads.
        num_encoder_layers: Number of transformer layers for object/edge context.
        dim_feedforward: Feed-forward dimension in transformer.
        use_freq_bias: Whether to use frequency bias.
        dropout: Dropout probability.
    """

    def __init__(self, num_classes: int = 151, num_predicates: int = 51,
                 visual_dim: int = 2048, hidden_dim: int = 512,
                 nhead: int = 8, num_encoder_layers: int = 2,
                 dim_feedforward: int = 2048,
                 use_freq_bias: bool = True, freq_bias_eps: float = 1e-12,
                 dropout: float = 0.1):
        super().__init__()

        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.hidden_dim = hidden_dim
        self.use_freq_bias = use_freq_bias

        # Object encoding
        self.obj_encoder = ObjectEncoder(visual_dim, num_classes, hidden_dim, dropout)

        # Object transformer context
        self.obj_transformer = TransformerEncoder(
            hidden_dim, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.obj_post = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Pair feature generation
        self.pair_gen = PairFeatureGenerator(hidden_dim, hidden_dim)

        # Edge transformer context
        self.edge_transformer = TransformerEncoder(
            hidden_dim, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.edge_post = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Predicate classifier
        self.pred_classifier = nn.Linear(hidden_dim, num_predicates)

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

    def forward(self, visual_feats: torch.Tensor, boxes: torch.Tensor,
                labels: torch.Tensor, return_obj_preds: bool = False):
        """
        Args:
            visual_feats: [N, visual_dim] ROI features per object.
            boxes: [N, 4] object boxes (cx, cy, w, h) normalized.
            labels: [N] object class labels (int).
            return_obj_preds: If True, also return object label predictions.

        Returns:
            dict with rel_logits, pair_indices, etc.
        """
        N = visual_feats.size(0)
        device = visual_feats.device

        # Object encoding
        obj_feats = self.obj_encoder(visual_feats, labels)  # [N, H]

        # Object transformer context
        obj_feats = obj_feats.unsqueeze(0)  # [1, N, H]
        obj_feats = self.obj_transformer(obj_feats)
        obj_feats = self.obj_post(obj_feats).squeeze(0)  # [N, H]

        # Object classification (for SGCLS)
        obj_logits = None
        if return_obj_preds:
            obj_logits = self.obj_classifier(obj_feats)

        # Generate pair features
        pairs = generate_object_pairs(N, device)  # [P, 2]
        if pairs.numel() == 0:
            return {
                "rel_logits": visual_feats.new_zeros(0, self.num_predicates),
                "pair_indices": pairs,
                "sub_boxes": boxes.new_zeros(0, 4),
                "obj_boxes": boxes.new_zeros(0, 4),
                "obj_labels": labels,
            }

        pair_feats = self.pair_gen(obj_feats, boxes, pairs)  # [P, H]

        # Edge transformer context
        pair_feats = pair_feats.unsqueeze(0)  # [1, P, H]
        pair_feats = self.edge_transformer(pair_feats)
        pair_feats = self.edge_post(pair_feats).squeeze(0)  # [P, H]

        # Predicate classification
        rel_logits = self.pred_classifier(pair_feats)

        if self.freq_bias is not None:
            rel_logits = self.freq_bias(rel_logits)

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

def build_transformer_sgg(args) -> TransformerSGGModel:
    """Build Transformer SGG model from config args."""
    model = TransformerSGGModel(
        num_classes=getattr(args, 'entity_nums', 151),
        num_predicates=getattr(args, 'rel_nums', 51),
        visual_dim=getattr(args, 'visual_dim', 2048),
        hidden_dim=getattr(args, 'hidden_dim', 512),
        nhead=getattr(args, 'transformer_nhead', 8),
        num_encoder_layers=getattr(args, 'transformer_encoder_layers', 2),
        dim_feedforward=getattr(args, 'transformer_dim_feedforward', 2048),
        use_freq_bias=getattr(args, 'use_freq_bias', True),
        freq_bias_eps=getattr(args, 'freq_bias_eps', 1e-12),
        dropout=getattr(args, 'dropout', 0.1),
    )
    return model
